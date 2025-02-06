import argparse
import os
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import pickle
import shutil
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import pandas as pd
from seqeval.metrics import classification_report as sen_classification_report
from models.model_SaSTI import MSTI as sarcasm
from transformers import BertTokenizer
from torch.utils.data.sampler import WeightedRandomSampler
from evaluate import evaluate as evaluate_sen,evaluate_each_class
from dataset.sarcasm_dataset_MSTI import sarcasm_collate
import matplotlib.pyplot as plt
import seaborn as sns
import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer
from sklearn import metrics
from torchcrf import CRF
from typing import Tuple
import torch
from torch import nn, Tensor
import warnings
import wandb
warnings.filterwarnings("ignore")
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from openpyxl import load_workbook


def load_tensor(text, image,crop_img_feature, word_tag, crop_label,feature_mask,device):
    text={'input_ids':text['input_ids'].to(device),\
          'attention_mask':text['attention_mask'].to(device),\
          'token_type_ids':text['token_type_ids'].to(device)}
    crop_img_feature,feature_mask,crop_label=padding(crop_img_feature,crop_label)
    word_tag=word_tag.to(device)

    crop_label=crop_label.to(device)

    feature_mask=feature_mask.to(device)
    image=torch.from_numpy(image).to(device)
    # image=image.cuda()
    crop_img_feature=crop_img_feature.to(device)
    
    return [text,image,crop_img_feature,word_tag, crop_label,feature_mask]


def padding(crop_img_feature,crop_label):
    crop_img_maxlength=max([x.shape[0] for x in crop_img_feature])
    temp_feature=[]
    feature_mask=[]
    feature_label=[]
    for i in range(len(crop_img_feature)):
        padding_feantures=torch.ones((crop_img_maxlength,crop_img_feature[0].shape[1]))
        padding_feantures[:crop_img_feature[i].shape[0],:]=crop_img_feature[i]      
        f_mask=torch.ones(crop_img_maxlength,dtype=torch.long)     
        if sum(crop_label[i]) == 0:     
            f_mask[0:]=0
        else:
            f_mask[crop_img_feature[i].shape[0]:]=0

        padding_label=torch.zeros(crop_img_maxlength,dtype=torch.long)
        padding_label[:len(crop_label[i])]=torch.tensor(crop_label[i],dtype=torch.long)
        feature_label.append(padding_label)
        feature_mask.append(f_mask)
        temp_feature.append(padding_feantures)
    crop_img_features=torch.stack(temp_feature,dim=0)
    feature_mask=torch.stack(feature_mask,dim=0)
    crop_label=torch.stack(feature_label,dim=0)
    
    return crop_img_features,feature_mask,crop_label

def train(model, data_loader, optimizer, epoch, warmup_steps, device, scheduler, config,args,crf_loss):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    img_ids=[]
    for i,(text, image,crop_img_feature, word_tag, crop_label,feature_mask,img_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if i !=0:
            continue
        text, image,crop_img_features, word_tag, crop_label,feature_mask\
            =load_tensor(text, image,crop_img_feature, word_tag, crop_label,feature_mask,device)
        b=image.shape[0]
        sentence_level_tag=torch.FloatTensor([1 if int(i)>2500 else 0 for i in img_id]).to(device).unsqueeze(-1)
        output = model(text, feature_mask,image,crop_img_features,img_id,\
                       args,'train',crop_label)
        try:
            beta = 0.5
            main_loss = - crf_loss(output['text_cls_feats'], word_tag, mask=text['attention_mask'].byte(), reduction='mean')
        except Exception as e:
            print(e)
            input()
        img_ids.extend(img_id)
        b,s,c=output['img_cls_feats'].shape
        img_feature=[]
        img_label=[]
        output['img_cls_feats'] = output['img_cls_feats'].contiguous().view(b*s,c)
        crop_label=crop_label.contiguous().view(-1)
        img_feature=[]
        img_label=[]
        for a in range(output['img_cls_feats'].shape[0]):
                if crop_label[a]==1 or crop_label[a]==2:
                    img_feature.append(output['img_cls_feats'][a,:])
                    img_label.append(crop_label[a]-1) 
        img_feature=torch.stack(img_feature,dim=0).to(device)
        crop_label=torch.stack(img_label,dim=0).to(device)
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([4,2])).float() ,
                                size_average=True).to(device)
        
        img_loss=criterion(img_feature,crop_label)
        criterion1 = nn.BCEWithLogitsLoss().to(device)
        
        sentence_level_loss=criterion1(output['sen_level_classification'],sentence_level_tag)
        loss = main_loss + img_loss+args.loss_alpha*sentence_level_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
               
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    

@torch.no_grad()
def evaluate(args,model, data_loader,device,epoch,crf):
    # test
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    labels_pred = []
    labels = []
    sent_lens = []
    header = 'Evaluation:'
    print_freq = 50
    tags= {'0':0,'B-sarcasm': 1, 'I-sarcasm': 2,'B-non_sarcasm': 3, 'I-non_sarcasm': 4,
                    'O': 5,'X':6,'[CLS]':7,'[SEP]':8}
    label_map={tags[key]:key for key in tags}
    pred=[]
    target=[]
    img_ids=[]
    text_id=[]
    text_pred=[]
    text_target=[]
    text_pred_idx=[]
    text_target_idx=[]
    for text, image,crop_img_feature, word_tag, crop_label,feature_mask,img_id in metric_logger.log_every(data_loader, print_freq, header):
        
        text, image,crop_img_features, word_tag, crop_label,feature_mask\
            =load_tensor(text, image,crop_img_feature, word_tag, crop_label,feature_mask,device)
        
        output = model(text, feature_mask,image,crop_img_features,img_id,\
                       args)                
        pre_sentence_label_index = crf.decode(output['text_cls_feats'],mask=text['attention_mask'].byte())
        labels_pred=pre_sentence_label_index
        labels=torch.tensor(word_tag).cpu().numpy()
        sen_len=torch.sum(text['attention_mask'],dim=1,dtype=torch.long).cpu().numpy()
        sent_lens.extend(sen_len)
        img_len=torch.sum(feature_mask,dim=1)
        b,s,c=output['img_cls_feats'].shape

        input_mask=text['attention_mask'].to('cpu').numpy()
        word_tag=word_tag.to('cpu').numpy()
        
        for i,(mask,id) in enumerate(zip(input_mask,img_id)):
            temp_1 = []
            temp_2 = []
            tmp1_idx = []
            tmp2_idx = []
            
            for j, m in enumerate(mask):
                if j == 0:#去除[CLS]
                    continue
                if m:#去除padding
                    if label_map[word_tag[i][j]] != "X" and label_map[word_tag[i][j]] != "[SEP]":
                        temp_2.append(label_map[word_tag[i][j]])
                        tmp2_idx.append(word_tag[i][j])
                        temp_1.append(label_map[labels_pred[i][j]])
                        tmp1_idx.append(labels_pred[i][j])
                        
                else:   
                    break
            tmp_text_idx=[img_id[i]]*len(tmp1_idx)
            text_id.append(tmp_text_idx)
            text_pred.append(temp_1)
            text_target.append(temp_2)
            text_pred_idx.append(tmp1_idx)
            text_target_idx.append(tmp2_idx)

        for a in range(b):
            ids=[img_id[a]]*output['img_cls_feats'][a][:img_len[a],:].shape[0]
            _,pred0=nn.functional.softmax(output['img_cls_feats'][a][:img_len[a],:]).max(1)
            t=crop_label[a][:img_len[a]]-1
            img_ids.extend(ids)
            pred.extend(pred0)
            target.extend(t)
        
    acc, f1, p, r= evaluate_sen(text_pred_idx,text_target_idx,tags,sen_len)
    sarcasm_f1, sarcasm_p, sarcasm_r=evaluate_each_class(text_pred_idx,text_target_idx,tags,sen_len,'sarcasm')
    non_sarcasm_f1, non_sarcasm_p, non_sarcasm_r=evaluate_each_class(text_pred_idx,text_target_idx,tags,sen_len,'non_sarcasm')
    text_report = sen_classification_report(text_target, text_pred, digits=4,output_dict=True)

    metric_logger.meters['text_f1_micro'].update(f1)
    metric_logger.meters['text_f1_macro'].update((sarcasm_f1+non_sarcasm_f1)/2)
    metric_logger.meters['text_f1_weighted'].update(text_report['weighted avg']['f1-score'])
        
    target=[x.cpu() for x in target]
    pred=[x.cpu() for x in pred]

    report = metrics.classification_report(target, pred, target_names=['sarcasm','non_sarcasm'],
                                               digits=4,output_dict=True)
    img_acc=metrics.accuracy_score(target, pred, normalize=True, sample_weight=None)
    metric_logger.meters['img_micro_f1'].update(img_acc)
    metric_logger.meters['img_macro_f1'].update(report['macro avg']['f1-score'])
    metric_logger.meters['img_weighted_f1'].update(report['weighted avg']['f1-score'])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())   
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
    
    
def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    datasets = create_dataset(args.dataset, config,tokenizer) 
    data=datasets[0].data
    weights = []
    for i in range(len(data)):
        data_dict=data[i]['crop_label']
        if 1 in data_dict:
            weights.append(5)
        else:
            weights.append(1)
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)         
    else:
        sampler = WeightedRandomSampler(weights,num_samples=len(datasets[0]), replacement=True)
        samplers = [sampler, None, None]
    
    train_loader, val_loader, test_loader = create_loader(datasets,samplers,batch_size=[config['batch_size']]*3,
                                              num_workers=[2,2,2],is_trains=[True,False,False], collate_fns=[sarcasm_collate,sarcasm_collate,sarcasm_collate])

    #### Model #### 
    print("Creating model")
    model = sarcasm(args.text_encoder,num_labels = config['num_of_tags'],visual_encoder=args.visual_encoder,local_config=config)
    crf_loss=CRF(config['num_of_tags'],batch_first=True).to(device)
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']    
        msg = model.load_state_dict(state_dict,strict=False)
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)

    model = model.to(device)  
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    
    print("Start training")
    start_time = time.time()
    best,img_best = 0,0
    best_epoch = 0
    best_text_acc,best_text_EM,best_text_macro_f1,best_text_weighted_f1=0,0,0,0
    best_img_acc,best_img_macro_f1,best_img_weighted_f1=0,0,0
    for epoch in range(0, max_epoch):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        train_stats = train(model, train_loader, optimizer,  epoch, warmup_steps, device, lr_scheduler, config,args,crf_loss)   
        val_stats = evaluate(args,model, val_loader, device,epoch,crf_loss)
        test_stats= evaluate(args,model, test_loader,device,epoch,crf_loss)

        if utils.is_main_process():  
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                        }
            result= (float(val_stats['text_f1_macro']) + float(val_stats['img_macro_f1']))/2           
            if result>best:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch)) 
                best = result
                best_epoch = epoch
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
            
        lr_scheduler.step(epoch+warmup_steps+1)  
    
    for item in os.listdir(args.output_dir):
        if os.path.isfile(os.path.join(args.output_dir, item)):  
            if item!='checkpoint_%02d.pth'%best_epoch and item.startswith('checkpoint_'):
                os.remove(os.path.join(args.output_dir, item))          
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    if utils.is_main_process():   
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)             

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/sarcasm.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--load_path', default='vilt_200k_mlm_itm.ckpt')
    parser.add_argument('--output_dir', default='output/sarcasm')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--finetune_first', default=False, type=bool)
    parser.add_argument('--text_encoder', default='../ALBEF/pretrained/bert-base-uncased')
    parser.add_argument('--visual_encoder', default='./pretrained/vit-base-patch32-384')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--dataset', default='MSTI', type=str)
    parser.add_argument('--loss_alpha', default=0.438, type=float)
    parser.add_argument('--cls_beta', default=0.911, type=float)
    args = parser.parse_args()
    
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    main(args, config)
