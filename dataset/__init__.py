import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.caption_dataset import re_train_dataset, re_eval_dataset, pretrain_dataset
from dataset.nlvr_dataset import nlvr_dataset
from dataset.ve_dataset import ve_dataset
from dataset.vqa_dataset import vqa_dataset
from dataset.grounding_dataset import grounding_dataset
from dataset.sarcasm_dataset_ner import sarcasm_dataset
from dataset.sarcasm_dataset_memory_ner import sarcasm_memory_dataset
from dataset.sarcasm_dataset_foracl_ner import sarcasm_dataset_foracl
from dataset.sarcasm_dataset_foracl_ner_1 import sarcasm_dataset_foracl as sarcasm_dataset_ner_with_acl_data
from dataset.sarcasm_dataset_MMSD import sarcasm_dataset_MMSD
from dataset.sarcasm_dataset_foracl_MMSD import sarcasm_dataset_mmsd_foracl
from dataset.sarcasm_dataset_vilt import sarcasm_dataset_vilt
from dataset.sarcasm_dataset_foracl_vilt import sarcasm_dataset_vilt_foracl
from dataset.sarcasm_dataset_MSTI import sarcasm_MSTI_dataset
from dataset.sarcasm_dataset_twitter import sarcasm_twitter_dataset
from dataset.sarcasm_dataset_foracl_MSTI import sarcasm_dataset_foracl_MSTI
from dataset.sarcasm_dataset_resnet import sarcasm_Resnet_dataset
from dataset.sarcasm_dataset_foracl_MSTI_1 import sarcasm_dataset_foracl_MSTI as sarcasm_dataset_MSTI_with_acl_data
from dataset.sarcasm_dataset_MSTI_attention_visual import sarcasm_MSTI_dataset as sarcasm_dataset_attention_visual
from dataset.sarcasm_dataset_MI import sarcasm_dataset_MI
from dataset.MSD_dataset import MSD_dataset

from dataset.randaugment import RandomAugment


import numpy as np

def create_dataset(dataset, config,tokenizer,vit_processor = None):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    # print(config['image_res'])
    # input()
    pretrain_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.2, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])    
    train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])  
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])   
    
    if dataset=='pretrain':
        dataset = pretrain_dataset(config['train_file'], pretrain_transform)                  
        return dataset      
               
    elif dataset=='re':          
        train_dataset = re_train_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = re_eval_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset, test_dataset   

    elif dataset=='vqa': 
        train_dataset = vqa_dataset(config['train_file'], train_transform, config['vqa_root'], config['vg_root'], split='train') 
        vqa_test_dataset = vqa_dataset(config['test_file'], test_transform, config['vqa_root'], config['vg_root'], split='test', answer_list=config['answer_list'])       
        return train_dataset, vqa_test_dataset

    elif dataset=='nlvr':   
        train_dataset = nlvr_dataset(config['train_file'], train_transform, config['image_root'])  
        val_dataset = nlvr_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = nlvr_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset, test_dataset        
               
    elif dataset=='ve':   
        train_dataset = ve_dataset(config['train_file'], train_transform, config['image_root'])  
        val_dataset = ve_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = ve_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset, test_dataset     
    
    elif dataset=='grounding':
        train_transform = transforms.Compose([                        
                transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])         
        train_dataset = grounding_dataset(config['train_file'], train_transform, config['image_root'], mode='train')       
        test_dataset = grounding_dataset(config['test_file'], test_transform, config['image_root'], mode='test')             
        return train_dataset, test_dataset    
    elif dataset=='MSTI':          
        train_dataset = sarcasm_MSTI_dataset(config['train_file'], config['feature_root'],config['image_root'], tokenizer,train_transform)
        val_dataset = sarcasm_MSTI_dataset(config['val_file'], config['feature_root'],config['image_root'], tokenizer,test_transform)  
        test_dataset = sarcasm_MSTI_dataset(config['test_file'], config['feature_root'],config['image_root'], tokenizer,test_transform)                
        return train_dataset, val_dataset, test_dataset
    

def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders  



def sarcasm_collate(batch):
    # print(batch)
    #x_flair, image,crop_img_feature, word_tag, crop_label
    x_flair = [x[0] for x in batch]
    # print(x[1] for x in batch)
    image = np.array([(x[1].numpy()) for x in batch])
    crop_img_feature = [x[2] for x in batch]
    word_tag = np.array([x[3] for x in batch])
    crop_label = [x[4] for x in batch]
    
    bool_mask = word_tag == 0
    mask = 1 - bool_mask.astype(int)

    # index of first 0 in each row, if no zero then idx = -1
    zero_indices = np.where(bool_mask.any(1), bool_mask.argmax(1), -1).astype(int)
    input_len = np.zeros(len(batch))
    for i in range(len(batch)):
        if zero_indices[i] == -1:
            input_len[i] = len(word_tag[i])
        else:
            input_len[i] = zero_indices[i]
    sorted_input_arg = np.argsort(-input_len)

    # Sort everything according to the sequence length
    x_flair = sorted(x_flair, key=lambda i: len(i), reverse=True)
    crop_img_feature = [crop_img_feature[i] for i in sorted_input_arg]
    word_tag = word_tag[sorted_input_arg]
    mask = mask[sorted_input_arg]
    input_len = input_len[sorted_input_arg]
    image = image[sorted_input_arg]
    crop_label=[crop_label[i] for i in sorted_input_arg]

    max_seq_len = int(input_len[0])
    # print(x_flair,max_seq_len)

    trunc_x_flair = []
    trunc_word_tag = np.zeros((len(batch), max_seq_len))
    trunc_mask = np.zeros((len(batch), max_seq_len))
    for i in range(len(batch)):
        trunc_x_flair.append(x_flair[i])
        trunc_word_tag[i] = word_tag[i, :max_seq_len]
        trunc_mask[i] = mask[i, :max_seq_len]

        # return to_tensor(trunc_x).long(), to_tensor(obj_x), to_tensor(trunc_y).long(), to_tensor(trunc_mask).long(), \
        #        to_tensor(input_len).int(), to_tensor(ifpairs).long()
    return trunc_x_flair, image,crop_img_feature,trunc_word_tag, crop_label,input_len, trunc_mask

def sarcasm_collate1(batch):
    text = {'input_ids':torch.stack([x[0]['input_ids'] for x in batch],dim=0),\
            'attention_mask':torch.stack([x[0]['attention_mask'] for x in batch],dim=0)}
    # print(x[1] for x in batch)
    image = np.array([(x[1].numpy()) for x in batch])
    crop_img_feature = [x[2] for x in batch]
    # for x in batch:
    #     print(x)
    #     input()
    word_tag = torch.tensor([x[3] for x in batch],dtype=torch.long)
    crop_label = [x[4] for x in batch]
    feature_mask = [x[5] for x in batch]
    img_id = [x[6] for x in batch]
    origin_id = [x[7] for x in batch]
    return text,image,crop_img_feature,word_tag,crop_label,feature_mask,img_id,origin_id

def sarcasm_collate2(batch):
    text = {'input_ids':torch.stack([x[0]['input_ids'] for x in batch],dim=0),\
            'attention_mask':torch.stack([x[0]['attention_mask'] for x in batch],dim=0)}
    # print(x[1] for x in batch)
    image = np.array([(x[1].numpy()) for x in batch])
    crop_img_feature = [x[2] for x in batch]
    # for x in batch:
    #     print(x)
    #     input()
    word_tag = torch.tensor([x[3] for x in batch],dtype=torch.long)
    crop_label = [x[4] for x in batch]
    feature_mask = [x[5] for x in batch]
    img_id = [x[6] for x in batch]
    return text,image,crop_img_feature,word_tag,crop_label,feature_mask,img_id