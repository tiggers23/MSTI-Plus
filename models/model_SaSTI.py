from functools import partial
from models.vit import VisionTransformer
# from models.xbert import BertConfig, BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.transformerEncoder import BertAttention
import torch
from torch import nn
from transformers import ViTModel,BertModel
from queue import Queue
import torch.nn.functional as F
import numpy as np
import math

class Config(object):
    hidden_size = 768
    num_attention_heads = 12
    attention_probs_dropout_prob = 0.5
    hidden_dropout_prob = 0.5

class MSTI(nn.Module):
    """Coupled Cross-Modal Attention BERT model for token-level classification with CRF on top.
    """
    def __init__(self, text_encoder, num_labels=2,visual_encoder=None,local_config=None):
        super(MSTI, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(text_encoder)
        self.hidden_size=Config.hidden_size
        self.classifier = nn.Linear(self.hidden_size * 2, num_labels)
        self.img_classifier = nn.Linear(self.hidden_size, local_config['class_num'])
        self.visual_encoder=ViTModel.from_pretrained(visual_encoder)
        self.txt_sarcasm_cls_token=torch.zeros(1,self.hidden_size*2)
        self.txt_non_sarcasm_cls_token=torch.zeros(1,self.hidden_size*2)
        self.img_sarcasm_cls_token=torch.zeros(1,self.hidden_size)
        self.img_non_sarcasm_cls_token=torch.zeros(1,self.hidden_size)
        self.semantics_memory=torch.zeros(1,self.hidden_size*2)
        
        self.flag=False
        self.sentence_level_classifier = nn.Sequential(
            nn.Linear(self.hidden_size*2,self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size,256),
            nn.Tanh(),
            nn.Linear(256,64),
            nn.Tanh(),
            nn.Linear(64,1)
        )
        self.text_down = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.img_down = nn.Linear(self.hidden_size, self.hidden_size)
        ###glu
        self.text_glu=nn.Sequential(
            nn.Linear(self.hidden_size*2,self.hidden_size*2),
            nn.Tanh(),
            nn.Linear(self.hidden_size*2,self.hidden_size*2)
        )
        self.img_glu=nn.Sequential(
            nn.Linear(self.hidden_size,self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size,self.hidden_size)
        )

        self.text_trans=nn.Sequential(
            nn.Linear(self.hidden_size*2,self.hidden_size*2),
            nn.Tanh(),
            nn.Linear(self.hidden_size*2,self.hidden_size*2)
        )
        self.img_trans=nn.Sequential(
            nn.Linear(self.hidden_size*2,self.hidden_size*2),
            nn.Tanh(),
            nn.Linear(self.hidden_size*2,self.hidden_size)
        )
        self.self_att = BertAttention(config=Config)
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                            num_layers=1, bidirectional=True)

    def memory(self,features,text,img,stage,img_id,args):
        self.add_txt_cls_token = self.text_glu(text)  #768*2
        self.add_img_cls_token = self.img_glu(img) #768
        with torch.no_grad():
            sen_level_classification0=None
            sarcasm_memory = []
            for id,token in zip(img_id,features):
                id=int(id)
                if id >= 2500:
                    sarcasm_memory.append(token)
            if len(sarcasm_memory)==0:
                sarcasm_memory=torch.zeros_like(self.semantics_memory)
            else:
                sarcasm_memory=torch.stack(sarcasm_memory).mean(0,keepdim=True)
            if not self.flag:
                self.semantics_memory = sarcasm_memory
                self.flag=True
            elif stage is not None:
                self.semantics_memory = (1-args.cls_beta)*self.semantics_memory + args.cls_beta*sarcasm_memory
            
            sarcasm_text_COS = nn.CosineSimilarity(dim=-1, eps=1e-6)
            sarcasm_img_COS  = nn.CosineSimilarity(dim=-1, eps=1e-6)

            img_feature = self.img_trans(self.semantics_memory)
            text_feature = self.text_trans(self.semantics_memory)
            img_sim=sarcasm_img_COS(img,img_feature).unsqueeze(-1)
            text_sims = sarcasm_text_COS(text,text_feature).unsqueeze(-1)
            return img_sim,text_sims,self.add_txt_cls_token,self.add_img_cls_token

    def forward(self, text, feature_mask,image,crop_img_feature,img_id=None,args=None,stage=None,crop_label=None):
        sequence_output = self.bert(text['input_ids'], token_type_ids=text['token_type_ids'], attention_mask=text['attention_mask'],
                                       ).last_hidden_state
        vis_embed_map = self.visual_encoder(image).last_hidden_state  
        img_mask = torch.ones((vis_embed_map.shape[0],vis_embed_map.shape[1]),dtype=torch.long).to(next(self.parameters()).device)
        img_len=vis_embed_map.shape[1]
        vis_embed_map=torch.cat((vis_embed_map,crop_img_feature),dim=1)

        img_mask=torch.cat((img_mask,feature_mask),dim=1)
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)
        extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0

        extended_attention_mask=torch.cat((text['attention_mask'],img_mask),dim=1).unsqueeze(1).unsqueeze(2).to(dtype=next(self.parameters()).dtype)
        extended_attention_mask=(1.0 - extended_attention_mask) * -10000.0

        features=torch.cat((sequence_output,vis_embed_map),dim=1)
        features = self.self_att(features, extended_attention_mask)

        text_features=features[:,:sequence_output.size(1),:]
        img_features=features[:,sequence_output.size(1):,:]


        text_features,_=self.lstm(text_features)

        final_text_output=self.text_down(text_features[:,0,:])
        final_img_output=self.img_down(img_features[:,0,:])
        text_merge_img_layer = torch.cat((final_text_output,final_img_output),dim=1)

        img_sim,text_sims,self.add_txt_cls_token,self.add_img_cls_token =\
            self.memory(text_merge_img_layer,text_features,img_features,stage,img_id,args)      
        
        final_output=torch.add(text_features,torch.mul(text_sims,self.add_txt_cls_token)) 
        final_img_feats=torch.add(img_features,torch.mul(img_sim, self.add_img_cls_token))
        
        sen_level_classification0=self.sentence_level_classifier(text_merge_img_layer)
        bert_feats = self.classifier(final_output)
        final_img_feats=self.img_classifier(final_img_feats)
    
        return {'text_cls_feats':bert_feats,
                'img_cls_feats':final_img_feats[:,img_len:,:],
                'sen_level_classification':sen_level_classification0
                }
