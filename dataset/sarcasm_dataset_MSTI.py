import os
import cv2
import numpy as np
import torch
import json
import pickle
from random import sample, shuffle
from transformers import BertTokenizer
from torch.utils.data import Dataset
from PIL import Image
from collections import Counter
from dataset.utils import pre_caption
from flair.data import Sentence


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 token_type_ids
                 ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.token_type_ids = token_type_ids

def sarcasm_collate(batch):
    text = {'input_ids':torch.stack([x[0]['input_ids'] for x in batch],dim=0),\
            'attention_mask':torch.stack([x[0]['attention_mask'] for x in batch],dim=0),\
            'token_type_ids':torch.stack([x[0]['token_type_ids'] for x in batch],dim=0)}
    image = np.array([(x[1].numpy()) for x in batch])
    crop_img_feature = [x[2] for x in batch]
    word_tag = torch.tensor([x[3] for x in batch],dtype=torch.long)
    crop_label = [x[4] for x in batch]
    feature_mask = [x[5] for x in batch]
    img_id = [x[6] for x in batch]
    return text,image,crop_img_feature,word_tag,crop_label,feature_mask,img_id
        
class sarcasm_MSTI_dataset(Dataset):
    def __init__(self, ann_file, feature_root,img_root, tokenizer,transformer,max_words=70, mode='train'):
        self.data= self.load_data(ann_file,feature_root,max_words,mode)
        self.transformer=transformer
        self.tokenizer=tokenizer
        self.image_root=img_root
        self.feature_root=feature_root
        self.max_words = max_words
        self.mode = mode
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        index   = index % len(self.data)
        data=self.data[index]
        img_id=data['img_id']
        text = data['sentence']
        word_tag = data['word_tag']
        crop_img_feature=data['crop_img_feature']
        feature_mask=data['feature_mask']
        crop_label=data['crop_label']

        img_path = os.path.join(self.image_root, img_id + '.jpg')
        image = Image.open(img_path).convert('RGB')   
        image = self.transformer(image) 
        return text, image,crop_img_feature, word_tag, crop_label,feature_mask,img_id
     
    def load_data(self,ann_file,feature_root,max_words,mode):
        print('calculating vocabulary...')
        datasplit, sentences, word_tags, img_id,crop_feature,feature_mask,cls_label = \
            self.load_sentence('IMGID',ann_file,feature_root,max_words,mode)
        data_list=[]
        for id,sentence_feature,word_tag,crop_img_feature,feature_mask,crop_label in zip(img_id,sentences,word_tags,crop_feature,feature_mask,cls_label):
            dict={}
            dict['img_id']=id
            dict['sentence']={'input_ids':torch.tensor(sentence_feature.input_ids,dtype=torch.long),
                              'attention_mask':torch.tensor(sentence_feature.input_mask,dtype=torch.long),
                              'token_type_ids':torch.tensor(sentence_feature.token_type_ids,dtype=torch.long)}
            dict['word_tag']=word_tag
            dict['crop_img_feature']=torch.cat(crop_img_feature,dim=0)
            dict['feature_mask']=torch.tensor(feature_mask)
            dict['crop_label']=crop_label
            data_list.append(dict)
        return data_list
    
    def handle_token(self,token,label,max_words,tokenizer,mode):
        label_map={'0':0,'B-sarcasm': 1, 'I-sarcasm': 2,'B-non_sarcasm':3,'I-non_sarcasm':4,
                    'O': 5,'X':6,'[CLS]':7,'[SEP]':8}
        #[O,B,I,X,[CLS],[SEP]]
        aux_label_map={'0':0,'B': 1, 'I': 2,'O': 3,'X':4,'[CLS]':5,'[SEP]':6}
        if len(token) >= max_words - 1:
            token = token[0:(max_words - 2)]
            label = label[:(max_words - 2)]
        new_token,label_map_id=[],[]
        new_token.append('[CLS]')
        label_map_id.append(label_map['[CLS]'])
        for t,l in zip(token,label):
            new_token.append(t)
            label_map_id.append(label_map[l])
        new_token.append('[SEP]')
        label_map_id.append(label_map['[SEP]'])

        while len(label_map_id)<max_words:
            new_token.append('[PAD]')
            label_map_id.append(0)
        input_ids=tokenizer.convert_tokens_to_ids(new_token)
        attention_mask=[0]*max_words
        attention_mask[:len(token)+2]=[1] * (len(token)+2)
        token_type_ids = [0] * len(input_ids)
        assert len(attention_mask)==len(input_ids)==len(label_map_id)
        return new_token,input_ids,attention_mask,label_map_id,token_type_ids
    
    def calculate_max_length(self,ann_file,tokenizer):
        sent_maxlen=0
        sentence = []
        with open(os.path.join(ann_file), 'r', encoding='utf-8') as file:
            last_line = ''
            for line in file:
                line = line.rstrip()
                if line == '':
                    sent_maxlen = max(sent_maxlen, len(sentence))
                    sentence = []
                else:
                    if 'IMGID' in line:
                            pass
                    else:
                        token=tokenizer.tokenize(line.split()[0])
                        sentence.extend(token)
        return sent_maxlen

    def load_sentence(self,IMAGEID, ann_file,feature_root,max_words,mode):
        """
        read the word from doc, and build sentence. every line contain a word and it's tag
        every sentence is split with a empty line. every sentence begain with an "IMGID:num"

        """
        img_id = []
        sentences = []
        sentence = []
        word_tag=[]
        word_tags=[]
        datasplit = []
        features=[]
        feature_masks=[]
        cls_label=[]
        input_len=[]
        word_maxlen,sent_maxlen=0,0
        tokenizer=BertTokenizer.from_pretrained('../ALBEF/pretrained/bert-base-uncased')
        max_words=min(512,self.calculate_max_length(ann_file,tokenizer))
        with open(os.path.join(ann_file), 'r', encoding='utf-8') as file:
            last_line = ''
            for line in file:
                line = line.rstrip()
                if line == '':
                    new_token,input_ids,attention_mask,new_word_tag,token_type_ids=\
                        self.handle_token(sentence,word_tag,max_words,tokenizer,mode)
        
                    token=InputFeatures(input_ids,attention_mask,token_type_ids)
                    sent_maxlen = max(sent_maxlen, len(sentence))
                    sentences.append(token)
                    word_tags.append(new_word_tag)
                    sentence = []
                    word_tag=[]
                    auxlabel=[]
                else:
                    if IMAGEID in line:
                        num = line[6:]
                        img_id.append(num)
                        if last_line != '':
                            print(num)
                    else:
                        if len(line.split()) == 1:
                            print(line)
                        token=tokenizer.tokenize(line.split()[0])
                        sentence.extend(token)
                        for m in range(len(token)):
                            if m==0:
                                if line.split()[1]=='1B-S':
                                    word_tag.append('B-sarcasm')
                                elif line.split()[1]=='1I-S':
                                    word_tag.append('I-sarcasm')
                                elif line.split()[1]=='B-S':
                                    word_tag.append('B-non_sarcasm')
                                elif line.split()[1]=='I-S':
                                    word_tag.append('I-non_sarcasm')
                                else:
                                    word_tag.append(line.split()[1])
                            else:
                                word_tag.append("X")
                        word_maxlen = max(word_maxlen, len(str(line.split()[0])))
                last_line = line
        crop_feature=open(feature_root, 'rb')
        crop_feature= pickle.load(crop_feature)
        max_features_length=-1
        for a in crop_feature:
            if 'class_name' in crop_feature[a].keys():
                max_features_length=max(max_features_length,len(crop_feature[a]['class_name']))
        for id in img_id:
            feature_mask=torch.zeros(max_features_length,dtype=torch.long)
            if 'features' in (crop_feature[str(id)+'.jpg']).keys() and 'class_name' in (crop_feature[str(id)+'.jpg']).keys():
                crop_img_feature = [x[:,0,:] for x in crop_feature[str(id)+'.jpg']['features']]
                class_label = [1 if x=='sarcasm' else 2 for x in crop_feature[str(id)+'.jpg']['class_name']]
            else:
                if features[0][0].dim()==2:
                    crop_img_feature=[torch.ones((1,768))]
                    class_label=[0]
            feature_masks.append(feature_mask)
            features.append(crop_img_feature)
            cls_label.append(class_label)
        datasplit.append(len(img_id))
        num_sentence = len(sentences)
        print("datasplit", datasplit)
        print('sent_maxlen', sent_maxlen)
        print('word_maxlen', word_maxlen)
        print('number sentence', num_sentence)
        print('number image', len(img_id))
        return [datasplit, sentences, word_tags, img_id,features,feature_masks,cls_label]



