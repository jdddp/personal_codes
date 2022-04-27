import os
import os.path as osp
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from PIL import Image
import time
import random
import math
import numpy as np
import collections
from torchvision import transforms

def default_loader(imgPath):
    with open(imgPath, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')




class MyDataset(Dataset): # 定义自己的数据类
    '''sample_type
    balance:each cate has equal prob to be sampled;
    square:math.sqrt(cate_nums)/math.sqrt(all_nums), which can alleviate imbalance
    '''
    def __init__(self, dataset_dir, txtPath, mode = '', data_transforms=None, weight_sample=False, sample_type='balance',loader = default_loader):        
        img_list = []
        label_list = []

        #For weight_sample
        self.label2imglst=collections.defaultdict(list)
        self.label2nums=collections.defaultdict(int)
        self.sample_type=sample_type
        self.weight_sample=weight_sample

        with open(txtPath, 'r') as f:
            for line in f:
                items= line.strip().split('\t')
                img_list.append(os.path.join(dataset_dir, items[0]))
                label_list.append(int(items[1]))
                if self.weight_sample:
                    self.label2imglst[int(items[1])].append(osp.join(dataset_dir, items[0]))
                    self.label2nums[int(items[1])]+=1
        
        self.imgs = img_list
        self.labels = label_list
        self.data_tranforms = data_transforms
        self.loader = loader
        self.mode = mode


       #prob distribution of each cate
        if self.weight_sample and self.mode=='train':
            assert self.sample_type in ['balance', 'square']

            self.label2imglst=dict(sorted(self.label2imglst.items(),key=lambda x:x[0]))
            self.label2nums=dict(sorted(self.label2nums.items(),key=lambda x:x[0]))

            self.prob_lst=self.get_prob_distribution(self.label2nums)

    def get_prob_distribution(self, label2num):
        # import pdb
        # pdb.set_trace()
        numlst=[num for _,num in label2num.items()]
        if self.sample_type=='balance':              
            self.prob_lst=np.array([1/len(numlst) for _ in numlst])
        elif self.sample_type=='square':
            numlst=[math.sqrt(num) for num in numlst]
            self.prob_lst=np.array([num/sum(numlst) for num in numlst])
 
    def __len__(self):
        return len(self.imgs)
 
    def __getitem__(self, item):
        if not self.weight_sample:
            img_name = self.imgs[item]
            label = self.labels[item]
            img = self.loader(img_name)
            # img=np.array(img)
    
            
        else:
            label_lst=list(self.label2nums.keys())
            sample_class=np.random.choice(np.array(label_lst), p=self.prob_lst)
            label=int(sample_class)
            img_name=random.choice(self.label2imglst[label])
            img=self.loader(img_name)
        
        if self.data_tranforms is not None:
            try:
                img = self.data_tranforms[self.mode](img)
            except Exception as e:
                print(e)
                # print("Cannot transform image: {}".format(img_name))
                
        return img, label






class InferDataset(Dataset): # 定义自己的数据类
    def __init__(self, img_list, mode = 'test', data_transforms=None, loader = default_loader):        

        self.imgs = img_list
        self.data_tranforms = data_transforms
        self.loader = loader
        self.mode = mode
 
    def __len__(self):
        return len(self.imgs)
 
    def __getitem__(self, item):
        img_name = self.imgs[item]
        img = self.loader(img_name)
 
        if self.data_tranforms is not None:
            try:
                img = self.data_tranforms(img)
            except Exception as e:
                print(e)
                # print("Cannot transform image: {}".format(img_name))

        return img, img_name

