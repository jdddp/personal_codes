import os
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from PIL import Image
import time
import copy
import numpy as np
from torchvision import transforms

def default_loader(imgPath):
    with open(imgPath, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class MyDataset(Dataset): # 定义自己的数据类
    def __init__(self, dataset_dir, txtPath, mode = '', data_transforms=None, loader = default_loader):        
        img_list = []
        label_list = []
        with open(txtPath, 'r') as f:
            for line in f:
                items= line.strip().split('\t')
                img_list.append(os.path.join(dataset_dir, items[0]))
                label_list.append(int(items[1]))
        self.imgs = img_list
        self.labels = label_list
        self.data_tranforms = data_transforms
        self.loader = loader
        self.mode = mode
 
    def __len__(self):
        return len(self.imgs)
 
    def __getitem__(self, item):
        img_name = self.imgs[item]
        label = self.labels[item]
        img = self.loader(img_name)
        # img=np.array(img)
 
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

