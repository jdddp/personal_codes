import os, sys
import os.path as osp
import torch
import torch.nn as nn
from torchvision import models,transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

def form_conversion(img):
    data_transform=transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    return data_transform(img).unsqueeze(0)

def img_loader(imgPath):
    with open(imgPath, 'rb') as f:
        with Image.open(f) as img:
            return form_conversion(img.convert('RGB'))

def fuse_model(model):
    '''set require_grad as False
    '''
    # if feature_learning:
    for param in model.parameters():
        param.requires_grad = False


#C:\Users\jdddp/.cache\torch\hub\checkpoints\resnet101-5d3b4d8f.pth
#/home/linke/.cache/torch/hub/checkpoints/resnet101-5d3b4d8f.pth
def feature_exactor():
    deepCNN = models.resnet101(pretrained=True)
    deepCNN.fc=nn.Sequential()
    fuse_model(deepCNN)
    return deepCNN

def get_feature(img_dir, txt_path, npy_path,ifCuda=True):
    device = torch.device('cuda:0')

    txtFile=open(txt_path, 'a', encoding='utf-8')
    deepCnn=feature_exactor()
    if ifCuda:
        deepCnn.to(device)

    for i,imgPath in tqdm(enumerate(os.listdir(img_dir))):
        txtFile.write(osp.join(img_dir,imgPath)+'\n')
        img=img_loader(osp.join(img_dir,imgPath))
        if ifCuda:
            img=img.to(device)
        feature=deepCnn(img).cpu().numpy()

        if i==0:
            ans_feat=feature
        else:
            ans_feat=np.vstack((ans_feat,feature))
    np.save(npy_path, ans_feat)
    return ans_feat

if __name__=='__main__':
    if len(sys.argv)>1:
        func=getattr(sys.modules[__name__], sys.argv[1])
        func(*(sys.argv[2:]))
    else:
        print('wrong!')

#python path/to/dcnn.py get_feature *argv

