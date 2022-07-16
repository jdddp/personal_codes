from genericpath import exists
import os,sys
import os.path as osp

from torch import classes
sys.path.append('/home/linke/codes/localCodes/mmdetection')
from mmdet.apis import init_detector, inference_detector
import mmcv
import glob
import pdb
import numpy as np
import json
import collections
import math
import cv2
import time
import torch
import torch.nn as nn
from torchvision import datasets, models
from PIL import Image, ImageFont, ImageDraw
from torchvision import transforms



from infer_slide import *

def get_model(config_file_det, weight_det, device):
    #initialize det_model
    det_model=init_detector(config_file_det,weight_det, device=device)
    return det_model

def get_part_img(img, dct, ratio=1.5):
    #crop for cls or seg
    ratio=float(ratio)

    h_o,w_o,_=img.shape
    x1,y1,w,h=dct['bbox']
    xc=float(w)/2+x1
    yc=float(h)/2+y1

    #越界判断
    x1_n=math.floor(max(0,xc-(ratio/2)*w))
    y1_n=math.floor(max(0,yc-(ratio/2)*h))

    x2_n=math.ceil(min(w_o,xc+(ratio/2)*w))
    y2_n=math.ceil(min(h_o,yc+(ratio/2)*h))
    # print(y1_n,y2_n,x1_n,x2_n)
    cropped_img=img[y1_n:y2_n,x1_n:x2_n]
    arg_tp=(xc+(ratio/2)*w, yc+(ratio/2)*h)
    return cropped_img, arg_tp

def convert_bbox(box_lst,arg_tp, ratio=1.5):
    x1,y1,w,h=box_lst
    xc=float(w)/2+x1
    yc=float(h)/2+y1

    #越界判断
    x1_n=math.floor(max(0,xc-(ratio/2)*w))
    y1_n=math.floor(max(0,yc-(ratio/2)*h))

    x2_n=math.ceil(min(arg_tp[0],xc+(ratio/2)*w))
    y2_n=math.ceil(min(arg_tp[1],yc+(ratio/2)*h))
    return (x1_n, y1_n, x2_n-x1_n, y2_n-y1_n)


def format_img_clsmodel(np_img):
    data_transforms=transforms.Compose([
            transforms.Resize(256), 

            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    np_img=Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
    np_img=torch.unsqueeze(data_transforms(np_img),dim=0)
    return np_img


def classify_part(part_img, cls_model, device='cuda:0'):
    part_img=format_img_clsmodel(part_img).to(device)
    output=cls_model(part_img)
    output=int(output.argmax())
    return output
    
def draw_imgs(np_img, dct_lst_other, dct_lst_target,color_other, color_target, goal_path, arg_tp):
    '''dirpath/imgs -> img_data
    dirpath/usual.json -> usualJson of img_data
    '''
    # for imgname, dctLst in usualJson.items():
        # img=cv2.imread(osp.join(imgDir, imgname))
    source_img=Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))

    draw=ImageDraw.Draw(source_img)

    if len(dct_lst_other)>0:
        for sgDct in dct_lst_other:
            label=sgDct['category'] 
            
            bbox=sgDct['bbox']
            bbox=convert_bbox(bbox, arg_tp)
            # color=tuple(np.random.randint(256,size=3))
            draw.rectangle(((bbox[0], bbox[1]), (bbox[2] + bbox[0], bbox[3] + bbox[1])),width=4,outline =color_other)
            if 'score' in sgDct:
                score=sgDct['score']
                draw.text((bbox[0], bbox[1]), '{}_{:.2}'.format(label,score),fill=color_other, font=ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 20))
            else:
                draw.text((bbox[0], bbox[1]), label,fill=color_other, font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18))
    if len(dct_lst_target)>0:
        for sgDct in dct_lst_target:
            # if 'flag' not in sgDct or sgDct['flag']==0:
            #     label=sgDct['category'] 
            # else:
            #     label=sgDct['category_change']
            label=sgDct['category'] 
            bbox=sgDct['bbox']
            bbox=convert_bbox(bbox, arg_tp)

            # color=tuple(np.random.randint(256,size=3))
            draw.rectangle(((bbox[0], bbox[1]), (bbox[2] + bbox[0], bbox[3] + bbox[1])),width=4,outline =color_target)
            if 'score' in sgDct:
                score=sgDct['score']
                draw.text((bbox[0], bbox[1]), '{}_{:.2}'.format(label,score),fill=color_target, font=ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 20))
            else:
                draw.text((bbox[0], bbox[1]), label,fill=color_target, font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18))
    source_img=source_img.convert('RGB')
    source_img.save(goal_path)

#/home/pikapika/codes/gitgit/personal_codes/classify_tool/projects/influid_v1/output/resnet18/best_model.pkl

def infer_main(config_file_det, checkpoint_file_det, weight_cls, img_dir, dst_dir, det_thre=0.7,scale=640,device='cuda:0'):
    '''
    config_file: configFile of mmdetection
    checkpoint_firl: path of weight
    imgDir: dirpath of imgs
    predJsonPath: pred ans in usual_format(.json)
    thre: threshold for bboxes
    scale: slide stride, (w,h can be set separately in infer_sg)
    '''
    t_s=time.time()
    os.makedirs(dst_dir, exist_ok=True)
    class_names=getClasses()

    det_thre=float(det_thre)
    det_model=get_model(config_file_det, checkpoint_file_det,device)

    #initialize cls model
    cls_model=models.resnet18(pretrained=False)
    num_ftrs = cls_model.fc.in_features
    cls_model.fc = nn.Linear(num_ftrs, len(class_names))
    cls_model.load_state_dict(torch.load(weight_cls))
    cls_model.to(device)
    cls_model.eval()


    # target_color=tuple(np.random.randint(256,size=3))
    target_color=(77, 60, 144)
    # other_color=tuple(np.random.randint(256,size=3))
    other_color=(196, 166, 105)
    # print(target_color)
    # print(other_color)

    # pdb.set_trace()
    # for i,imgname in enumerate(os.listdir(img_dir)):
    for j in range(137):
        # print(str(i)+'.jpg')
        imgname=str(j)+'.jpg'

        img_path=osp.join(img_dir, imgname)
        if j==0:
            dct_lst_other,dct_lst_target, img=infer_sg(img_path, det_model, scale, det_thre)
        else:
            img=mmcv.imread(img_path)

        #cls or seg
        for i,dct in enumerate(dct_lst_target):
            src_cate=dct['category']
            #(x_max, y_max)
            part_img, arg_tp=get_part_img(img, dct)
            new_cate=class_names[classify_part(part_img, cls_model)]
            # if new_cate!=src_cate:
            # dct['category']=new_cate
            if new_cate!='l_v_m':
                # dct['category_change']=src_cate+'>>>>>'+new_cate
                dct['category']='l_v_m'+'>>>>>'+new_cate

            #     dct['flag']=1
            # else:
            #     dct['flag']=0
            dct_lst_target[i]=dct
            # print(dct_lst_target[i])
        path_n=osp.join(dst_dir, imgname)
        draw_imgs(img, dct_lst_other,dct_lst_target, other_color,target_color,path_n, arg_tp)
    print(time.time()-t_s)

if __name__=='__main__':
    if len(sys.argv)>1:
        func=getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('input wrong')
        

#/home/pikapika/codes/useless0716


