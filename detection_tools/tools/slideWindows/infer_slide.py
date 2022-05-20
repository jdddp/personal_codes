'''based on mmdetection
For other model, just need to change get_model and line 114-138; just make sure pred of model in format of usualJson
'''
import os,sys
import os.path as osp
sys.path.append('/home/linke/codes/localCodes/mmdetection')
from mmdet.apis import init_detector, inference_detector
import mmcv
import glob
import pdb
import numpy as np
import json
import collections
import math
import time

def getClasses():
    class_names=('fluid_inclusions',)
    return class_names

def get_model(config_file, weight, device):
    model=init_detector(config_file,weight, device=device)
    return model

def _iou(list1,list2,threshold=0.2):
    w1,h1=list1[2:]
    w2,h2=list2[2:]
    xl=max(list1[0],list2[0])
    yl=max(list1[1],list2[1])
    xr=min(list1[0]+w1,list2[0]+w2)
    yr=min(list1[1]+h1,list2[1]+h2)


    s_i=max((xr-xl),0)*max((yr-yl), 0)
    s_o=w1*h1+w2*h2-s_i
    return s_i/s_o > threshold

def __deduplicate(dct_lst):
    x1_n,y1_n,x2_n,y2_n=3000, 3000, 0, 0
    score_n=0
    for sgDct in dct_lst:
        score=sgDct['score']
        x1,y1,x2,y2=sgDct['bbox'][0], sgDct['bbox'][1], \
            sgDct['bbox'][2]+sgDct['bbox'][0], \
            sgDct['bbox'][1]+sgDct['bbox'][3]
        x1_n=min(x1_n, x1)
        y1_n=min(y1_n, y1)
        x2_n=max(x2_n, x2)
        y2_n=max(y2_n, y2)
        score_n=max(score_n, score)

    new_dct={
        'category': dct_lst[0]['category'],
        'bbox': [x1_n, y1_n, x2_n-x1_n,y2_n-y1_n]
    }

    return new_dct

def deduplicate_bboxes(dct_lst):
    if len(dct_lst)<=1:
        return dct_lst
    ansLst=[]
    #队列来做把
    useless_lst=[]
    for i in range(len(dct_lst)):
        if i in set(useless_lst):
            continue
        flag=0
        temp_lst=[]
        for j in range(i+1,len(dct_lst)):
            if _iou(dct_lst[i]['bbox'], dct_lst[j]['bbox']):
                flag=1
                temp_lst.append(dct_lst[j])
                useless_lst.append(j)
        if flag==0:
            ansLst.append(dct_lst[i])
        else:
            ansLst.append(__deduplicate(temp_lst))
    return ansLst



def infer_sg(img_path, model, scale=640, thre=0.5):
    class_names=getClasses()

    img=mmcv.imread(img_path)
    h,w,_ =img.shape

    #num in row and col
    w_scale=scale
    h_scale=scale
    row_n=w//w_scale if w%w_scale==0 else w//w_scale+1
    col_n=h//h_scale if h%h_scale==0 else h//h_scale+1

    w_s,h_s=w//row_n, h//col_n
    w_lst=[]
    h_lst=[]
    for i in range(row_n):
        w_lst.append(i*w_s)
    for i in range(col_n):
        h_lst.append(i*h_s)
    print('图片横向 {} 个；纵向 {} 个'.format(row_n, col_n))

    dct_lst=[]
    for i in range(col_n):
        for j in range(row_n):
            h_need=min(i*h_s+h_scale, h)
            w_need=min(j*w_s+w_scale, w)
            #此处应该有个遍历不同ij对应的wh变化
            w_change=w_lst[j]
            h_change=h_lst[i]

            img_temp=img[i*h_s:h_need, j*w_s:w_need,:]
            #
            bbox_result=inference_detector(model, img_temp)

            bboxes=np.vstack(bbox_result)
            labels=[
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels=np.concatenate(labels)

            scores=bboxes[:,-1]
            inds=scores>thre
            bboxes=bboxes[inds, :]
            labels=labels[inds]
            for idx, bbox_score in enumerate(bboxes):
                label=class_names[labels[idx]]
                # temp
                dct_lst.append(
                    {
                        'category':label,
                        'score': float(bbox_score[-1]),
                        'bbox': [math.floor(bbox_score[0]+w_change),math.floor(bbox_score[1]+h_change), math.ceil(bbox_score[2]-bbox_score[0]), math.ceil(bbox_score[3]-bbox_score[1])]
                    }
                )
            #
    #dct_lst need deduplication
    dct_lst=deduplicate_bboxes(dct_lst)
    return dct_lst

def inferSlide(config_file, checkpoint_file,imgDir, predJsonPath,thre=0.5,scale=640,device='cuda:0'):
    '''
    config_file: configFile of mmdetection
    checkpoint_firl: path of weight
    imgDir: dirpath of imgs
    predJsonPath: pred ans in usual_format(.json)
    thre: threshold for bboxes
    scale: slide stride, (w,h can be set separately in infer_sg)
    '''
    model=get_model(config_file, checkpoint_file,device)

    # img_path='/home/linke/codes/localCodes/dataset/lh_test_split/imgs_test/2-imgE.jpg'

    ansDct={}
    for imgname in os.listdir(img_dir):

        img_path=osp.join(img_dir, imgname)
        ansDct[imgname]=infer_sg(img_path, model)

    with open(predJsonPath, 'w', encoding='utf-8')as f:
        f.write(json.dumps(ansDct))

if __name__=='__main__':
    if len(sys.argv)>1:
        func=getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('input wrong')
# config_file='/home/linke/codes/localCodes/mmdetection/configs/a_projects/tjhv1_yolov1.py'
# checkpoint_file='/home/linke/codes/localCodes/mmdetection/output/lh_test/best_bbox_mAP_epoch_17.pth'
# img_dir='/home/linke/codes/localCodes/dataset/lh_test/imgs'
# predJsonPath='/home/linke/codes/localCodes/dataset/lh_test/usual_pred.json'
# device='cuda:0'
# scale=640
# thre=0.5

#python path/to/infer_slide.py inferSlide *(config_file, checkpoint_file, img_dir, predJsonPath)
