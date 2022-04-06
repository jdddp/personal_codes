import os
import sys
import os.path as osp
import json
import random
import math
import shutil
import cv2
from tqdm import tqdm

from xml2usual import xml2usual,makeDir
'''
dataset/images/train/*.jpg
       /images/test
       /labels/train/*.txt
       /labels/test
'''

def dct2txt(dct_lst,label2id,tp):
    ans_list=[]
    if len(dct_lst)>0:
        for dct in dct_lst:
            idd=label2id[dct['category']]
            x1,y1,w,h=dct['bbox']
            xc,yc=x1+w/2.0, y1+h/2.0
        ans_list.append(str(idd)+' '+str(xc/tp[0])+' '+str(yc/tp[1])+' ' \
            + str(w/tp[0])+' '+str(h/tp[1]))
    return ans_list

def makeTxt(dirpath, txtPath):
    imglist=os.listdir(dirpath)
    imgPath_list=[osp.join(dirpath, x) for x in imglist]
    with open(txtPath,'w', encoding='utf-8')as f:
        f.write('\n'.join(imgPath_list))

def xml2yolov5(imgDir,xmlDir, goalDir,ratio=0.9):
    ratio=float(ratio)

    xml2usual(imgDir,xmlDir,goalDir)
    label2id=json.loads(open(osp.join(goalDir, 'label2id.json')).read())
    usualJson=json.loads(open(osp.join(goalDir, 'usual.json')).read())

    for dirName in ['images','labels']:
        for dirname in ['train', 'test']:
            makeDir(osp.join(goalDir,dirName, dirname))
    makeDir(osp.join(goalDir, 'pathFile'))
    
    all_annos_tp_list=list(usualJson.items())
    random.shuffle(all_annos_tp_list)
    random.shuffle(all_annos_tp_list)
    train_anns = dict(all_annos_tp_list[:math.floor(len(all_annos_tp_list)*ratio)])
    test_anns = dict(all_annos_tp_list[math.floor(len(all_annos_tp_list)*ratio):])
    
    for imgname in tqdm(train_anns):
        imgpath=osp.join(goalDir,'imgs',imgname)
        txtPath=osp.join(goalDir, 'labels', 'train',imgname.split('.')[0]+'.txt')
        img=cv2.imread(imgpath)
        h,w,_=img.shape
        ansLst=dct2txt(train_anns[imgname],label2id,(w,h))
        shutil.move(imgpath, osp.join(goalDir, 'images', 'train'))
        with open(txtPath,'w')as f:
            f.write('\n'.join(ansLst)) 
    
    for imgname in tqdm(test_anns):
        imgpath=osp.join(goalDir,'imgs',imgname)
        txtPath=osp.join(goalDir, 'labels', 'test',imgname.split('.')[0]+'.txt')
        img=cv2.imread(imgpath)
        h,w,_=img.shape
        ansLst=dct2txt(test_anns[imgname],label2id,(w,h))
        shutil.move(imgpath, osp.join(goalDir, 'images', 'test'))
        with open(txtPath, 'w')as f:
            f.write('\n'.join(ansLst)) 
    
    shutil.rmtree(osp.join(goalDir, 'imgs'))

    makeTxt(osp.join(goalDir,'images','train'), osp.join(goalDir, 'pathFile', 'train.txt'))
    makeTxt(osp.join(goalDir,'images','test'), osp.join(goalDir, 'pathFile', 'test.txt'))

    



if __name__=='__main__':
    if len(sys.argv)>0:
        func=getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('wrong')


    







