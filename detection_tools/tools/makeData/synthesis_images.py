'''三种方法，两种是opencv自带的，一种是直接抠图，可根据需要注掉部分；
'''
import os, sys, pdb, glob
import cv2
import os.path as osp
import random
import numpy as np
import math
from tqdm import tqdm
import collections
import json

def merge_img(bg_path, obj_path, save_path, det_path, cate):
    '''
    det_path:annotation file(.txt)
    '''
    try:
        bg = cv2.imread(bg_path)
        obj = cv2.imread(obj_path)
    except:
        return
    
    bg_h, bg_w = bg.shape[:2]
    obj_h, obj_w = obj.shape[:2]

    center = (bg_w // 2, bg_h // 2)
    
    #random choose scale for target
    s = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ratio = s[random.randint(0,len(s)-1)]

    #加个保险
    if (center[0] + math.ceil(obj_w*0.4) > bg_w) or (center[1] + math.ceil(obj_h*0.4) > bg_h) \
        or (center[0] - math.ceil(obj_w*0.4) < 0) or (center[1] - math.ceil(obj_h*0.4) < 0):
        ratio=min(float(obj_h)/bg_h, float(obj_w)/bg_w)

    while ((center[0] + math.ceil(obj_w*ratio) > bg_w) or (center[1] + math.ceil(obj_h*ratio) > bg_h)
        or (center[0] - math.ceil(obj_w*ratio) < 0) or (center[1] - math.ceil(obj_h*ratio) < 0)):
        #ratio = 0.8 #np.random.rand(1)[0]
        obj = cv2.resize(obj,(math.ceil(obj_w*ratio), math.ceil(obj_h*ratio)))
        obj_h, obj_w = obj.shape[:2]
        ratio = s[random.randint(0,len(s)-1)]


    # create an all white mask
    mask = 255 * np.ones(obj.shape, obj.dtype)
    
    #1
    normal_clone = cv2.seamlessClone(obj, bg, mask, center, cv2.NORMAL_CLONE)
    #2
    mixed_clone = cv2.seamlessClone(obj, bg, mask, center, cv2.MIXED_CLONE)
    #3
    split_clone=bg
    for j in range(obj_h):
        for i in range(obj_w):
            split_clone[center[1] - obj.shape[0]//2+j,center[0] - obj.shape[1]//2+i,:]=obj[j,i,:]
    
    x1 = center[0] - obj.shape[1]//2
    x2 = center[0] + obj.shape[1]//2
    y1 = center[1] - obj.shape[0]//2
    y2 = center[1] + obj.shape[0]//2

    # Write results
    cv2.imwrite(save_path+"_normal.jpg", normal_clone)
    cv2.imwrite(save_path+"_mixed.jpg", mixed_clone)
    cv2.imwrite(save_path+'_split.jpg', split_clone)

    f = open(det_path+"_normal.txt", 'w')
    f.write("%d %d %d %d %s\n" % (x1, y1, x2, y2, cate))
    f.close()
    
    f = open(det_path+"_mixed.txt", 'w')
    f.write("%d %d %d %d %s\n" % (x1, y1, x2, y2, cate))
    f.close()

    f = open(det_path+"_split.txt", 'w')
    f.write("%d %d %d %d %s\n" % (x1, y1, x2, y2, cate))
    f.close()

def syn_imgs(obj_dir, bg_dir, save_dir,cate_nums=2000):
    '''obj_dir:crop ans
    bg_dir:background
    save_dir:path of new data
    '''
    cate_nums=int(cate_nums)

    imgDir=osp.join(save_dir,'fake_imgs')
    os.makedirs(osp.join(save_dir,'fake_imgs'),exist_ok=True)
    detDir=osp.join(save_dir,'fake_annos')
    os.makedirs(osp.join(save_dir,'fake_annos'),exist_ok=True)

    bg_list=glob.glob(bg_dir+'/*.jpg')
    bg_nums=len(bg_list)

    for label in tqdm(os.listdir(obj_dir)):
        obj_list=glob.glob(osp.join(obj_dir,label)+ '/.jpg')

        #assigned nums for each cate
        for i in range((cate_nums-len(obj_list))//3):

            obj_index=random.randint(0,len(obj_list)-1)
            obj_path=obj_index[obj_index]

            bg_index=random.randint(0,bg_nums-1)
            bg_path=bg_index[bg_index]

            imgPath=osp.join(imgDir, str(i)+'-'+osp.basename(obj_path).split('.')[0])
            detPath=osp.join(detDir, str(i)+'-'+osp.basename(obj_path).split('.')[0])
            merge_img(bg_path, obj_path, imgPath, detPath,label)

def txt2usualJson(save_dir,jsonPath=None):
    ansDct=collections.defaultdict(list)
    if jsonPath==None:
        jsonPath=osp.join(save_dir, 'fake_usual.json')

    txtDir=osp.join(save_dir, 'fake_annos')
    for txtFile in os.listdir(txtDir):
        imgname=osp.basename(txtFile).split('.')[0]+'.jpg'
        with open(txtDir, encoding='utf-8')as f:
            for line in f:
                x1 ,y1, x2, y2, cate=line.strip().split(' ')
                sgDct={
                    'category':cate,
                    'bbox':[math.floor(x1), math.floor(y1), math.ceil(x2)-math.floor(x1), math.ceil(y2)-math.floor(y1)]
                }
                ansDct[imgname].append(sgDct)
    
    with open(jsonPath,'w',encoding='utf-8')as f:
        f.write(json.dumps(ansDct))


if __name__ == "__main__":
    if len(sys.argv>1):
        func=getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('wrong!')

#1
# ./cutImage.py for obj_dir

#2
#python path/to/.py syn_imgs obj_dir bg_dir save_dir

#3 convert format
#python path/to/.py




