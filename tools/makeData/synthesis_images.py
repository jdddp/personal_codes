# -*- coding: utf-8 -*-
import os, sys, pdb, glob
import cv2
import os.path as osp
import random
import numpy as np
from math import sqrt
from tqdm import tqdm

def possion_merge(bg_path, obj_path, save_path, det_path, cate):
    # Read images : src image will be cloned into dst
    try:
        bg = cv2.imread(bg_path)
        obj = cv2.imread(obj_path)
    except:
        return
    
    bg_h, bg_w = bg.shape[:2]
    obj_h, obj_w = obj.shape[:2]

    center = (bg_w // 2, bg_h // 2)
    
    s = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    while ((center[0] + obj_w*0.5 > bg_w) or (center[1] + obj_h*0.5 > bg_h)
        or (center[0] - obj_w*0.5 < 0) or (center[1] - obj_h*0.5 < 0)):
        #ratio = 0.8 #np.random.rand(1)[0]
        ratio = s[random.randint(0,len(s)-1)]
        obj = cv2.resize(obj,(obj_w//2, obj_h//2))
        obj_h, obj_w = obj.shape[:2]

    # Create an all white mask
    mask = 255 * np.ones(obj.shape, obj.dtype)
    
    normal_clone = cv2.seamlessClone(obj, bg, mask, center, cv2.NORMAL_CLONE)
    mixed_clone = cv2.seamlessClone(obj, bg, mask, center, cv2.MIXED_CLONE)
    split_clone=bg
    for j in range(obj_h):
        for i in range(obj_w):
            split_clone[center[1] - obj.shape[0]//2+j,center[0] - obj.shape[1]//2+i,:]=obj[j,i,:]
    
    x1 = center[0] - obj.shape[1]//2
    x2 = center[0] + obj.shape[1]//2
    y1 = center[1] - obj.shape[0]//2
    y2 = center[1] + obj.shape[0]//2
    #cv2.rectangle(normal_clone,(x1,y1),(x2,y2), (0,255,0),5)
    #cv2.rectangle(mixed_clone,(x1,y1),(x2,y2), (0,255,0),5)
    # Write results
    cv2.imwrite(save_path+"_normal.jpg", normal_clone)
    cv2.imwrite(save_path+"_fluid.jpg", mixed_clone)

    cv2.imwrite(save_path+'_split.jpg', split_clone)

    f = open(det_path+"_normal.txt", 'w')
    f.write("%d %d %d %d %s\n" % (x1, y1, x2, y2, cate))
    f.close()
    
    f = open(det_path+"_fluid.txt", 'w')
    f.write("%d %d %d %d %s\n" % (x1, y1, x2, y2, cate))
    f.close()

    f = open(det_path+"_split.txt", 'w')
    f.write("%d %d %d %d %s\n" % (x1, y1, x2, y2, cate))
    f.close()

if __name__ == "__main__":
    obj_dir = "/home/disk1/jiangzhipeng01/sensitiveFlag/sensitiveFlagV2/forhecheng"
    bg_dir = "/home/disk1/jiangzhipeng01/sensitiveFlag/dikuHe/backgroud_img/album_data_resized" #"/home/vis/zhouzhichao01/ROI/data/original/cover_1w/"

    bg_list = glob.glob(bg_dir+"/*.jpg")
    # print(len(bg_list))
    save_dir = "/home/disk1/jiangzhipeng01/sensitiveFlag/sensitiveFlagV2/hechengImgs"
    det_dir = "/home/disk1/jiangzhipeng01/sensitiveFlag/sensitiveFlagV2/hechengTxtdir"
    os.makedirs(save_dir)
    os.makedirs(det_dir)
    lable_list=os.listdir(obj_dir)
    
    num=0
    for lable in tqdm(lable_list):
        obj_list=glob.glob(osp.join(obj_dir, lable)+"/*.jpg")
        lableforwrite=lable
        # print(lableforwrite)
        # if '_' in lableforwrite:
        #     lableforwrite=lableforwrite.split('_')[0]
        # if '-' in lableforwrite:
        #     lableforwrite=lableforwrite.split('-')[0]
        
        # for obj_path in obj_list:
        for i in range((2000-len(obj_list))//3):
            obj_ls=random.randint(0,len(obj_list)-1)
            obj_path=obj_list[obj_ls]
            num+=1
            if num>400000:
                num=num%400000
            bg_path=bg_list[num]
            savePath=osp.join(save_dir, str(i)+osp.basename(obj_path).split('.')[0])
            detPath=osp.join(det_dir, str(i)+osp.basename(obj_path).split('.')[0])
            possion_merge(bg_path, obj_path, savePath, detPath, lableforwrite)






