# -*- coding:utf-8 -*- 
import cv2
import json
import os
import sys
import os.path as osp
import numpy as np
from PIL import Image, ImageFont, ImageDraw


def draw_bbox(src_dir,dst_dir,img_name,res):
    '''res->dict;三个字段匹配usualJson;bbox[xywh]、cate、score
    '''
    cv2_im = cv2.imread(osp.join(src_dir,img_name))
    h,w,_ = cv2_im.shape
    width = 2 if h*w >= 200*200 else 3
    source_img = Image.fromarray(cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)) 
    draw = ImageDraw.Draw(source_img)

    for lab in res:
        label = lab['label']
        score = lab['score']
        bbox = lab['bbox']
        print(label,label.decode('utf-8'),label.decode('unicode-escape'))
        color = tuple(np.random.randint(256, size=3))
        draw.rectangle(((bbox[0], bbox[1]), (bbox[2] + bbox[0], bbox[3] + bbox[1])),width=width+1,outline =color)
        draw.text((bbox[0], bbox[1]), '{}_{:.2}'.format(label,score),fill=color, font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc"))
    source_img.save(osp.join(dst_dir,img_name))


def draw_imgs(dirpath):
    '''dirpath/imgs -> img_data
    dirpath/usual.json -> usualJson of img_data
    '''
    imgDir=osp.join(dirpath, 'imgs')
    jsonPath=osp.join(dirpath, 'usual.json')
    ansDir=osp.join(dirpath, 'visual_img')
    os.makedirs(ansDir,exist_ok=True)

    if osp.isfile(jsonPath):
        usualJson=json.loads(open(jsonPath).read())
        for imgname, dctLst in usualJson.items():
            # img=cv2.imread(osp.join(imgDir, imgname))
            source_img=Image.open(osp.join(imgDir, imgname))

            draw=ImageDraw.Draw(source_img)

            if len(dctLst)>0:
                for sgDct in dctLst:
                    label=sgDct['category'] 
                    bbox=sgDct['bbox']
                    color=tuple(np.random.randint(256,size=3))
                    draw.rectangle(((bbox[0], bbox[1]), (bbox[2] + bbox[0], bbox[3] + bbox[1])),width=4,outline =color)
                    draw.text((bbox[0], bbox[1]), label,fill=color, font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',35))
            source_img=source_img.convert('RGB')
            # try:
            source_img.save(osp.join(ansDir, imgname))
            # except:
            #     print(imgname)

if __name__=='__main__':
    if len(sys.argv)>1:
        func=getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('input wrong')
        
