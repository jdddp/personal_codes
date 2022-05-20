# -*- coding:utf-8 -*- 
import cv2
import json
import os
import sys
import os.path as osp
import numpy as np
from PIL import Image, ImageFont, ImageDraw


def draw_bbox_with_score(src_dir,dst_dir,img_name,res):
    '''res->dict;三个字段匹配usualJson;bbox[xywh]、cate、score
    '''
    cv2_im = cv2.imread(osp.join(src_dir,img_name))
    h,w,_ = cv2_im.shape
    wid_th = 4 if h*w >= 200*200 else 3
    source_img = Image.fromarray(cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)) 
    draw = ImageDraw.Draw(source_img)

    for lab in res:
        label = lab['label']
        score = lab['score']
        bbox = lab['bbox']
        print(label,label.decode('utf-8'),label.decode('unicode-escape'))
        color = tuple(np.random.randint(256, size=3))
        draw.rectangle(((bbox[0], bbox[1]), (bbox[2] + bbox[0], bbox[3] + bbox[1])),width=wid_th+1,outline =color)
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
                    if 'score' in sgDct:
                        score=sgDct['score']
                        draw.text((bbox[0], bbox[1]), '{}_{:.2}'.format(label,score),fill=color, font=ImageFont.truetype("/usr/share/fonts/truetype/pagul/Pagul.ttf", 20))
                    else:
                        draw.text((bbox[0], bbox[1]), label,fill=color, font=ImageFont.truetype('/usr/share/fonts/truetype/pagul/Pagul.ttf',14))
                    #C:/Windows/Fonts/msyh.ttc
            source_img=source_img.convert('RGB')
            # try:
            source_img.save(osp.join(ansDir, imgname))
            # except:
            #     print(imgname)

#for ../../detection_tools/miss_mistake
def draw_miss_mistake(imgDir,miss_mistake_json,goalDir):
    
    os.makedirs(goalDir, exist_ok=True)
    res=json.loads(open(miss_mistake_json).read())
    miss_color=tuple(np.random.randint(256, size=3))
    mistake_color=tuple(np.random.randint(256, size=3))
    for imgname,img_info in res.items():
        if len(img_info['miss'])==0 and len(img_info['mistake'])==0:
            continue
        cv2_im = cv2.imread(osp.join(imgDir,imgname))
        h,w,_ = cv2_im.shape
        wid_th = 4 if h*w >= 200*200 else 3
        source_img = Image.fromarray(cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)) 
        draw = ImageDraw.Draw(source_img)

        for sgDct in img_info['mistake']:
            label = sgDct['category']
            score = sgDct['score']
            bbox = sgDct['bbox']
            draw.rectangle(((bbox[0], bbox[1]), (bbox[2] + bbox[0], bbox[3] + bbox[1])),width=wid_th+1,outline =mistake_color)
            draw.text((bbox[0], bbox[1]), '{}_{:.2}'.format(label,score),fill=mistake_color, font=ImageFont.truetype("/usr/share/fonts/truetype/pagul/Pagul.ttf", 15))
        
        for sgDct in img_info['miss']:
            label = sgDct['category']
            bbox = sgDct['bbox']
            draw.rectangle(((bbox[0], bbox[1]), (bbox[2] + bbox[0], bbox[3] + bbox[1])),width=wid_th+1,outline =miss_color)
            draw.text((bbox[0], bbox[1]), '{}'.format(label),fill=miss_color, font=ImageFont.truetype("/usr/share/fonts/truetype/pagul/Pagul.ttf", 15))
            

        source_img.save(osp.join(goalDir,imgname))

if __name__=='__main__':
    if len(sys.argv)>1:
        func=getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('input wrong')
        
