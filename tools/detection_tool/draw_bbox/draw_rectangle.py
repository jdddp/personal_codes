# -*- coding:utf-8 -*- 
import cv2
import json
import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import argparse
import os.path as osp
# from pathlib import Path
from tools.common_use import *

def draw_bbox(src_dir,dst_dir,img_name,res):
    cv2_im = cv2.imread(osp.join(src_dir,img_name))
    h,w,c = cv2_im.shape
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
        draw.text((bbox[0], bbox[1]), '{}_{:.2}'.format(label,score),fill=color, font=ImageFont.truetype("/home/porn/rongkang/visualize_tool/vis_tool_backend/public/stylesheets/MS.ttc"))
    source_img.save(osp.join(dst_dir,img_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('-p','--project_path', required=True, help="project's absolute path")
    parser.add_argument('-i','--img_folder', default='imgs', help="imgs's folder path")
    # parser.add_argument('-o','--output_path', default='detect_img', help="drawed imgs's folder path")
    
    args = parser.parse_args()
    os.chdir(args.project_path) 
    # Path('./detect_img').mkdir(parents=True, exist_ok=True)
    mkDir('./detect_img')

    # if(osp.exists('./client_result.txt')):

    #     with open('./client_result.txt')as f:
    #         for line in f:
    #             try:
    #                 name,res= line.strip().split('\t')
    #                 name = osp.basename(name)
    #                 res = json.loads(res)

    #                 cv2_im = cv2.imread(osp.join(args.img_folder,name))
    #                 h,w,c = cv2_im.shape
    #                 source_img = Image.fromarray(cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)) 
    #                 draw = ImageDraw.Draw(source_img)
                    
    #                 for lab in res:
    #                     label = lab['label']
    #                     score = lab['score']
    #                     bbox = lab['bbox']
    #                     color = tuple(np.random.randint(256, size=3))
    #                     if w*h <=200*200:
    #                         draw.rectangle(((bbox[0], bbox[1]), (bbox[2] + bbox[0], bbox[3] + bbox[1])),width=2,outline =color)
    #                         draw.text((bbox[0], bbox[1]), '{}_{:.2}'.format(label,score),fill=color, font=ImageFont.truetype("/home/porn/rongkang/visualize_tool/vis_tool_backend/public/stylesheets/MS.ttc",20))
    #                     elif w * h <=500*500:
    #                         draw.rectangle(((bbox[0], bbox[1]), (bbox[2] + bbox[0], bbox[3] + bbox[1])),width=4,outline =color)
    #                         draw.text((bbox[0], bbox[1]), '{}_{:.2}'.format(label,score),fill=color, font=ImageFont.truetype("/home/porn/rongkang/visualize_tool/vis_tool_backend/public/stylesheets/MS.ttc",35))
    #                     else:
    #                         draw.rectangle(((bbox[0], bbox[1]), (bbox[2] + bbox[0], bbox[3] + bbox[1])),width=6,outline =color)
    #                         draw.text((bbox[0], bbox[1]), '{}_{:.2}'.format(label,score),fill=color, font=ImageFont.truetype("/home/porn/rongkang/visualize_tool/vis_tool_backend/public/stylesheets/MS.ttc",45))
    #                 source_img.save('./detect_img/'+ name)

    #             except:
    #                 print('draw error')
    #                 pass
    
    # if osp.exists('./usual_json/knife_usual.json'):
    #     all_ann = json.loads(open('./usual_json/knife_usual.json').read())
    #     for item in all_ann:
    #         cv2_im = cv2.imread(osp.join(args.img_folder,item))
    #         h,w,c = cv2_im.shape
    #         source_img = Image.fromarray(cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)) 
    #         draw = ImageDraw.Draw(source_img)

    #         for bbox in all_ann[item]:
    #             label = bbox['category']
    #             bbox = bbox['bbox']
    #             color = tuple(np.random.randint(256, size=3))
    #             draw.rectangle(((bbox[0], bbox[1]), (bbox[2] + bbox[0], bbox[3] + bbox[1])),width=4,outline =color)
    #             draw.text((bbox[0], bbox[1]), label,fill=color, font=ImageFont.truetype("/home/porn/rongkang/visualize_tool/vis_tool_backend/public/stylesheets/MS.ttc",35))
    #         source_img.save('./detect_img/'+ item)

def draw_imgs(dirpath):
    imgDir=osp.join(dirpath, 'imgs')
    jsonPath=osp.join(dirpath, 'usual.json')
    ansDir=osp.join(dirpath, 'detect_img')

    if osp.isfile(jsonPath):
        usualJson=json.loads(open(osp.join(dirpath, 'usual.json').read()))
        for imgname, dctLst in usualJson.items():
            img=cv2.imread(osp.join(imgDir, imgname))
            h,w,c=img.shape
            #format change
            source_img=Image.fromArray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw=ImageDraw.Draw(source_img)

            if len(dctLst)>0:
                for sgDct in dctLst:
                    label=sgDct['category'] 
                    bbox=sgDct['bbox']
                    color=tuple(np.random.randint(256,size=3))
                    draw.rectangle(((bbox[0], bbox[1]), (bbox[2] + bbox[0], bbox[3] + bbox[1])),width=4,outline =color)
                    draw.text((bbox[0], bbox[1]), label,fill=color, font=ImageFont.truetype('./MS.ttc",35))
            source_img.save(osp.join(dirpath, imgname))

if __name__='__main__':
    if len(sys.argv)>1:
        func=sys.getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('input wrong')
        