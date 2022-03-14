import sys
import json
import os
import os.path as osp
import cv2
import random
import math

def decon_coco(cocoJson):
    '''
        解构原始COCO格式至标准通用解析格式
    '''
    label_name = {}

    ann_list = {}
    final_decon = {}
    for item in ann['categories']:
        label_name[int(item['id'])] = item['name']

    for item in ann['annotations']:
        if int(item['image_id']) in ann_list:
            ann_list[int(item['image_id'])].append({
                'category':label_name[int(item['category_id'])],
                'bbox':item['bbox']
            })
        else:
            ann_list[int(item['image_id'])] = [{
                'category':label_name[int(item['category_id'])],
                'bbox':item['bbox']
            }]
        
    for item in ann['images']:
        if(int(item['id']) in ann_list):
            final_decon[item['file_name']] = ann_list[int(item['id'])]
        else:
            # final_decon[item['file_name']] = []
            pass

    return final_decon

def main(cocoJson_path, usualJson_path):
    cocoJson=json.loads(open(cocoJson_path).read())
    goalJson=decon_coco(cocoJson)
    with open(usualJson_path,'w')as f:
        json.dump(goalJson, f)

if __name__ == '__main__':
    if len(sys)>1:
        fun=getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('input wrong')

#python path/of/decoco.py path/of/coco.json path/of/usual.json