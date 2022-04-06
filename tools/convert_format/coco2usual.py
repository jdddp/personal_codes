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
    id2label = {}

    ann_list = {}
    final_decon = {}
    for item in cocoJson['categories']:
        id2label[int(item['id'])] = item['name']

    for item in cocoJson['annotations']:
        if int(item['image_id']) in ann_list:
            ann_list[int(item['image_id'])].append({
                'category':id2label[int(item['category_id'])],
                'bbox':item['bbox']
            })
        else:
            ann_list[int(item['image_id'])] = [{
                'category':id2label[int(item['category_id'])],
                'bbox':item['bbox']
            }]
        
    for item in cocoJson['images']:
        if(int(item['id']) in ann_list):
            final_decon[item['file_name']] = ann_list[int(item['id'])]
        else:
            # final_decon[item['file_name']] = []
            pass

    return final_decon

def main(cocoJson_path, usualJson_path):
    # all_anns={}
    if osp.isdir(cocoJson_path):
        goalJson={}
        for ann in os.listdir(cocoJson_path):
            cocoJson=json.loads(open(osp.join(cocoJson_path,ann)).read())
            goalJson.update(decon_coco(cocoJson))
    else:
        goalJson=decon_coco(json.loads(open(cocoJson_path).read()))
    with open(osp.join(usualJson_path, 'usual.json'),'w', encoding='utf-8')as f:
        json.dump(goalJson, f)

if __name__ == '__main__':
    if len(sys.argv)>1:
        func=getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('input wrong')

#python path/of/decoco.py path/of/coco.json path/of/usual.json