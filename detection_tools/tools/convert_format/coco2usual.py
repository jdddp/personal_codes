import sys
import json
import os
import os.path as osp
import cv2
import random
import math

def decon_coco(cocoJson):
    '''解构原始COCO格式至标准通用解析格式
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
            final_decon[item['file_name']] = []
            # pass
    # print(final_decon)
    return final_decon

#批量cocoORusual 2 usual
def coco2u(cocoJson_path, usualJson_path):
    '''cocoJson_path: dir with cocojson or usualjson
    usualJson_path:dir for usual.json
    '''
    # all_anns={}
    if osp.isdir(cocoJson_path):
        goalJson={}
        for ann in os.listdir(cocoJson_path):
            cocoJson=json.loads(open(osp.join(cocoJson_path,ann)).read())
            if 'images' in cocoJson and 'categories' in cocoJson and 'annotations' in cocoJson:
                goalJson.update(decon_coco(cocoJson))
            else:
                goalJson.update(cocoJson)
    else:
        goalJson=decon_coco(json.loads(open(cocoJson_path).read()))
    with open(osp.join(usualJson_path, 'usual.json'),'w', encoding='utf-8')as f:
        f.write(json.dumps(goalJson))
    return goalJson

def sg_usual2coco(usualJson_path,goal_coco_dataset, ratio=0.9):
    '''goal_coco_dataset:
            -imgs
            -annos(train.json,test.json)
    usualJson_path: path of usualJson
    '''
    ratio=float(ratio)
    img_dir=osp.join(goal_coco_dataset,'imgs')
    os.makedirs(img_dir, exist_ok=True)
    coco_dir=osp.join(goal_coco_dataset, 'annos')
    os.makedirs(coco_dir, exist_ok=True)

    label2id={}
    res=json.loads(open(usualJson_path).read())

    print('define id of each cate')
    for imgname,dctlst in res.items():
        for sgDct in dctlst:
            if sgDct['category'] not in label2id:
                label2id[sgDct['category']]=int(input('{} : '.format(sgDct['category'])).strip())
    
    train_coco = {
        'images':[],
        'annotations':[],
        'categories':[{"id": label2id[x], "name": x, "supercategory": ''} for x in label2id]
    }
    test_coco = {
        'images':[],
        'annotations':[],
        'categories':[{"id": label2id[x], "name": x, "supercategory": ''} for x in label2id]
    }
    
    #spilt train/test
    if (ratio < 1):
        res_tp_list = list(res.items())
        random.shuffle(res_tp_list)     
        train_anns = dict(res_tp_list[:math.floor(len(res_tp_list)*ratio)])
        test_anns = dict(res_tp_list[math.floor(len(res_tp_list)*ratio):])
    else:
        train_anns = res
        test_anns = {}


    image_id,bbox_id=0,0
    for img in train_anns:
        image_file = osp.join(img_dir,img)
        #检验图片是否损坏
        im = cv2.imread(image_file)
        if im is not None:
            h, w, c = im.shape
            train_coco['images'].append({"id": image_id, "width": w, "height": h, "file_name": img})
            for ann in train_anns[img]:
                train_coco['annotations'].append({
                    "segmentation": [], 
                    "iscrowd": 0, 
                    "image_id": image_id, 
                    "bbox": ann['bbox'], 
                    "area": ann['bbox'][-1] * ann['bbox'][-2], 
                    "category_id": label2id[ann['category']], 
                    "id": bbox_id
                    })
                bbox_id +=1
            image_id += 1
    with open(osp.join(coco_dir,'train.json'),'w',encoding='utf-8')as f:
        f.write(json.dumps(train_coco))
    print('train.json is done,img_nums is {}'.format(math.floor(len(list(res.items()))*ratio)))

    image_id = 0
    bbox_id = 0
    for img in test_anns:
        image_file = osp.join(img_dir,img)
        im = cv2.imread(image_file)
        if im is not None:
            h, w, c = im.shape
            test_coco['images'].append({"id": image_id, "width": w, "height": h, "file_name": img})
            for ann in test_anns[img]:
                test_coco['annotations'].append({
                    "segmentation": [], 
                    "iscrowd": 0, 
                    "image_id": image_id, 
                    "bbox": ann['bbox'], 
                    "area": ann['bbox'][-1] * ann['bbox'][-2], 
                    "category_id": label2id[ann['category']], 
                    "id": bbox_id
                    })
                bbox_id +=1
            image_id += 1

    with open(osp.join(coco_dir,'test.json'),'w', encoding='utf-8')as f:
        f.write(json.dumps(test_coco))
    if ratio!=1:
        print('train.json is done,img_nums is {}'.format(math.ceil(len(list(res.items()))*(1-ratio))))
    else:
        print('train.json is done')

def usual2coco(usualJson_dir,goal_coco_dataset, ratio=0.9):
    '''goal_coco_dataset:
            -imgs
            -annos(train.json,test.json)
    usualJson_path: path of usualJson
    '''
    ratio=float(ratio)
    img_dir=osp.join(goal_coco_dataset,'imgs')
    os.makedirs(img_dir, exist_ok=True)
    coco_dir=osp.join(goal_coco_dataset, 'annos')
    os.makedirs(coco_dir, exist_ok=True)

    label2id={}

    res=coco2u(usualJson_dir, usualJson_dir)

    print('define id of each cate')
    for imgname,dctlst in res.items():
        for sgDct in dctlst:
            if sgDct['category'] not in label2id:
                label2id[sgDct['category']]=int(input('{} : '.format(sgDct['category'])).strip())
    
    train_coco = {
        'images':[],
        'annotations':[],
        'categories':[{"id": label2id[x], "name": x, "supercategory": ''} for x in label2id]
    }
    test_coco = {
        'images':[],
        'annotations':[],
        'categories':[{"id": label2id[x], "name": x, "supercategory": ''} for x in label2id]
    }
    
    #spilt train/test
    if (ratio < 1):
        res_tp_list = list(res.items())
        random.shuffle(res_tp_list)     
        train_anns = dict(res_tp_list[:math.floor(len(res_tp_list)*ratio)])
        test_anns = dict(res_tp_list[math.floor(len(res_tp_list)*ratio):])
    else:
        train_anns = res
        test_anns = {}


    image_id,bbox_id=0,0
    for img in train_anns:
        image_file = osp.join(img_dir,img)
        #检验图片是否损坏
        im = cv2.imread(image_file)
        if im is not None:
            h, w, c = im.shape
            train_coco['images'].append({"id": image_id, "width": w, "height": h, "file_name": img})
            for ann in train_anns[img]:
                train_coco['annotations'].append({
                    "segmentation": [], 
                    "iscrowd": 0, 
                    "image_id": image_id, 
                    "bbox": ann['bbox'], 
                    "area": ann['bbox'][-1] * ann['bbox'][-2], 
                    "category_id": label2id[ann['category']], 
                    "id": bbox_id
                    })
                bbox_id +=1
            image_id += 1
    with open(osp.join(coco_dir,'train.json'),'w',encoding='utf-8')as f:
        f.write(json.dumps(train_coco))
    print('train.json is done,img_nums is {}'.format(math.ceil(len(list(res.items()))*ratio)))

    image_id = 0
    bbox_id = 0
    for img in test_anns:
        image_file = osp.join(img_dir,img)
        im = cv2.imread(image_file)
        if im is not None:
            h, w, c = im.shape
            test_coco['images'].append({"id": image_id, "width": w, "height": h, "file_name": img})
            for ann in test_anns[img]:
                test_coco['annotations'].append({
                    "segmentation": [], 
                    "iscrowd": 0, 
                    "image_id": image_id, 
                    "bbox": ann['bbox'], 
                    "area": ann['bbox'][-1] * ann['bbox'][-2], 
                    "category_id": label2id[ann['category']], 
                    "id": bbox_id
                    })
                bbox_id +=1
            image_id += 1

    with open(osp.join(coco_dir,'test.json'),'w', encoding='utf-8')as f:
        f.write(json.dumps(test_coco))
    if ratio!=1:
        print('train.json is done,img_nums is {}'.format(math.ceil(len(list(res.items()))*(1-ratio))))
    else:
        print('train.json is done')
    

if __name__ == '__main__':
    if len(sys.argv)>1:
        func=getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('input wrong')

#python path/of/decoco.py path/of/coco.json path/of/usual.json