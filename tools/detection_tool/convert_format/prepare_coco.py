import argparse
import sys
import json
import os
from pathlib import Path
import os.path as osp
import cv2
import random
import math
from tqdm import tqdm

def decon_coco(cocoJson):
    '''
        解构原始COCO格式至通用解析格式
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
            final_decon[item['file_name']] = []

    return final_decon
    



def main(config):
    label_dict = {}

    all_annos = {}
    # 遍历所有的ann文件,统一成标准解析格式后合并
    for ann in os.listdir(config.ann_dir):
        ann_json = json.loads(open(osp.join(config.ann_dir,ann)).read())
        if 'images' in ann_json and 'categories' in ann_json and 'annotations' in ann_json:
            all_annos.update(decon_coco(ann_json))
        else:
            all_annos.update(ann_json)

    print('请定义各个类别的label用于网络训练 (int)\n')
    for item in all_annos:
        for bbox in all_annos[item]:
            if (bbox['category'] not in label_dict):
                label_dict[bbox['category']] = int((input('{} : '.format(bbox['category']))).strip())


    train_coco = {
        'images':[],
        'annotations':[],
        'categories':[{"id": label_dict[x], "name": x, "supercategory": ''} for x in label_dict]
    }    # 最终的coco 格式的train

    test_coco = {
        'images':[],
        'annotations':[],
        'categories':[{"id": label_dict[x], "name": x, "supercategory": ''} for x in label_dict]
    }    # 最终的coco 格式的test
 


    """
    
    获取全部的图片的annos，然后随机后乱序，最后按照ratio 切割分成 train 和 test 集
    隐患：无法根据类别均分
    """

    trainset_ratio = float(config.trainset_ratio)

    if (trainset_ratio < 1):

        all_annos_tuple_list = list(all_annos.items())
        random.shuffle(all_annos_tuple_list)     
        train_anns = dict(all_annos_tuple_list[:math.floor(len(all_annos_tuple_list)*trainset_ratio)])
        test_anns = dict(all_annos_tuple_list[math.floor(len(all_annos_tuple_list)*trainset_ratio):])
    else:
        train_anns = all_annos
        test_anns = {}


    image_id = 0
    bbox_id = 0
    for img in train_anns:
        image_file = osp.join(config.img_dir,img)
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
                    "category_id": label_dict[ann['category']], 
                    "id": bbox_id
                    })
                bbox_id +=1
            image_id += 1

    with open(osp.join(config.output_path,'train.json'),'w')as f:
        f.write(json.dumps(train_coco,ensure_ascii=False))

    image_id = 0
    bbox_id = 0
    for img in test_anns:
        image_file = osp.join(config.img_dir,img)
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
                    "category_id": label_dict[ann['category']], 
                    "id": bbox_id
                    })
                bbox_id +=1
            image_id += 1

    with open(osp.join(config.output_path,'test.json'),'w')as f:
        f.write(json.dumps(test_coco,ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # default_save_path = osp.join(osp.split(osp.realpath(sys.argv[0]))[0],'output')   #默认产出都在当前的output文件夹下

    parser.add_argument(
        "-a",
        "--ann_dir",
        type=str,
        required=True,
        help="项目所用的ann的地址")


    parser.add_argument(
        "-i",
        "--img_dir",
        type=str,
        required=True,
        help="项目所用的img的地址,用于检查校验")

    parser.add_argument(
        "-r",
        "--trainset_ratio",
        default='1',
        type=str,
        help="验证集的占比 (0-1),默认1")
    
    parser.add_argument(
        "-o",
        "--output_path",
        default=default_save_path,
        type=str,
        help="生成的coco.json的保存地址")


    config = parser.parse_args()

    Path(config.output_path).mkdir(parents=True, exist_ok=True)

    assert osp.isdir(config.ann_dir) and osp.isdir(config.img_dir)

    main(config)
