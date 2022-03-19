import os
import sys
import glob
import os.path as osp
import json
import shutil
import numpy as np
import xml.etree.ElementTree as ET
import cv2
import random
import math


def makeDir(dirpath):
    if not osp.exists(dirpath):
        os.makedirs(dirpath)


def get(root, name):
    return root.findall(name)

def get_and_check(root,name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars

def get_caregory_list(xml_list):
    category_list=[]
    for xmlPath in xml_list:
        tree= ET.parse(xmlPath)
        root = tree.getroot()
        for obj in get(root, 'object'):
            category=get_and_check(obj, 'name', 1).text
            if category not in set(category_list):
                category_list.append(category)
    return category_list

def convert(imgDir, xml_list, label2id, goalImgDir):
    #finish cateKey first
    cate_value=[{"id": label2id[x], "name": x, "supercategory": ''} for x in label2id]
    json_dict = {"images": [], "annotations": [], "categories":cate_value}

    image_id=0
    for xml_f in xml_list:
        tree = ET.parse(xml_f)
        root = tree.getroot()
        
        imgname = os.path.basename(xml_f)[:-4] + ".jpg"
        if cv2.imread(osp.join(imgDir, imgname)) is not None:
            shutil.copy(osp.join(imgDir, imgname), goalImgDir)
            size = get_and_check(root, 'size', 1)
            width = int(get_and_check(size, 'width', 1).text)
            height = int(get_and_check(size, 'height', 1).text)
    
            if width==0 or height==0:
                img = cv2.imread(osp.join(imgPath, imgname))
                width,height,_ = img.shape
    
            image = {'file_name': imgname, 'height': height, 'width': width, 'id':image_id}
            json_dict['images'].append(image)

            bbox_id=0
            for obj in get(root, 'object'):
                category = get_and_check(obj, 'name', 1).text
                category_id = label2id[category]

                bndbox = get_and_check(obj, 'bndbox', 1)
                xmin = math.floor(float(get_and_check(bndbox, 'xmin', 1).text))
                ymin = math.floor(float(get_and_check(bndbox, 'ymin', 1).text))
                xmax = math.floor(float(get_and_check(bndbox, 'xmax', 1).text))
                ymax = math.floor(float(get_and_check(bndbox, 'ymax', 1).text))
                assert(xmax > xmin), "xmax <= xmin, {}".format(line)
                assert(ymax > ymin), "ymax <= ymin, {}".format(line)
                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)
                ann = {'area': o_width*o_height, 
                'iscrowd': 0, 
                'image_id':image_id, 
                'bbox':[xmin, ymin, o_width, o_height],
                'category_id': category_id,
                'id': bbox_id, 
                'segmentation': []}
                json_dict['annotations'].append(ann)
                bbox_id += 1
            image_id +=1
        else:
            continue
    return json_dict

def xml2json(imgDir, xmlDir, goalDir, ratio=0.9):
    ratio=float(ratio)
    makeDir(goalDir)
    imgDir_n=osp.join(goalDir, 'imgs')
    annosDir=osp.join(goalDir, 'annos')
    makeDir(imgDir_n)
    makeDir(annosDir)

    xml_list=glob.glob(osp.join(xmlDir, '*.xml'))

    cate_list=get_caregory_list(xml_list)
    label2id={}
    for cate in cate_list:
        label2id[cate] = int(input('{} : '.format(cate)).strip())

    random.shuffle(xml_list)

    train_nums=math.floor(len(xml_list)*0.9)
    xml_list_train=xml_list[:train_nums]
    xml_list_test=xml_list[train_nums:]

    train_anno=convert(imgDir, xml_list_train, label2id, imgDir_n)
    print('train.json is done!')
    test_anno=convert(imgDir, xml_list_test, label2id, imgDir_n)
    with open(osp.join(annosDir, 'train.json'),'w', encoding='utf-8')as f:
        f.write(json.dumps(train_anno))
    with open(osp.join(annosDir, 'test.json'),'w', encoding='utf-8')as f:
        f.write(json.dumps(test_anno))
    print('test.json is done!')

if __name__=='__main__':
    if len(sys.argv)>0:
        func=getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('wrong')

 