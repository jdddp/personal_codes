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
import collections


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

def convert(imgDir, xml_list,goalImgDir):
    label_list=[]
    ansDct=collections.defaultdict(list)
    for xml_f in xml_list:
        tree = ET.parse(xml_f)
        root = tree.getroot()
        
        imgname = os.path.basename(xml_f)[:-4] + ".jpg"

        if cv2.imread(osp.join(imgDir, imgname)) is not None:

            shutil.copy(osp.join(imgDir, imgname), goalImgDir)
            for obj in get(root, 'object'):
                category = get_and_check(obj, 'name', 1).text
                if category not in set(label_list):
                    label_list.append(category)
                bndbox = get_and_check(obj, 'bndbox', 1)
                xmin = math.floor(float(get_and_check(bndbox, 'xmin', 1).text))
                ymin = math.floor(float(get_and_check(bndbox, 'ymin', 1).text))
                xmax = math.floor(float(get_and_check(bndbox, 'xmax', 1).text))
                ymax = math.floor(float(get_and_check(bndbox, 'ymax', 1).text))
                assert(xmax > xmin), "xmax <= xmin, {}".format(line)
                assert(ymax > ymin), "ymax <= ymin, {}".format(line)
                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)
                ansDct[imgname].append({'category':category,'bbox':[xmin,ymin,o_width,o_height]})
        else:
            continue
    return ansDct,label_list

def xml2usual(imgDir, xmlDir, goalDir):
    makeDir(goalDir)
    imgDir_n=osp.join(goalDir, 'imgs')
    makeDir(imgDir_n)

    xml_list=glob.glob(osp.join(xmlDir, '*.xml'))

    random.shuffle(xml_list)

    nums=len(xml_list)
    anno,label_list=convert(imgDir, xml_list, imgDir_n)
    with open(osp.join(goalDir, 'usual.json'),'w', encoding='utf-8')as f:
        f.write(json.dumps(anno))
    print('usual.json is done! nums=[%d]'%nums)

    label2id={}
    for label in label_list:
        label2id[label]=int(input('{} :'.format(label)).strip())
    with open(osp.join(goalDir, 'label2id.json'),'w', encoding='utf-8')as f:
        f.write(json.dumps(label2id))



if __name__=='__main__':
    if len(sys.argv)>0:
        func=getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('wrong')

 