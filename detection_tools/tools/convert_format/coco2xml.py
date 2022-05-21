import os,sys
import os.path as osp
import shutil
from tqdm import tqdm
import cv2
import json
import pandas as pd
from pycocotools.coco import COCO

headstr = """\
<annotation>
    <folder>JEPGImages</folder>
    <filename>%s</filename>
    <source>
        <database>MyDatabase</database>
    </source>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""
 
tailstr = '''\
</annotation>
'''

def get_id2label(coco):
    id2label={}
    label_lst=[]
    for sgDct in coco.dataset['categories']:
        id2label[sgDct['id']] = sgDct['name']
        label_lst.append(sgDct['name'])
    return id2label,label_lst
 
def write_xml(anno_path,head, objs, tail):
    f = open(anno_path, "w")
    f.write(head)
    for obj in objs:
        f.write(objstr%(obj[0],obj[1],obj[2],obj[3],obj[4]))
    f.write(tail)
 
 
def save_annotations_and_imgs(xml_dir,coco_img_dir,img_name,objs):
    img_dir = osp.join(xml_dir, 'JEPGImages')
    anno_dir = osp.join(xml_dir, 'Annotations')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(anno_dir, exist_ok=True)
    
    anno_path=osp.join(anno_dir, img_name.split('.')[0]+'.xml')
    src_img_path=osp.join(coco_img_dir, img_name)
    dst_img_path=osp.join(img_dir,img_name)
    try:
        img=cv2.imread(src_img_path)
        if img is not None:
            shutil.copy(src_img_path, dst_img_path)
            head=headstr % (img_name, img.shape[1], img.shape[0], img.shape[2])
            tail = tailstr
            write_xml(anno_path,head, objs, tail)
        else:
            print(img_name)
    except Exception as e:
        print(e)

def coco2xml(coco_dir, xml_dir):
    coco_img_dir=osp.join(coco_dir, 'imgs')
    coco_anno_dir=osp.join(coco_dir, 'annos')
    for cocoJson in os.listdir(coco_anno_dir):
        annoFile=osp.join(coco_anno_dir, cocoJson)

        coco_res=json.loads(open(annoFile).read())
        img_list=coco_res['images']

        id2label={}
        for sgDct in coco_res['categories']:
            id2label[sgDct['id']]=sgDct['name']
        
        anno_tmp=pd.DataFrame(coco_res['annotations'])

        for i in tqdm(range(len(img_list))):
            img_name=img_list[i]['file_name']
            img_id=img_list[i]['id']

            annos=anno_tmp[anno_tmp["image_id"].isin([img_id])]
            objs=[]
            for _, row in annos.iterrows():
                bbox=row['bbox']
                cate_name=id2label[row['category_id']]
                x1,y1=int(bbox[0]),int(bbox[1])
                x2,y2=int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])
                objs.append([cate_name,x1,y1,x2,y2])

            save_annotations_and_imgs(xml_dir, coco_img_dir, img_name, objs)
        break

def usual2coco(img_dir, usualJsonPath, xml_dir):
    res=json.loads(open(usualJsonPath).read())
    for img_name in os.listdir(img_dir):
        sgDct_lst=res[img_name]
        objs=0
        for sgDct in sgDct_lst:
            bbox=sgDct['bbox']
            cate_name=sgDct['category']
            x1,y1=int(bbox[0]),int(bbox[1])
            x2,y2=int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])
            objs.append([cate_name,x1,y1,x2,y2])
        save_annotations_and_imgs(xml_dir, img_dir, img_name, objs)



if __name__=='__main__':
    if len(sys.argv)>0:
        func=getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('wrong')