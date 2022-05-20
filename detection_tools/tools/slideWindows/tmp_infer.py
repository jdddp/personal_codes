import os,sys
import os.path as osp
sys.path.append('/home/linke/codes/localCodes/mmdetection')
from mmdet.apis import init_detector, inference_detector
import mmcv
import glob
import pdb
import numpy as np
import json
import collections
import math
import time

config_file='/home/linke/codes/localCodes/mmdetection/configs/a_projects/tjhv1_yolov1.py'
checkpoint_file='/home/linke/codes/localCodes/mmdetection/output/lh_test/best_bbox_mAP_epoch_17.pth'
thre=0.5
img_dir='/home/linke/codes/localCodes/dataset/lh_test_split/imgs_test'



model=init_detector(config_file, checkpoint_file,device='cuda:0')
# if osp.isdir(img_dir):
#     img_lst=glob.glob(osp.join(img_dir, '*.jpg'))
#     # print(img_lst)
# else:
#     img_lst=[img_dir]
# print(img_lst)
img_path='/home/linke/codes/localCodes/dataset/lh_test_split/imgs_test/2-imgE.jpg'
img_lst=[mmcv.imread(img_path)]
# pdb.set_trace()
ansDCt=collections.defaultdict(list)
time_lst=[]
for img_path in img_lst:
    img_name=osp.basename(img_path)
    start_time=time.time()
    result=inference_detector(model, img_path)

    bbox_result=result
    bboxes=np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels=np.concatenate(labels)
    class_names=('influid')


    # img=mmcv.imread(img).astype(np.unint8)
    scores=bboxes[:,-1]
    inds=scores>thre
    bboxes = bboxes[inds, :]
    labels = labels[inds]
    for i, bbox_score in enumerate(bboxes):
        label_id=labels[i]
        label='fluid_inclusions'
        # temp
        ansDCt[img_name].append(
            {
                'category':label,
                'score': float(bbox_score[-1]),
                'bbox': [math.floor(bbox_score[0]),math.floor(bbox_score[1]), math.ceil(bbox_score[2]-bbox_score[0]), math.ceil(bbox_score[3]-bbox_score[1])]
            }
        )
    # if img_name not in ansDCt:
    #     ansDCt[img_name]=[]
    # time_lst.append(time.time()-start_time)
# print(time_lst)
# with open('/home/linke/codes/localCodes/dataset/lh_test_split/test_usual0_5.json', 'w', encoding='utf-8')as f:
#     f.write(json.dumps(ansDCt))
print(ansDCt)


