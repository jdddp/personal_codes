import os
import os.path as osp
import sys
import cv2
from PIL import Image
from tqdm import tqdm
import uuid
import numpy as np
import math
import multiprocessing
from common_use import get_imagelist, makeDir

def preprocess(info_tp):
    imgpath, short_size, dstpath=info_tp
    img=cv2.imread(imgpath)
    # img=cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)

    if short_size != 0:
        h,w,_=img.shape
        if min(h,w)>short_size:
            ratio=float(short_size)/min(h,w)
            w_n,h_n=math.floor(w*ratio),math.floor(h*ratio)
            img=cv2.resize(img, (w_n,h_n))
    
    cv2.imwrite(dstpath, img)

def formatDir(dirpath, outdir, format='dir', \
    short_size=0, reserve_name=0):
    imglist=get_imagelist(dirpath,format)
    print('get images %d'% len(imglist))
    os.makedirs(outdir, exist_ok=True)
    short_size, reserve_name=int(short_size), int(reserve_name)
    if reserve_name==0:
        info_tp=[(p, short_size, osp.join(outdir,osp.basename(p).split('.')[0]+'.jpg')) \
            for p in imglist]
    else:
        info_tp=[(p, short_size, osp.join(outdir,str(uuid.uuid1())+'.jpg')) \
            for p in imglist]
    
    pool=multiprocessing.Pool(8)
    for _ in tqdm(pool.imap_unordered(preprocess, info_tp)):
        pass

#test
def format(imgpath, outdir):
    short_size=0
    info_tp=(imgpath, short_size, osp.join(outdir, str(uuid.uuid1())+'.jpg'))
    preprocess(info_tp)


if __name__ == '__main__':
    if len(sys.argv) > 1 :
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('wrong!')