from ast import arg
import os
import sys
import os.path as osp
import json
import cv2
import math
from multiprocessing import Pool
from tqdm import tqdm

def crop_img(src,dst,bbox,ratio=1.5):
    '''src: original path of img
    dst: path of cropped img
    bbox:[x1,y1,w,h]
    '''
    ratio=float(ratio)

    img=cv2.imread(src)
    h_o,w_o,_=img.shape
    x1,y1,w,h=bbox
    xc=float(w)/2+x1
    yc=float(h)/2+y1

    #越界判断
    x1_n=math.floor(max(0,xc-(ratio/2)*w))
    y1_n=math.floor(max(0,yc-(ratio/2)*h))

    x2_n=math.ceil(min(w_o,xc+(ratio/2)*w))
    y2_n=math.ceil(min(h_o,yc+(ratio/2)*h))
    # print(y1_n,y2_n,x1_n,x2_n)
    cropped=img[y1_n:y2_n,x1_n:x2_n]
    cv2.imwrite(dst, cropped)

def test1(imgPath,dstPath):
    bbox=[500,500,600,600]
    crop_img(imgPath,dstPath,bbox)

def det2cls(imgDir,usualJson,goalDir):
    tpLst=[]
    res=json.loads(open(usualJson).read())

    for imgname,dctLst in res.items():
        if len(dctLst)==0:
            continue
        imgPath=osp.join(imgDir, imgname)
        for i,sgDct in enumerate(dctLst):
            os.makedirs(osp.join(goalDir,sgDct['category']),exist_ok=True)
            dctPath=osp.join(goalDir, sgDct['category'],str(i)+'-'+osp.basename(imgname))
            tpLst.append((imgPath,dctPath,sgDct['bbox']))

    pool=Pool(8)
    for i in tqdm(range(len(tpLst))):
        # import pdb
        # pdb.set_trace()
        pool.apply_async(crop_img,args=tpLst[i])
    
    pool.close()
    pool.join()


if __name__ == '__main__':
    if len(sys.argv)>0:
        func=getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('wrong')

