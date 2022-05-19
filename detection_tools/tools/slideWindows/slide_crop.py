'''
based on usual.json
convert_tools can be get in ../convert_format/*,which just support coco/xml
'''
import os,sys
import os.path as osp
import cv2
import json
import shutil
import collections
import json
import pdb

def searchInsert(nums, target):
    '''二分排序寻找插入位置
    '''
    flag=0
    left = 0 

    right = len(nums) - 1
    while left <= right:
        middle = (left + right) // 2
        if nums[middle] < target:
            left = middle + 1
        elif nums[middle] > target:
            right = middle - 1
        else:
            flag=1
            return middle+1, flag
    return right+1,flag

#13-7f83edb9-d27e-11ec-b00f-b025aa3fb8e8.jpg
#10-7f83db66-d27e-11ec-9cc2-b025aa3fb8e8.jpg
def slideCrop(imgPath,goal_dir, dct_lst, scale=640, thre=0.75):
    '''
    scale: size of img
    thre:threshold for bbox in over 1 img
    '''
    ansDct=collections.defaultdict(list)

    os.makedirs(goal_dir,exist_ok=True)
    imgname=osp.basename(imgPath)
    img=cv2.imread(imgPath)
    h,w,_=img.shape

    #calcuNum in row/col
    w_scale=scale
    h_scale=scale
    row_n=w//w_scale if w%w_scale==0 else w//w_scale+1
    col_n=h//h_scale if h%h_scale==0 else h//h_scale+1

    #暂存宽高起点;问题：x2不应该共用这个表
    w_s,h_s=w//row_n, h//col_n
    w_lst=[]
    h_lst=[]
    w_2lst=[0]
    h_2lst=[0]
    for i in range(row_n):
        w_lst.append(i*w_s)
        w_2lst.append(min(i*w_s+w_scale, w))
    for i in range(col_n):
        h_lst.append(i*h_s)
        h_2lst.append(min(i*h_s+h_scale, h))
    print('图片横向 {} 个；纵向 {} 个'.format(row_n, col_n))
    
    #搞定图片先[[1,2,3],[4,5,6]]
    #row_n is num, w_s is 宽高起点
    for i in range(col_n):
        for j in range(row_n):
            img_id=str(i*row_n+j+1)
           #边缘越界判断
            h_need=min(i*h_s+h_scale, h)
            w_need=min(j*w_s+w_scale, w)

            img_temp=img[i*h_s:h_need, j*w_s:w_need,:]
            cv2.imwrite(osp.join(goal_dir, img_id+'-'+imgname), img_temp)
    
    #w_lst,h_lst判断框从属
    for sgDct in dct_lst:
        x1_b,y1_b,w_b,h_b=sgDct['bbox']
        s_b=w_b*h_b
        x2_b,y2_b=x1_b+w_b,y1_b+h_b
        (x1_pos,flag_x1),(x2_pos,flag_x2)=searchInsert(w_lst,x1_b), searchInsert(w_2lst,x2_b)
        (y1_pos, flag_y1),(y2_pos, flag_y2)=searchInsert(h_lst,y1_b), searchInsert(h_2lst,y2_b)
        
        if x1_pos<x2_pos:
            w_l=w_2lst[x1_pos]-x1_b
            try:
                w_r=x2_b-w_lst[x2_pos-1] if (x2_pos-1)<len(w_lst) else 0
            except:
                pdb.set_trace()
            if y1_pos<y2_pos:
                h_t=h_2lst[y1_pos]-y1_b
                h_d=y2_b-h_lst[y2_pos]
                if w_l*h_t>thre*s_b and w_l>0 and h_t>0:
                    #左上
                    img_id=(y1_pos-1)*row_n+x1_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(0,x1_b-w_lst[x1_pos-1]),max(0,y1_b-h_lst[y1_pos-1]), w_l, h_t]
                        }
                    )
                if w_l*h_d>thre*s_b and w_l>0 and h_d>0:
                    #左下
                    img_id=(y2_pos-1)*row_n+x1_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(0,x1_b-w_lst[x1_pos-1]),max(0,y1_b-h_lst[y2_pos-1]), w_l, h_d]
                        }
                    )
                if w_r*h_t>thre*s_b and w_r>0 and h_t>0:
                    #右上
                    img_id=(y1_pos-1)*row_n+x2_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(0,x1_b-w_lst[x2_pos-1]),max(0,y1_b-h_lst[y1_pos-1]), w_r, h_t]
                        }
                    )
                if w_r*h_d>thre*s_b and w_r>0 and h_d>0:
                    #右下
                    img_id=(y2_pos-1)*row_n+x2_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(0,x1_b-w_lst[x2_pos-1]),max(0,y1_b-h_lst[y2_pos-1]), w_r, h_t]
                        }
                    )
            elif y1_pos==y2_pos:
                h_t=h_2lst[y1_pos-1]-y1_b if y1_pos>0 else 0
                h_m=h_b
                h_d=y2_b-h_lst[y1_pos+1] if y1_pos+1<len(h_lst) else 0
                if w_l*h_t>thre*s_b and w_l>0 and h_t>0:
                    #左上
                    img_id=(y1_pos-1-1)*row_n+x1_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(0,x1_b-w_lst[x1_pos-1]),max(0,y1_b-h_lst[y1_pos-1-1]), w_l, h_t]
                        }
                    )
                if w_l*h_d>thre*s_b and w_l>0 and h_d>0:
                    #左下
                    img_id=(y2_pos-1+1)*row_n+x1_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(0,x1_b-w_lst[x1_pos-1]),max(0,y1_b-h_lst[y2_pos+1-1]), w_l, h_d]
                        }
                    )
                if w_l*h_m>thre*s_b and w_l>0 and h_m>0:
                    #左中
                    img_id=(y1_pos-1)*row_n+x1_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(0,x1_b-w_lst[x1_pos-1]),max(0,y1_b-h_lst[y2_pos-1]), w_l, h_m]
                        }
                    )
                if w_r*h_t>thre*s_b and w_r>0 and h_t>0:
                    #右上
                    img_id=(y1_pos-1-1)*row_n+x2_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(x1_b-w_lst[x2_pos-1],0),max(0,y1_b-h_lst[y1_pos-1-1]), w_r, h_t]
                        }
                    )
                if w_r*h_d>thre*s_b and w_r>0 and h_d>0:
                    #右下
                    img_id=(y2_pos+1-1)*row_n+x2_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(x1_b-w_lst[x2_pos-1],0),max(0,y1_b-h_lst[y2_pos+1-1]), w_r, h_d]
                        }
                    )
                if w_r*h_m>thre*s_b and w_r>0 and h_m>0:
                    #右中
                    img_id=(y2_pos-1)*row_n+x2_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(x1_b-w_lst[x2_pos-1],0),max(0,y1_b-h_lst[y2_pos-1]), w_r, h_m]
                        }
                    )
            else: #y1_pos>y2_pos；必然都有
                h_t,h_d=h_b,h_b
                if w_l*h_t>thre*s_b:
                    #l 上
                    img_id=(y2_pos-1)*row_n+x1_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(0,x1_b-w_lst[x1_pos-1]),max(0,y1_b-h_lst[y2_pos-1]), w_l, h_t]
                        }
                    )
                if w_l*h_d>thre*s_b:
                    #l下
                    img_id=(y1_pos-1)*row_n+x1_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(0,x1_b-w_lst[x1_pos-1]),max(0,y1_b-h_lst[y1_pos-1]), w_l, h_d]
                        }
                    )
                if w_r*h_t>thre*s_b:
                    #右上
                    img_id=(y2_pos-1)*row_n+x2_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(0,x1_b-w_lst[x2_pos-1]),max(0,y1_b-h_lst[y2_pos-1]), w_r, h_t]
                        }
                    )
                if w_r*h_d>thre*s_b:
                    #右下
                    img_id=(y1_pos-1)*row_n+x2_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(0,x1_b-w_lst[x2_pos-1]),max(0,y1_b-h_lst[y1_pos-1]), w_r, h_d]
                        }
                    )
        elif x1_pos==x2_pos:
            w_l=w_2lst[x1_pos-1]-x1_b if x1_pos>0 else 0
            w_m=w_b
            w_r=x2_b-w_lst[x1_pos] if x1_pos+1<=len(w_lst) else 0
            if y1_pos<y2_pos:
                h_t=h_2lst[y1_pos]-y1_b
                try:
                    h_d=y2_b-h_lst[y2_pos-1]
                except:
                    pdb.set_trace()
                if w_l*h_t>thre*s_b and w_l>0 and h_t>0:
                    #左上
                    img_id=(y1_pos-1)*row_n+x2_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(x1_b-w_lst[x2_pos-1-1], 0),max(0,y1_b-h_lst[y1_pos-1]), w_l, h_t]
                        }
                    )
                if w_r*h_t>thre*s_b and w_r>0 and h_t>0:
                    #右上
                    img_id=(y1_pos-1)*row_n+x2_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(x1_b-w_lst[x2_pos+1-1], 0),max(0,y1_b-h_lst[y1_pos-1]), w_r, h_t]
                        }
                    )
                if w_m*h_t>thre*s_b and w_m>0 and h_t>0:
                    #中上
                    img_id=(y1_pos-1)*row_n+x2_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(0,x1_b-w_lst[x2_pos-1]),max(0,y1_b-h_lst[y1_pos-1]), w_m, h_t]
                        }
                    )
                if w_l*h_d>thre*s_b and w_l>0 and h_d>0:
                    #左xia
                    img_id=(y2_pos-1)*row_n+x2_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(0,x1_b-w_lst[x2_pos-1-1]),max(0,y1_b-h_lst[y2_pos-1]), w_l, h_d]
                        }
                    )
                if w_r*h_d>thre*s_b and w_r>0 and h_d>0:
                    #右xia
                    img_id=(y2_pos-1)*row_n+x2_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(0,x1_b-w_lst[x2_pos+1-1]),max(0,y1_b-h_lst[y2_pos-1]), w_r, h_d]
                        }
                    )
                if w_m*h_d>thre*s_b and w_m>0 and h_d>0:
                    #中xia
                    img_id=(y2_pos-1)*row_n+x2_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(x1_b-w_lst[x2_pos-1],0),max(0,y1_b-h_lst[y2_pos-1]), w_m, h_d]
                        }
                    )
            elif y1_pos==y2_pos:
                h_t=h_2lst[y1_pos-1]-y1_b if y1_pos>0 else 0
                h_m=h_b
                h_d=y2_b-h_lst[y1_pos+1] if y1_pos+1<len(h_lst) else 0
                if w_l*h_t>thre*s_b and w_l>0 and h_t>0:
                    #左上
                    img_id=(y1_pos-1-1)*row_n+x2_pos-1
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(0,x1_b-w_lst[x2_pos-1-1]),max(0,y1_b-h_lst[y2_pos-1-1]), w_l, h_t]
                        }
                    )
                if w_l*h_m>thre*s_b and w_l>0 and h_m>0:
                    #左zhong
                    img_id=(y1_pos-1)*row_n+x2_pos-1
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(x1_b-w_lst[x2_pos-1-1],0),max(y1_b-h_lst[y1_pos-1],0), w_l, h_m]
                        }
                    )
                if w_l*h_d>thre*s_b and w_l>0 and h_d>0:
                    #左xia
                    img_id=(y1_pos-1+1)*row_n+x2_pos-1
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(0,x1_b-w_lst[x2_pos-1-1]),max(0,y1_b-h_lst[y1_pos+1-1]), w_l, h_d]
                        }
                    )
                if w_m*h_t>thre*s_b and w_m>0 and h_t>0:
                    #zhong上
                    img_id=(y1_pos-1-1)*row_n+x2_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(0,x1_b-w_lst[x2_pos-1]),max(0,y1_b-h_lst[y1_pos-1-1]), w_m, h_t]
                        }
                    )
                if w_m*h_m>thre*s_b and w_m>0 and h_m>0:
                    #zhongzhong
                    img_id=(y1_pos-1)*row_n+x2_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(0,x1_b-w_lst[x2_pos-1]),max(0,y1_b-h_lst[y2_pos-1]), w_m, h_m]
                        }
                    )
                if w_m*h_d>thre*s_b and w_m>0 and h_d>0:
                    #zhongxia
                    img_id=(y1_pos-1+1)*row_n+x2_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(0,x1_b-w_lst[x2_pos-1]),max(0,y1_b-h_lst[y1_pos+1-1]), w_m, h_d]
                        }
                    )
                if w_r*h_t>thre*s_b and w_r>0 and h_t>0:
                    #you上
                    img_id=(y1_pos-1-1)*row_n+x2_pos+1
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(x1_b-w_lst[x2_pos+1-1],0),max(0,y1_b-h_lst[y1_pos-1-1]), w_r, h_t]
                        }
                    )
                if w_r*h_m>thre*s_b and w_r>0 and h_m>0:
                    #youzhong
                    img_id=(y1_pos-1)*row_n+x2_pos+1
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(x1_b-w_lst[x2_pos+1-1],0),max(0,y1_b-h_lst[y1_pos-1]), w_r, h_m]
                        }
                    )
                if w_r*h_d>thre*s_b and w_r>0 and h_d>0:
                    #you xia
                    img_id=(y1_pos-1+1)*row_n+x2_pos+1
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(x1_b-w_lst[x2_pos+1-1], 0),max(0,y1_b-h_lst[y1_pos+1-1]), w_r, h_d]
                        }
                    )
            else: #y1>y2
                h_t,h_d=h_b,h_b
                if w_l*h_t>thre*s_b and w_l>0 and h_t>0:
                    #l t
                    img_id=(y2_pos-1)*row_n+x2_pos-1
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(0,x1_b-w_lst[x2_pos-1-1]),max(0,y1_b-h_lst[y2_pos-1]), w_l, h_t]
                        }
                    )
                if w_l*h_d>thre*s_b and w_l>0 and h_d>0:
                    #l d
                    img_id=(y1_pos-1)*row_n+x2_pos-1
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(0,x1_b-w_lst[x2_pos-1-1]),max(0,y1_b-h_lst[y1_pos-1]), w_l, h_d]
                        }
                    )
                if w_m*h_t>thre*s_b and w_m>0 and h_t>0:
                    #m t
                    img_id=(y2_pos-1)*row_n+x2_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(0, x1_b-w_lst[x2_pos-1]),max(0, y1_b-h_lst[y2_pos-1]), w_m, h_t]
                        }
                    )
                if w_m*h_d>thre*s_b and w_m>0 and h_d>0:
                    #m d
                    img_id=(y1_pos-1)*row_n+x2_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(0,x1_b-w_lst[x2_pos-1]),max(0,y1_b-h_lst[y1_pos-1]), w_m, h_d]
                        }
                    )
                if w_r*h_t>thre*s_b and w_r>0 and h_t>0:
                    #r t
                    img_id=(y2_pos-1)*row_n+x2_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(0,x1_b-w_lst[x2_pos+1-1]),max(0,y1_b-h_lst[y2_pos-1]), w_r, h_t]
                        }
                    )
                if w_r*h_d>thre*s_b and w_r>0 and h_d>0:
                    #r d
                    img_id=(y1_pos-1)*row_n+x2_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(0,x1_b-w_lst[x2_pos+1-1]),max(0,y1_b-h_lst[y1_pos-1]), w_r, h_d]
                        }
                    )
        else: #x1_pos>x2_pos
            w_l,w_r=w_b, w_b
            if y1_pos<y2_pos:
                h_t=h_2lst[y1_pos]-y1_b
                h_d=y2_b-h_lst[y2_pos]
                if w_l*h_t>thre*s_b and w_l>0 and h_t>0:
                    #l t
                    img_id=(y1_pos-1)*row_n+x2_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[x1_b-w_lst[x2_pos-1],y1_b-h_lst[y1_pos-1], w_l, h_t]
                        }
                    )
                if w_l*h_d>thre*s_b and w_l>0 and h_d>0:
                    #l d
                    img_id=(y2_pos-1)*row_n+x2_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[x1_b-w_lst[x2_pos-1],y1_b-h_lst[y2_pos-1], w_l, h_d]
                        }
                    )
                if w_r*h_t>thre*s_b and w_r>0 and h_t>0:
                    #r t
                    img_id=(y1_pos-1)*row_n+x1_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[x1_b-w_lst[x1_pos-1],y1_b-h_lst[y1_pos-1], w_r, h_t]
                        }
                    )
                if w_r*h_d>thre*s_b and w_r>0 and h_d>0:
                    #r d
                    img_id=(y2_pos-1)*row_n+x1_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[x1_b-w_lst[x1_pos-1],y1_b-h_lst[y2_pos-1], w_r, h_d]
                        }
                    )
            elif y1_pos==y2_pos:
                h_t=h_2lst[y1_pos-1]-y1_b if y1_pos>1 else 0
                h_m=h_b
                h_d=y2_b-h_lst[y1_pos] if y1_pos+1<=len(h_lst) else 0
                if w_l*h_t>thre*s_b and w_l>0 and h_t>0:
                    #l t
                    img_id=(y1_pos-1-1)*row_n+x2_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[x1_b-w_lst[x2_pos-1],max(y1_b-h_lst[y2_pos-1-1], 0), w_l, h_t]
                        }
                    )
                if w_l*h_d>thre*s_b and w_l>0 and h_d>0:
                    #l d
                    img_id=(y2_pos-1+1)*row_n+x2_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[x1_b-w_lst[x2_pos-1],max(y1_b-h_lst[y2_pos+1-1],0), w_l, h_d]
                        }
                    )
                if w_l*h_m>thre*s_b and w_l>0 and h_m>0:
                    #l m
                    img_id=(y1_pos-1)*row_n+x2_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[max(0,x1_b-w_lst[x2_pos-1]),max(0,y1_b-h_lst[y2_pos-1]), w_l, h_m]
                        }
                    )
                if w_r*h_t>thre*s_b and w_r>0 and h_t>0:
                    #r t
                    img_id=(y2_pos-1-1)*row_n+x1_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[x1_b-w_lst[x1_pos-1],y1_b-h_lst[y2_pos-1-1], w_r, h_t]
                        }
                    )
                if w_r*h_m>thre*s_b and w_r>0 and h_m>0:
                    #r m
                    img_id=(y1_pos-1)*row_n+x1_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[x1_b-w_lst[x1_pos-1],y1_b-h_lst[y2_pos-1], w_r, h_m]
                        }
                    )
                if w_r*h_d>thre*s_b and w_r>0 and h_d>0:
                    #r d
                    img_id=(y2_pos-1+1)*row_n+x1_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[x1_b-w_lst[x1_pos-1],max(y1_b-h_lst[y2_pos+1-1],0), w_r, h_d]
                        }
                    )
            else: #y1_pos>y2_pos；必然都有
                h_t,h_d=h_b,h_b
                if w_l*h_t>thre*s_b and w_l>0 and h_t>0:
                    #l t
                    img_id=(y2_pos-1)*row_n+x2_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[x1_b-w_lst[x2_pos-1],y1_b-h_lst[y2_pos-1], w_l, h_t]
                        }
                    )
                if w_l*h_d>thre*s_b and w_l>0 and h_d>0:
                    #l d
                    img_id=(y1_pos-1)*row_n+x2_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[x1_b-w_lst[x2_pos-1],y1_b-h_lst[y1_pos-1], w_l, h_d]
                        }
                    )
                if w_r*h_t>thre*s_b and w_r>0 and h_t>0:
                    #r t
                    img_id=(y2_pos-1)*row_n+x1_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[x1_b-w_lst[x1_pos-1],y1_b-h_lst[y2_pos-1], w_r, h_t]
                        }
                    )
                if w_r*h_d>thre*s_b and w_r>0 and h_d>0:
                    #r d
                    img_id=(y1_pos-1)*row_n+x1_pos
                    ansDct[str(img_id)+'-'+imgname].append(
                        {
                            'category':sgDct['category'],
                            'bbox':[x1_b-w_lst[x1_pos-1],y1_b-h_lst[y1_pos-1], w_r, h_d]
                        }
                    )
# #13-7f83edb9-d27e-11ec-b00f-b025aa3fb8e8.jpg
        # if imgname=='7f83db66-d27e-11ec-9cc2-b025aa3fb8e8.jpg' and img_id==10:
        # if w_l==26 or w_m==26 or w_r==26 and (h_t==24 or h_d==24 or h_m==24):
        #     pdb.set_trace()
        #     print(img_id)
    return ansDct

def slideCropDir(img_dir, goal_dir, jsonPath, scale=640, thre=0.6):
    '''img_dir: origin path of imgs
    goal_dir:
        imgs(split)
        usual.json(split)
    jsonPath: usual.json(original)
    scale: crop_img h,w; can be set separately in sub_function
    '''
    scale=int(scale)
    thre=float(thre)

    goal_imgDir=osp.join(goal_dir, 'imgs')
    gen_jsonPath=osp.join(goal_dir, 'usual.json')
    os.makedirs(goal_imgDir, exist_ok=True)

    res_split={}

    res=json.loads(open(jsonPath).read())
    for imgname, dct_lst in res.items():
        img_path=osp.join(img_dir, imgname)
        if osp.isfile(img_path):
            res_split.update(slideCrop(img_path, goal_imgDir,dct_lst,scale,thre))
    
    for imgname in os.listdir(goal_imgDir):
        if imgname not in res_split:
            res_split[imgname]=[]
    with open(gen_jsonPath, 'w', encoding='utf-8')as f:
        f.write(json.dumps(res_split))

if __name__=='__main__':
    if len(sys.argv)>1:
        func=getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('input wrong')

            









                
            
                
                








    






