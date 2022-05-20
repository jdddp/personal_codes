from operator import gt
import os
import os.path as osp
import sys
import json
from tqdm import tqdm

def iou(list1,list2,threshold):
    w1,h1=list1[2:]
    w2,h2=list2[2:]
    xl=max(list1[0],list2[0])
    yl=max(list1[1],list2[1])
    xr=min(list1[0]+w1,list2[0]+w2)
    yr=min(list1[1]+h1,list2[1]+h2)


    s_i=max((xr-xl),0)*max((yr-yl), 0)
    s_o=w1*h1+w2*h2-s_i
    return (float(s_i)/s_o)>threshold


def __find_mistake(gt_lst,pred_lst,threshold):
    ansDct={
        'miss':[],
        'mistake':[]
    }
    for sgDct in gt_lst:
        flag=False
        bbox_gt,cate_gt=sgDct['bbox'],sgDct['category']
        for predDct in pred_lst:
            bbox_pred, cate_pred=predDct['bbox'], predDct['category']
            if cate_pred==cate_gt:
                flag=iou(bbox_gt, bbox_pred, threshold)
                #找到预测就剔除预测；没找到，那这个框就是漏召回
                if flag:
                    pred_lst.remove(predDct)
                else:
                    continue
            else:
                continue
            if not flag:
                ansDct['miss'].append(sgDct)
    #找完之后剩下的即误召回
    ansDct['mistake'].extend(pred_lst)
    return ansDct



def find_mistake_box(gt_path,pred_path,ansJson,threshold=0.5):
    threshold=float(threshold)
    ansDct={}
    gt_res=json.loads(open(gt_path).read())
    pred_res=json.loads(open(pred_path).read())
    for imgname, dct_lst in tqdm(gt_res.items()):
        if len(dct_lst)==0:
            ansDct[imgname]={
                'miss':[],
                'mistake':pred_res[imgname] if imgname in pred_res else []
            }
        elif len(pred_res[imgname])==0 or imgname not in pred_res:
            ansDct[imgname]={
                'miss':gt_res[imgname],
                'mistake':[]
            }
        else:
            ansDct[imgname]=__find_mistake(dct_lst, pred_res[imgname], threshold)
    
    #分开算把;适用于单类别det;浪费些时间可与上合并
    gt=0
    pred=0
    tp=0
    incorrect_nums=0
    gt += len(dct_lst) for _,dct_lst in gt_res.items()
    pred += len(dct_lst) for _,dct_lst in pred_res.items()
    for _,sgDct in ansDct.items():
        incorrect_nums+=len(sgDct['mistake'])
    tp=pred-incorrect_nums
    print('*'*20)
    print('precison is {:4f}'.format(tp*1.0/gt))
    print('recall is {:4f}'.format(tp*1.0/pred))
    print('*'*20)
    

    with open(ansJson,'w',encoding='utf-8')as f:
        f.write(json.dumps(ansDct))

if __name__=='__main__':
    if len(sys.argv)>1:
        func=getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('input wrong')
            



