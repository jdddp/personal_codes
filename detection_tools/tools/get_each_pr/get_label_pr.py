'''
基本前提条件：本次项目各个目标物体之间几乎不存在重叠，即使重叠，交并比也是极小的
'''
import os
import sys
import os.path as osp
import json
import collections
import pandas as pd

def if_iou(list1,list2, threshold=0.5):
    w1,h1=list1[2:]
    w2,h2=list2[2:]
    xl=max(list1[0],list2[0])
    yl=max(list1[1],list2[1])
    xr=min(list1[0]+w1,list2[0]+w2)
    yr=min(list1[1]+h1,list2[1]+h2)


    s_i=max((xr-xl),0)*max((yr-yl), 0)
    s_o=w1*h1+w2*h2-s_i
    return s_i/s_o > threshold

def json2excel(res, excel_path):
    '''convert json to excel
    '''
    # res=json.loads(open(jsonPath).read())
    with pd.ExcelWriter(excel_path)as writer:
        for name, sgDct in res.items():
            df=pd.DataFrame.from_dict(sgDct)
            df.to_excel(writer, sheet_name=name)

def get_args_for_pr(dct_gt, dct_pred):
    '''args: 
    dct_gt:
    {
        'gt':{
            'label1':1,
            'label2':1
        }
    }
    {
        'label1':{
            'label1':1,
            'label2':1,
            'miss':1 #漏检的
        }
    }
    '''
    num_to_cal=collections.defaultdict(dict)
    pred_ans=collections.defaultdict(dict)

    #get label_num of each label
    for img_name,dct_lst in dct_gt.items():
        for dct in dct_lst:
            if dct['category'] in num_to_cal['gt']:
                num_to_cal['gt'][dct['category']]+=1
            else:
                num_to_cal['gt'][dct['category']]=1


    #get label_num of each label
    for img_name,dct_lst in dct_pred.items():
        for dct in dct_lst:
            if dct['category'] in num_to_cal['pred']:
                num_to_cal['pred'][dct['category']]+=1
            else:
                num_to_cal['pred'][dct['category']]=1
    
    #get pred_ans of each_label
    for img_name, dct_lst_pred in dct_pred.items():
        dct_lst_gt=dct_gt[img_name]

        #
        useless_idx=set()
        #遍历gt
        for sub_dct_gt in dct_lst_gt:
            #标识符，没找到的话，这个属于漏召了
            flag=1
            for i in range(len(dct_lst_pred)):
                if i in useless_idx:
                    continue
                sub_dct_pred=dct_lst_pred[i]

            # for sub_dct_pred in dct_lst_pred:
                if if_iou(sub_dct_gt['bbox'], sub_dct_pred['bbox']):
                    # if sub_dct_gt['category']==sub_dct_pred['category']:
                    flag=0
                    useless_idx.add(i)
                    # dct_lst_pred.remove(sub_dct_pred)
                    if sub_dct_pred['category'] in pred_ans[sub_dct_gt['category']]:
                        pred_ans[sub_dct_gt['category']][sub_dct_pred['category']]+=1
                    else:
                        pred_ans[sub_dct_gt['category']][sub_dct_pred['category']]=1
                    break
        
            if flag==1:
                if 'missing' in pred_ans[sub_dct_gt['category']]:
                    pred_ans[sub_dct_gt['category']]['missing']+=1
                else:
                    pred_ans[sub_dct_gt['category']]['missing']=1
    print(pred_ans)
    return (num_to_cal, pred_ans)


def get_pr(gt_pred_dct, pred_distrib_dct, excel_path):
    '''
    arg1,arg2=get_args_for_pr(...)
    表格设计
        label1 label2 label3 label4 missing num_gt num_pred prec recall
    label1
    label2
    label3
    label4
    '''
    pr_out={}
    for label,label2num in pred_distrib_dct.items():
        gt_num,pred_num=gt_pred_dct['gt'][label], gt_pred_dct['pred'].get(label, 0)
        precision=float(label2num.get(label, 0))/pred_num if pred_num!=0 else 0
        recall  =float(label2num.get(label, 0))/gt_num if gt_num!=0 else 0
        dct_tmp={
            'gt_num':gt_num,
            'pred_num':pred_num,
            'precison':round(precision,4),
            'recall':round(recall,3)
        }
        pr_out[label]=dct_tmp
        pred_distrib_dct[label].update(dct_tmp)
    
    excel_json={'sheet1':[]}
    for k,v in pred_distrib_dct.items():
        v.update({'row_title':k})
        excel_json['sheet1'].append(v)
    
    # print(excel_json)
    json2excel(excel_json, excel_path)
    print('生成表格地址{}'.format(excel_path))
    print(pr_out)

def main(gt_json_path, pred_json_path, excel_path):
    gt_res=json.loads(open(gt_json_path).read())
    pred_res=json.loads(open(pred_json_path).read())
    arg1,arg2=get_args_for_pr(gt_res,pred_res)
    get_pr(arg1,arg2, excel_path)

if __name__ == '__main__':
    if len(sys.argv)>1 :
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('wrong!')



#python path/to/get_label_pr.py main path/to/gt.json path/to/pred.json path/for/excel
    
