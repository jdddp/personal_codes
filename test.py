import logging
import multiprocessing
import os
import time
from multiprocessing import TimeoutError
from pathlib import Path

# import numpy as np
# from sklearn.metrics.pairwise import euclidean_distances

# # a=np.ones((1,2048))
# # b=np.random.randint(1,size=(1,2048))
# # # print(type(euclidean_distances(a,b)))
# # print(euclidean_distances(a,b)[0][0])
# # print(euclidean_distances(a,b))

# a=np.random.randint(9,size=(3,3))
# print(a)
# import pdb 
# pdb.set_trace()
# a=1.0
# b=3
# print(a/b)
# print("%3f"%(a/b))
# print(type("%3f"%(a/b)))

# import numpy as np
# print(tuple(np.random.randint(256, size=3)))
#(79, 189, 86)

def cal_pre_recall(dctT,dctF):
    tp,fp,fn=0,0
    for k,v in dctT.items():
        if v==0:
            tp+=1
        else:
            fp+=1
    for k,v in dctF.item():
        if v==0:
            fn+=1
    prec=float(tp)/(tp+fp)
    recall=float(tp)/(tp+fn)
    return prec,recall

def list2dct(score_lst, label_lst):
    label2score={}
    for i in range(len(score_lst)):
        label2score[label_lst[i]]=score_lst[i]
    tp_lst=sorted(label2score.items(), lambda x:x[1],reverse=False)

    dct_600=dict(tp_lst[0:600])
    dct_back=dict(tp_lst[600:])

    pre,recall=cal_pre_recall(dct_600, dct_back)
    return pre,recall

'''
{
    'score'=[],
    'label'=[]
}
'''
import os.path as osp
import json
def cal_all(dirpath):
    for filename in os.listdir(dirpath):
        jsonPath=osp.join(dirpath,filename)
        res=json.loads(open(jsonPath).read())
        pre,rec=list2dct(res['score'], res['label'])
        print(filename,pre,rec)

    
'''
测试样本中，600（正）
对每个模型的输出得分进行sort(reverse=False)，每个对应的label也伴随sort
这时候取各自的第600个值作为判断阈值，分别计算pre、rec
'''

    

