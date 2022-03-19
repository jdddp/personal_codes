import os
import os.path as osp
import shutil
import torch
import torch.nn as nn
import numpy as np
import collections
import time

from dataset.dataset import default_loader
from basic.log import LoggerInfer, Logger


def sort_parallel(list1,list2):
    '''
    输出设计成[{scorelist:[],imgpath:"",label_list:""},{},...]
    将score_list按得分排序，label_list与之匹配
    '''
    zipped=zip(list1,list2)
    sort_zipped=sorted(zipped, key=lambda x:(x[0],x[1]), reverse=True)
    result=list(zip(*sort_zipped))
    # print(result)
    return list(result[0]), list(result[1])

def design_result(results,label_list,imgnames,ans_nums):
    assert ans_nums<=len(label_list)
    ans=[]
    for i, output in enumerate(results):
        imgPath=imgnames[i]
        output_n,label_list_n=sort_parallel(output, label_list)
        ans.append({'score_list':output_n[:ans_nums], 'label_list':label_list_n[:ans_nums], 'imgpath':imgPath})
    return ans


def infer(dataloader, model,label_list,ans_nums,projectPath):
    log=Logger(osp.join(projectPath,'infer-'+time.strftime("%H-%M")+'.log'))
    # log=LoggerInfer(osp.join(projectPath,'infer-'+time.strftime("%H-%M")+'.log'))

    device=torch.device('cuda:0')
    with torch.no_grad():
        model.eval()
        for step, (inputs,imgnames) in enumerate(dataloader):
            inputs = inputs.to(device)
            # labels = labels.to(device)

            outputs = model(inputs)
            outputs=nn.Softmax(dim=1)(outputs)
            outputs = outputs.cpu().numpy().tolist()

            ans=design_result(outputs, label_list, imgnames, ans_nums)
            log.print(ans)
            
            # time.sleep(2)


            
            
    
    