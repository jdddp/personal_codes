import os
import os.path as osp
import json
import sys

from kmeans import clusterBoxes


'''
boxlist不同来源
'''

def usualJson(jsonPath,k=9):
    '''通用json格式
    '''
    k=int(k)
    box_list=[]
    res=json.loads(open(jsonPath).read())
    for _,dctLst in res.items():
        if len(dctLst)>0:
            for sgDct in dctLst:
                box_list.append([sgDct['bbox'][2],sgDct['bbox'][3]])
        else:
            continue
    clusterBoxes(box_list,k)


if __name__ == '__main__':
    if len(sys.argv)>0:
        func=getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('wrong')