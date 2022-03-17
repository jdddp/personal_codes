import os
import sys
import os.path as osp
import shutil
import random
import json
import cv2
import math
from tqdm import tqdm

def makeDir(dirpath):
    if not osp.exists(dirpath):
        os.makedirs(dirpath)

def makeMapFileRandom(dirpath, filePath):
    '''dirpath: path of dataset
    filePath: path of [id<->label]
    '''
    id2label={}
    label2id={}
    for i,label in enumerate(os.listdir(dirpath)):
        id2label[str(i)]=label
        label2id[label]=str(i)

    if osp.isfile(osp.join(filePath, 'id2label.txt')):
        os.system('rm {}'.format(osp.join(filePath, 'id2label.txt')))
    txtFile=open(osp.join(filePath, 'id2label.txt'), 'a', encoding='utf-8')
    for label, idd in label2id.items():
        txtFile.write(label+' '+idd+'\n')


    with open(osp.join(filePath, 'id2label.json'), 'w')as f:
        f.write(json.dumps(id2label))
    with open(osp.join(filePath, 'label2id.json'), 'w')as f:
        f.write(json.dumps(label2id))

def makeMapFile(dirpath, filePath):
    '''dirpath: path of dataset
    filePath: path of [id<->label]
    '''
    label2id={}
    id2label={}
    for label in os.listdir(dirpath):
        label2id[label]=str((input('{} :'.format(label))).strip())
    
    for label, idd in label2id.items():
        id2label[idd]=label

    txtFile=open(osp.join(filePath, 'id2label.txt'), 'a', encoding='utf-8')
    for label, idd in label2id.items():
        txtFile.write(label+' '+idd+'\n')

    with open(osp.join(filePath, 'id2label.json'), 'w')as f:
        f.write(json.dumps(id2label))
    with open(osp.join(filePath, 'label2id.json'), 'w')as f:
        f.write(json.dumps(label2id))

def makeTrainFiles(dirpath, filePath,ratio=0.9):
    '''dirpath: path of dataset
    filePath: path of [id<->label]
    '''
    ratio=float(ratio)
    label2id=json.loads(open(osp.join(filePath, 'label2id.json')).read())
    train_list=[]
    val_list=[]
    for label in tqdm(label2id):
        temp_list=[]
        label_id=label2id[label]
        for imgname in os.listdir(osp.join(dirpath,label)):
            if cv2.imread(osp.join(dirpath,label,imgname)) is not None:
                temp_list.append(label+'/'+imgname+'\t'+label_id)
                random.shuffle(temp_list)
                
                train_list.extend(temp_list[0:math.floor(len(temp_list)*ratio)])
                val_list.extend(temp_list[math.floor(len(temp_list)*ratio):])
            else:
                print(osp.join(dirpath,label,imgname))
    
    random.shuffle(train_list)
    random.shuffle(train_list)
    random.shuffle(val_list)
    random.shuffle(val_list)

    f_train=open(osp.join(filePath, 'train.txt'), 'w', encoding='utf-8')
    f_train.write('\n'.join(train_list))
    print('train.txt is done! len is {}'.format(len(train_list)))

    f_val=open(osp.join(filePath, 'test.txt'), 'w', encoding='utf-8')
    f_val.write('\n'.join(val_list))
    print('test.txt is done! len is {}'.format(len(val_list)))

    


if __name__=="__main__":
    if len(sys.argv) > 1:
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('wrong!')



