import os,sys
import os.path as osp
import shutil
import json

def classify_cluster(jsonFile, imgDir):
    '''jsonFile:ans of cluster
    imgDir: dir of sub_dir
    '''
    res=json.loads(open(jsonFile).read())
    for label_id, label_dct in res.items():
        goalDir=osp.join(imgDir, label_id)
        os.makedirs(goalDir, exist_ok=True)
        for sgDct in label_dct['image_detail']:
            shutil.copy(sgDct['imgpath'], goalDir)


if __name__=='__main__':
    if len(sys.argv)>1:
        func=getattr(sys.modules[__name__], sys.argv[1])
        func(*(sys.argv[2:]))
    else:
        print('wrong!')