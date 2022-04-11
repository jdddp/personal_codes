import os
import sys
import os.path as osp
import shutil
import uuid
from tqdm import tqdm

def makeDir(dirpath):
    if not osp.exists(dirpath):
        os.makedirs(dirpath)

def format_folder(dirpath, goalDir):
    makeDir(goalDir)
    for imgname in os.listdir(dirpath):
        if osp.splitext(imgname)[-1].lower() in ['.jpg', '.jpeg', '.png']:
            new_imgname=str(uuid.uuid4())+'.'+imgname.split('.')[-1]
            shutil.copy(osp.join(dirpath, imgname), osp.join(goalDir, new_imgname))

def format_folders(dirpath, goalDir):
    makeDir(goalDir)
    for label in tqdm(os.listdir(dirpath)):
        #pool加速
        format_folder(osp.join(dirpath, label), osp.join(goalDir, label))

if __name__=="__main__":
    if len(sys.argv) > 1:
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('wrong!')
