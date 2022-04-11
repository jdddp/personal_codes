import os
import sys
import os.path as osp

def get_imagelist_from_folder(imagepath):
    imglist = os.listdir(imagepath)
    exts = ['.png', '.jpg', '.jpeg']
    imglist = [osp.join(imagepath, imgname) for imgname in imglist if osp.splitext(imgname)[-1].lower() in exts]
    return imglist

def get_imagelist_from_txt(imagepath):
    imglist = []
    with open(imagepath, 'r') as f:
        for line in f:
            imglist.append(line.strip().split()[0])
    return imglist

def get_imagelist_from_folders(imagepath):
    imglist = []
    foldernames = os.listdir(imagepath)
    for foldername in foldernames:
        folder = osp.join(imagepath, foldername)
        if osp.isdir(folder):
            imglist.extend(get_imagelist_from_folder(folder))
    return imglist

def get_imagelist(imagepath, format):
    if format == 'dir':
        imglist = get_imagelist_from_folder(imagepath)
    elif format == 'txtfile':
        imglist = get_imagelist_from_txt(imagepath)
    elif format == 'folders':
        imglist = get_imagelist_from_folders(imagepath)

def calculate_folders(dirpath):
    for dirname in os.listdir(dirpath):
        print('{}: '.format(dirname), len(os.listdir(osp.join(dirpath, dirname))))

def makeDir(dirpath):
    if not osp.exists(dirpath):
        os.makedirs(dirpath)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('wrong!')
