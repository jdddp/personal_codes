import os
import os.path as osp

def get_img_list(datasetPath):
    if osp.isdir(datasetPath):
        return [osp.join(datasetPath, imgname) for imgname in os.listdir(datasetPath) if osp.splitext(imgname)[-1].lower() in ['.jpg', '.jpeg', '.png']]
    elif osp.isfile(datasetPath):
        if osp.splitext(datasetPath)[-1].lower() in ['.jpg', '.jpeg', '.png']:
            return [datasetPath]
    else:
        return []