'''
pip install imagededup; recommend to create a new environment for this;
'''
import os,sys
import os.path as osp
import shutil
from tqdm import tqdm
from imagededup.methods import PHash
from imagededup.utils import plot_duplicates


def duplicate_dir(img_dir):
    phasher = PHash()
    encodings=phasher.encode_images(image_dir=img_dir)
    duplicates = phasher.find_duplicates_to_remove(encoding_map=encodings)

    print('[{}] >> {} in total, {} need to be deleted'.format(osp.basename(img_dir),len(os.listdir(img_dir)), len(duplicates)))
    for i in tqdm(range(len(duplicates))):
        os.remove(osp.join(img_dir, duplicates[i]))

def duplicate_dirs(dir_path):
    for dirname in os.listdir(dir_path):
        sub_dirpath=osp.join(dir_path, dirname)
        if osp.isdir(sub_dirpath):
            duplicate_dir(sub_dirpath)
        else:
            continue

if __name__ == '__main__':
    if len(sys.argv) > 1:
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('wrong!')
        


