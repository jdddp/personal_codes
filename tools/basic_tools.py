import os
import sys
import os.path as osp

def get_num_sgdir(dirpath):
    return len(os.listdir(dirpath))

def get_nums(dirpath, format='dirs'):
    '''get num of each subdir pr dir
    '''
    if format=='dir':
        print('[dirname]: {}, nums: {}'.format(osp.basename(dirpath), get_num_sgdir(dirpath)))
    elif format=='dirs':
        for dirname in os.listdir(dirpath):
            sub_dirpath=osp.join(dirpath, dirname)
            if osp.isdir(sub_dirpath):
                print('[dirname]: {}, nums: {}'.format(dirname, get_num_sgdir(sub_dirpath)))


if __name__ == '__main__':
    if len(sys.argv)>1 :
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('wrong!')