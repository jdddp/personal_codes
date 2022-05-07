from logging import exception
import os
import os.path as osp
import shutil
import json
import argparse
from tqdm import tqdm

def copy_test_data(cocoJsonPath, usualJsonPath,imgDir, goalDir,ansJsonDir):
    '''测试图片，json转为通用格式，方便对比；copy一份方便可视化
    '''
    os.makedirs(goalDir,exist_ok=True)

    ansDct={}
    res=json.loads(open(cocoJsonPath).read())
    usualJson=json.loads(open(usualJsonPath).read())
    for sgDct in tqdm(res['images']):
        try:
            shutil.copy(osp.join(imgDir,sgDct['file_name']),goalDir)
            ansDct[sgDct['file_name']]=usualJson[sgDct['file_name']]
        except Exception as e:
            print(e)
    
    with open(osp.join(ansJsonDir, 'test_usual.json'),'w', encoding='utf-8')as f:
        f.write(json.dumps(ansDct))
    return ansDct

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='process args')
    parser.add_argument('-c','--cocoJson', required=True, help="cocoTestJson path")
    parser.add_argument('-u','--usualJson', required=True, help="usualJson path")
    parser.add_argument('-i','--imgDir', required=True, help="path of dataset")
    parser.add_argument('-g','--goalDir', required=True, help="path of test_dataset")
    parser.add_argument('-d','--ansJsonDir', required=True, help="path of test_data_usualJson")
    args=parser.parse_args()

    copy_test_data(args.cocoJson, args.usualJson, args.imgDir, args.goalDir, args.ansJsonDir)








