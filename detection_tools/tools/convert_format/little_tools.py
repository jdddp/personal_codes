import os,sys
import os.path as osp
import json
import cv2
import pdb

def check_anno_img(dirpath):
    '''
    dirpath:
        imgs
        usual.json
    '''
    json_path=osp.join(dirpath, 'usual.json')
    img_dir=osp.join(dirpath, 'imgs')

    res=json.loads(open(json_path).read())
    for img_name, dct_lst in res.items():
        img=cv2.imread(osp.join(img_dir, img_name))
        h_s,w_s,_=img.shape
        for sg_dct in dct_lst:
            x1,y1,w,h=sg_dct['bbox']
            x2=x1+w
            y2=y1+h
            if x1>=0 and y1>=0 and x2<=w_s and y2<=h_s :
                continue
            else:
                pdb.set_trace()
            # assert x1>=0 and y1>=0 and x2<=w_s and y2<=h_s
    return res

def copy_test_data(src_dir, dst_dir):
    '''测试图片，json转为通用格式，方便对比；copy一份方便可视化
    src_dir:(original)
        imgs
        annos(coco.json)
        usual.json
    dst_dir:(generate)
        testData
        usual_test.json
    '''
    cocoJsonPath=osp.join(src_dir, 'annos/test.json')
    ori_img_dir=osp.join(src_dir, 'imgs')
    ori_usualJsonPath=osp.join(src_dir, 'usual.json')

    test_img_dir=osp.join(dst_dir, 'testData')
    test_usualJsonPath=osp.join(dst_dir, 'usual_test.json')
    os.makedirs(test_img_dir,exist_ok=True)

    ansDct={}
    res=json.loads(open(cocoJsonPath).read())
    usualJson=json.loads(open(ori_usualJsonPath).read())
    for sgDct in tqdm(res['images']):
        try:
            shutil.copy(osp.join(ori_img_dir,sgDct['file_name']), test_img_dir)
            ansDct[sgDct['file_name']]=usualJson[sgDct['file_name']]
        except Exception as e:
            print(e)
    
    with open(test_usualJsonPath,'w', encoding='utf-8')as f:
        f.write(json.dumps(ansDct))
    return ansDct

if __name__ == '__main__':
    if len(sys.argv)>1:
        func=getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('input wrong')
