import os,sys
import os.path as osp
import cv2
import json
import shutil
import collections
import json

def searchInsert(nums, target):
    '''二分排序寻找插入位置
    '''
    flag=0
    left = 0 

    right = len(nums) - 1
    while left <= right:
        middle = (left + right) // 2
        if nums[middle] < target:
            left = middle + 1
        elif nums[middle] > target:
            right = middle - 1
        else:
            flag=1
            return middle+1, flag
    return right+1,flag

# nums=[0,1,3,4,5]
# target=1.5
# a=searchInsert(nums, target)
# print(target,a)


def slideCrop(imgPath,goal_dir, dct_lst, scale=640, thre=0.75):
    '''目前针对单一类别;本次判断跨图边框是否有效看他面积占比是否超过0.75
    scale: size of img
    thre:threshold for bbox in over 1 img
    '''
    ansDct=collections.defaultdict(list)

    os.makedirs(goal_dir,exist_ok=True)
    imgname=osp.basename(imgPath)
    img=cv2.imread(imgPath)
    h,w,_=img.shape

    #calcuNum in row/col
    w_scale=scale
    h_scale=scale
    row_n=w//w_scale if w%w_scale==0 else w//w_scale+1
    col_n=h//h_scale if h%h_scale==0 else h//h_scale+1

    #暂存宽高起点
    w_s,h_s=w//row_n, h//col_n
    w_lst=[]
    h_lst=[]
    for i in range(w_s):
        w_lst.append(i*w_s)
    for i in range(h_s):
        h_lst.append(i*h_s)
    
    #搞定图片先[[1,2,3],[4,5,6]]
    #row_n is num, w_s is 宽高起点
    for i in range(row_n):
        for j in range(col_n):
            img_id=str(j*row_n+i+1)
            #边缘越界判断
            h_need=min(j*h_s+h_scale, h)
            w_need=min(i*w_s+w_scale, w)

            img_temp=img[j*h_s:h_need, i*w_s:w_need,:]
            cv2.imwrite(osp.join(goal_dir, img_id+'-'+imgname), img_temp)
    
    #w_lst,h_lst判断框从属
    for sgDct in dct_lst:
        x1_b,y1_b,w_b,h_b=sgDct['bbox']
        s_b=w_b*h_b
        x2_b,y2_b=x1_b+w_b,y1_b+h_b
        #确定x1,y1,x2,y2分别属于哪个区间
        #flag=0,位于输出位置左侧；flag=1，位于输出位置
        (x1_pos,flag_x1),(x2_pos,flag_x2)=searchInsert(w_lst,x1_b), searchInsert(w_lst,x2_b)
        (y1_pos, flag_y1),(y2_pos, flag_y2)=searchInsert(h_lst,y1_b), searchInsert(h_lst,y2_b)
        #四种:刚好在某一块；横跨了；上下取舍；左右取舍
        if (x1_pos<x2_pos and flag_x2==1) or (x1_pos==x2_pos):
            #左右无需取舍
            if (y1_pos<y2_pos and flag_y2==1) or (y1_pos==y2_pos):
                #上下无需取舍
                x1_id=x1_pos 
                y1_id=y1_pos 
                img_id=(y1_id-1)*col_n+x1_id

                ansDct[str(img_id)+'-'+imgname].append({
                    "category": sgDct['category'],
                    "bbox": [x1_b-(x1_id-1)*w_scale, y1_b-(y1_id-1)*h_scale, w_b, h_b]
                })
            else:
                #y1_pos<y2_pos
                h_up,h_down=(y1_pos*h_scale)-y1_b, y2_b-(y1_pos*h_scale)
                if h_up/h_b>=thre:
                    #在上
                    x1_id=x1_pos
                    y1_id=y1_pos
                    img_id=(y1_id-1)*col_n+x1_id
                    ansDct[str(img_id)+'-'+imgname].append({
                    "category": sgDct['category'],
                    "bbox": [x1_b-(x1_id-1)*w_scale, y1_b-(y1_id-1)*h_scale, w_b, h_up]
                })
                else:
                    x1_id=x1_pos
                    y1_id=y2_pos
                    img_id=(y1_id-1)*col_n+x1_id
                    ansDct[str(img_id)+'-'+imgname].append({
                    "category": sgDct['category'],
                    "bbox": [x1_b-(x1_id-1)*w_scale, (y1_id-1)*h_scale, w_b, h_down]
                })
        #上下无需取舍,左右需取舍;x1_pos<x2_pos
        elif ((y1_pos<y2_pos and flag_y2==1) or (y1_pos==y2_pos)) \
            and (x1_pos!=x2_pos):
            y1_id=y1_pos
            w_left,w_right=(x1_pos*w_scale)-x1_b, x2_b-(x1_pos*w_scale)
            if w_left/w_b>=thre:
                #在左
                x1_id=x1_pos
                img_id=(y1_id-1)*col_n+x1_id
                ansDct[str(img_id)+'-'+imgname].append({
                "category": sgDct['category'],
                "bbox": [x1_b-(x1_id-1)*w_scale, y1_b-(y1_id-1)*h_scale, w_left, h_b]
            })
            if w_right/w_b>=thre:
                #在右
                x1_id=x2_pos
                img_id=(y1_id-1)*col_n+x1_id
                ansDct[str(img_id)+'-'+imgname].append({
                "category": sgDct['category'],
                "bbox": [(x1_id-1)*w_scale, y1_b-(y1_id-1)*h_scale, w_right, h_b]
            })
        #上下左右都需取舍
        else:
            #s_b
            s_thre=s_b*thre
            w_l,w_r=x1_pos*w_scale-x1_b, x2_b-x1_pos*w_scale
            h_t,h_d=y1_pos*h_scale-y1_b, y2_b-y1_pos*h_scale
            s_tl, s_tr, s_dl, s_dr=w_l*h_t, w_r*h_t, w_l*h_d, w_r*h_d
            if s_tl==max(s_tl, s_tr, s_dl, s_dr) and s_tl>=s_thre:
                #left top
                x1_id, y1_id=x1_pos, y1_pos
                img_id=(y1_id-1)*col_n+x1_id
                ansDct[str(img_id)+'-'+imgname].append({
                "category": sgDct['category'],
                "bbox": [x1_b-(x1_id-1)*w_scale, y1_b-(y1_id-1)*h_scale, w_l, h_t]
            })
            if s_dl==max(s_tl, s_tr, s_dl, s_dr) and s_tl>=s_thre:
                #left down
                x1_id, y1_id=x1_pos, y2_pos
                img_id=(y1_id-1)*col_n+x1_id
                ansDct[str(img_id)+'-'+imgname].append({
                "category": sgDct['category'],
                "bbox": [x1_b-(x1_id-1)*w_scale, (y1_id-1)*h_scale, w_l, h_d]
            })
            if s_tr==max(s_tl, s_tr, s_dl, s_dr) and s_tl>=s_thre:
                #right up
                x1_id, y1_id=x2_pos, y1_pos
                img_id=(y1_id-1)*col_n+x1_id
                ansDct[str(img_id)+'-'+imgname].append({
                "category": sgDct['category'],
                "bbox": [(x1_id-1)*w_scale, y1_b-(y1_id-1)*h_scale, w_r, h_t]
            })
            if s_dr==max(s_tl, s_tr, s_dl, s_dr) and s_tl>=s_thre:
                #right dwon
                x1_id, y1_id=x2_pos, y2_pos
                img_id=(y1_id-1)*col_n+x1_id
                ansDct[str(img_id)+'-'+imgname].append({
                "category": sgDct['category'],
                "bbox": [(x1_id-1)*w_scale, (y1_id-1)*h_scale, w_r, h_d]
            })
    return ansDct

def slideCropDir(img_dir, goal_dir, jsonPath, scale=640, thre=0.75):
    '''img_dir: origin path of imgs
    goal_dir:
        imgs(split)
        usual.json(split)
    jsonPath: usual.json(original)
    '''
    scale=int(scale)
    thre=float(thre)

    goal_imgDir=osp.join(goal_dir, 'imgs')
    gen_jsonPath=osp.join(goal_dir, 'usual.json')
    os.makedirs(goal_imgDir, exist_ok=True)

    res_split={}

    res=json.loads(open(jsonPath).read())
    for imgname, dct_lst in res.items():
        img_path=osp.join(img_dir, imgname)
        res_split.update(slideCrop(img_path, goal_imgDir,dct_lst,scale,thre))
    
    for imgname in os.listdir(goal_imgDir):
        if imgname not in res_split:
            res_split[imgname]=[]
    with open(gen_jsonPath, 'w', encoding='utf-8')as f:
        f.write(json.dumps(res_split))

if __name__=='__main__':
    if len(sys.argv)>1:
        func=getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('input wrong')

            









                
            
                
                








    






