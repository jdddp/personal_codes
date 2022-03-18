import os
import os.path as osp
import json
import sys
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report

def calcul_pre_rec(label2id_path, ansPath):
    '''ansPath: output of infer(.log)
    '''
    label2id=json.loads(open(label2id_path).read())

    a=sorted(label2id.items(), key=lambda x:x[1])
    label_list=list(zip(*a))[0]
    # print(a)
    gt_list=[]
    pred_list=[]
    with open(ansPath, encoding='utf-8')as f:
        for line in f:
            line=eval(line.strip())
            for dct in line:
                pred=int(label2id[dct['label_list'][0]])
                gt=int(osp.basename(dct['imgpath']).split('_')[0])
                gt_list.append(gt)
                pred_list.append(pred)

    ans=classification_report(gt_list, pred_list, target_names=label_list)
    print(ans)

if __name__=='__main__':
    if len(sys.argv) > 1:
        func=getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('wrong!')

        