from re import sub
import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pdb
import math
import json

'''
逻辑很简单：
不断循环，数据越分越少，每次循环就管两个，划分值以及维度

    数据循环维度，算维度下不同划分值，可得的熵，选出最小的熵
            各维度再用各自最好的熵算信息增益，最大的信息增益对应的维度的最小的熵对应的划分值
    
    某批数据label全部一样，或者信息增益为零，那就说明没得分了，直接去这个批次label众数

测试就很简单，if else嘛
各个节点有对应的维度，大于特征值进入右子树，小于就是左子树
不断往深处走，直到某一层不需要分了，直接给你label
'''


def get_dataset():
    iris=datasets.load_iris()

    return iris

def split_data(original_dataset):
    #数据划分，也有库，我再给你们写一个
    #>>>>>>>>>1\调库,8:2划分<<<<<<<<<<<
    # x_train, x_test, y_train, y_test = train_test_split(original_dataset.data, original_dataset.target, test_size=0.2, random_state=22)
    
    # >>>>>>>>>2\自己写<<<<<<<<<<<<<<
    # 先data和label拼起来再打乱后随机取前八成当训练
    
    datas=original_dataset.data
    labels=np.expand_dims(original_dataset.target,axis=-1)
    dataset=np.concatenate((datas,labels), axis=-1)
    #
    np.random.shuffle(dataset)
    np.random.shuffle(dataset)

    data_train,data_test=dataset[0:math.floor(dataset.shape[0]*0.8)],dataset[math.floor(dataset.shape[0]*0.8):]
    x_train,x_test=data_train[:, :4], data_test[:, :4]
    y_train,y_test=data_train[:, 4], data_test[:, 4]
    #可对照库，殊途同归啊
    return x_train,x_test,y_train,y_test

def calEnt(labels):
    '''信息熵计算, based on label_prob
    '''
    n = len(labels)                           
    label2num=Counter(labels)          
    prob_lst=[num/n for _,num in label2num.items()] 
    return -(prob_lst*np.log(prob_lst)).sum()

def majorityCnt(cls_lst):
    '''统计出现最多次的类别，求众数
    '''
    label2num=Counter(cls_lst)
    sort_label2num=sorted(label2num.items(), key=lambda x:x[-1])
    return sort_label2num[0][0]

def splitDataSet_c(dataSet,feat_dim,value): 
    '''dataset:
    feat_dim: which feat_dim
    value: mean(feat_m, feat_(m+1) )
    flag:values < or > value

    这一步就是根据某个维度的某个选择的划分值将数据切两份
    '''
    l_choose=dataSet[:, feat_dim]<value
    r_choose=dataSet[:, feat_dim]>value
        
    return l_choose,r_choose

def chooseBestDim(dataset,labels):
    '''选择最优特征维度

    说白了两层循环，确定最好的维度，以及最好的维度该用哪个划分值
    里层是，每个维度下哪个划分值，熵越小就选哪个，熵越大不确定性越强嘛
    外层是，哪个维度信息增益越大就选哪个，熵-条件熵，信息增益越大分类特征越强
    '''
    #信息熵
    base_entropy=calEnt(labels)

    ##最优特征维度，最优信息增益, 最佳划分值，说白了针对连续特征，选个值分成两类
    best_dim, best_infoGain, best_partV=-1, 0, None


    for feat_dim in range(dataset.shape[1]):
        #各维度特征值，这一步相当于算维度筛选了
        feat_in_dim=dataset[:, feat_dim]
        feat_in_dim=set(feat_in_dim)

        #仅针对连续数据
        sort_feat_in_dim=list(feat_in_dim)
        sort_feat_in_dim.sort()
        minEntropy=float('inf') #赋个无限大的初值


        newEntropy=0.0
        best_partV_dim=None

        for j in range(len(sort_feat_in_dim)-1):
            #循环当前维度下特征值，看看怎么切好弄
            part_v=(float(sort_feat_in_dim[j])+float(sort_feat_in_dim[j+1]))/2
            l_choose, r_choose=splitDataSet_c(dataset,feat_dim, part_v)
            dataset_l, dataset_r=dataset[l_choose], dataset[r_choose]
            prob_l, prob_r=len(dataset_l)/len(dataset), len(dataset_r)/len(dataset)

            entropy=prob_l*calEnt(labels[l_choose])+prob_r*calEnt(labels[r_choose])

            if entropy<minEntropy:
                #越小越好
                best_partV_dim=part_v
                minEntropy=entropy

        newEntropy=minEntropy
        infoGain=base_entropy-newEntropy #信息增益

        if infoGain>best_infoGain:
            #越大越好
            best_infoGain=infoGain
            best_dim=feat_dim
            best_partV=best_partV_dim
    
    return best_dim, best_partV


def create_tree(dataset, labels, feat_idxs):
    '''dataset with label
    labels: labels
    feat_idxs 维度集合


    逻辑其实挺简单的，反正就是递归去把数据分批，
    截至条件三个：1、这个批次全是一个label了，那走到这层了；
    2、信息增益为零，或者特征筛的只剩一维了那就停，
    但我看这个代码逻辑划分的子数据集没有剔除用过的维度，那我肯定也懒得弄
    所以其实截至条件就两个、不需要分了，
    '''
    #这个dataset是在逐渐衰减的
    if len(set(labels))==1:
        #只剩一个类了，那还分个什么
        return float(labels[0])

    if dataset.shape[1]==1:
        #相当于只剩一维特征了，没得筛了，那哪个label占比最多就哪个呗，其实没用到，
        #你们可以自己想想，每次划分后踢掉某个维度说不定也有好处，有点矛盾，因为他每个维度每次只划分两类，
        #不好说
        return float(majorityCnt(labels))
    
    best_dim, best_partV=chooseBestDim(dataset, labels)

    if best_dim==-1:
        return majorityCnt(labels)
    
    #什么意思每一层记录这一层用了什么维度，以及这个维度的最佳划分值
    best_feat_dim=str(best_dim)+'_'+str(best_partV)
    myTree={best_feat_dim:{}}
    #左右子树
    l_choose, r_choose=splitDataSet_c(dataset, best_dim, best_partV)
    dataset_l, dataset_r=dataset[l_choose], dataset[r_choose]
    labels_l, labels_r=labels[l_choose], labels[r_choose]
    myTree[best_feat_dim]['Left'] = create_tree(dataset_l, labels_l,feat_idxs)
    myTree[best_feat_dim]['Right'] = create_tree(dataset_r, labels_r, feat_idxs)

    return myTree

def classify(inputTree, feat_dixs, sample):
    firstStr = list(inputTree.keys())[0]  # 根节点

    #当前层选择的维度以及划分值
    feat_dim,best_v = str(firstStr).split('_')

    sub_dct = inputTree[firstStr]
    '''
    对每个分支循环,其实就两个啊，我看的这个代码可能比较low，他反正你离散数据我都是取个值分两类就完了
    但仔细想想又不是，因为他划分好数据集后，子数据集没有剔除用过的维度，不排除某个维度被分成四分或者八份等等的可能性
    '''
    for _ in sub_dct.keys():  
        partValue = float(best_v)
        if sample[int(feat_dim)] < partValue:  # 进入左子树
            if isinstance(sub_dct['Left'], dict):
                classLabel = classify(sub_dct['Left'], feat_dixs, sample)
            else:
                classLabel = sub_dct['Left']
        else:
            if isinstance(sub_dct['Right'], dict):
                classLabel = classify(sub_dct['Right'], feat_dixs, sample)
            else:
                classLabel = sub_dct['Right']

    return classLabel

def get_acc(pred,gt):
    n=len(pred)

    tp=0
    for i in range(n):
        tp+=1 if pred[i]==gt[i] else 0
    
    return float(tp)/n


def main():
    '''他离散不离散还分开处理，我们数据集鸢尾花啊，我也不跟你烦，我直接按连续来弄
    '''
    origin_data=get_dataset()
    x_train,x_test,y_train,y_test=split_data(origin_data)

    #相当于把有几维度拎出来，方便作优先级选择
    feat_idxs=list(range(len(x_train[0])))

    myTree=create_tree(x_train, y_train, feat_idxs)

    #这个树本质就是个字典啊，我把它保存下来，看看啥玩意方便理解
    #菜鸟教程、python下面json、自己看去
    with open(r'./tree.json', 'w')as f:
        f.write(json.dumps(myTree))


    pred_ans=[None]*len(y_test)
    for i, sample in enumerate(x_test):
        sub_pred=classify(myTree, feat_idxs,sample)
        pred_ans[i]=sub_pred

    print(pred_ans)
    print(y_test)
    print(get_acc(pred_ans, y_test))



main()






