'''先验概率得后验概率;
每个特征维度视作一个属性，计算各个属性的条件概率;
这个算法按我理解局限性太大了啊，他等于说要每个特征值在训练集也有，它是根据特定值得概率，
就是说我所有特征第一维度训练集出现了五个值，那你测试集出现这五个值以外的值直接没用了，
当然这是最基本的理解，就是假定特征是离散的，如果假设他是符合高斯分布那可能就是另一种说法
'''
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pdb
import math
from collections import Counter,defaultdict

def get_dataset():
    '''这边直接调了sklearn里面一个数据集，先了解一下数据结构再搞别的
    '''
    iris=datasets.load_iris()
    # print(type(iris))
    # print(dir(iris))

    print('数据集中共有哪几个特征：',iris.feature_names)
    print('数据集维度：',iris.data.shape)
    print('数据集共几类，三类可见：',np.unique(iris.target))
    print('数据集共几类，三类可见：',iris.target)
    print('数据集各id对应label_name：',iris.target_names)  
    #ok,到这结束，这是一个三分类任务，150条数据，每条数据4个维度
    #也可以自己造啊，无非就是一个150*4的数据，再来150个label，不想造了，瑞了

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

def get_prior_prob(data_labels):
    #计算各个大类分类概率,也就是所谓先验概率
    label2prob=dict()

    num_all=len(data_labels)
    #python 字典自己看一下，很有用
    label2num=Counter(data_labels)
    for label, num in label2num.items():
        label2prob[label]=num/num_all
    
    return label2prob

def get_condition_prob(data, labels, label2prob):
    '''data:训练数据
    labels:与data顺序相对应的label

    #条件概率,也叫似然函数？
    #计算各个大类下，各维度特征值出现的概率
    '''
    label2feat2prob=defaultdict(lambda:defaultdict(dict))
    #不知道对不对啊，我计算各label下各个维度下各个label的prob(ability)
    for label,_ in label2prob.items():
        #循环各个label
        target_data=data[labels==label]
        # pdb.set_trace()
        #分特征维度去统计，相当于复用各个列（一条数据）中各个数的概率
        for v_feat in range(target_data.shape[1]):
            row_feat=target_data[:, v_feat].flatten()
            label2feat2prob[label][v_feat]=get_prior_prob(row_feat)
            # pdb.set_trace()
    
    return label2feat2prob

def infer(sample, prior_prob, label2feat2prob,eps=1e-9):
    '''sample:单个测试样本
    prior_prob:各个类别的概率
    label2feat2prob:各类别下各列各特征维度中各个值概率
    '''
    #存放各个label的得分以便取极大值输出label
    label_scores=np.zeros((len(prior_prob.items()),))
    for i,cate in enumerate(prior_prob.keys()):
        #遍历label
        row2feat2label=label2feat2prob[cate]
        sub_prob=prior_prob[cate]
        for row, feat in enumerate(sample):
            sub_row_prob=row2feat2label[row].get(feat, eps)
            sub_prob*=sub_row_prob
        label_scores[i]=sub_prob
    return label_scores.argmax()

def sim_bayes_cls(origin_data):
    x_train,x_test,y_train,y_test=split_data(origin_data)
    #计算各个类先验
    prior_label2probs=get_prior_prob(y_train)
    condition_prob=get_condition_prob(x_train, y_train, prior_label2probs)
    # print(np.concatenate((x_train,np.expand_dims(y_train,axis=-1)), axis=1))
    #有了先验开始算测试集预测答案，
    # 防止出现意外值，我们给没出现过的特征值的概率赋一个极小值
    predict_results=np.zeros_like(y_test)
    for i,test_data in enumerate(x_test):
        label_id=infer(test_data, prior_label2probs, condition_prob)
        predict_results[i]=label_id
    print('>>>>>>>>>>>>>>>see see result<<<<<<<<<<<<<<<<<<<<')
    print('>>>>>>>>>>>>>>>see see result<<<<<<<<<<<<<<<<<<<<')
    print('>>>>>>>>>>>>>>>see see result<<<<<<<<<<<<<<<<<<<<')

    print('预测结果', predict_results)
    print('实际结果', y_test)
    #这边可以加个acc的计算，我懒得搞了
    #然后，用官方库切出来的数据集鲁棒性好像更好，自己切结果时好时坏，不管了




def main(dataset_path):
    #我们这边直接调的库，就没有这一说啦
    #original_data=np.load(dataset_path)

    original_data=get_dataset()

    sim_bayes_cls(original_data)

main(r'asd')