#我是按我理解的最简单的逻辑来的，应该没什么问题
import numpy as np

def get_dist(single_sample, train_data):
    #我直接均方差来计算距离，距离函数有很多
    return np.sqrt(np.sum((single_sample-train_data)**2, axis=1))

def k_dist(dataset, samples, k=10):
    '''dataset:多条有label的数据[m,n+1]，多出来的一维是因为他有label
    samples:要infer的数据[mm,n]
    k:取最近邻的个数
    '''
    train_data=dataset[:, :-1]
    train_labels=dataset[:, -1]

    #求出数目维度
    m,n=train_data.shape
    m_te, n_te=samples.shape

    assert n==n_te, '测试数据与原始数据维度不一致'
    assert m>=k,'k值太大了'

    ans=[None]*m_te
    for i, sample in enumerate(samples):
        #求出当前数据与所有数据的距离，以此选出最近的k个
        distances=get_dist(sample, train_data)
        idx_order=distances.argsort()[:k]
        k_labels=train_labels[idx_order].astype(int)

        #ok,我们求众数，这边你们可以遍历统计，我肯定得装一下
        res=np.bincount(k_labels)
        sample_idx=k_labels[np.where(res==max(res))]
        ans[i]=sample_idx
    
    return ans


#写代码一定要debug

#做一个有label的数据集出来
a=np.concatenate((np.random.randint(0,10,(1000, 20)),np.ones((1000,1))), axis=1)
b=np.concatenate((np.random.randint(10,20,(1000, 20)),np.ones((1000,1))*2), axis=1)
c=np.concatenate((np.random.randint(20,30,(1000, 20)),np.ones((1000,1))*3), axis=1)
d=np.concatenate((np.random.randint(30,40,(1000, 20)),np.ones((1000,1))*4), axis=1)
e=np.concatenate((np.random.randint(40,50,(1000, 20)),np.ones((1000,1))*5), axis=1)
train_data=np.concatenate((a,b,c,e,d), axis=0)


#测试样本搞一个
aa=np.random.randint(10,20,(1, 20))
bb=np.random.randint(10,20,(1, 20))
cc=np.random.randint(20,30,(1, 20))
dd=np.random.randint(30,40,(1, 20))
test_data=np.concatenate((aa,bb,cc,dd), axis=0)
# pdb.set_trace()
ans=k_dist(train_data, test_data)
print(ans)



    


