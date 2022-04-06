import numpy as np
import json
import sys

 
def iou(box, clusters):
    """每个gt和k个先验框的iou
    box: w,h
    clusters: (k,2)
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou_ = intersection / (box_area + cluster_area - intersection)
    return iou_


def avg_iou(boxes, clusters):
    """
    计算all_gt和k个Anchor的iou均值。
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])

def kmeans(boxes, k=9, dist=np.median):
    """
    利用IOU值进行K-means聚类
    参数boxes: 形状为(r, 2)的ground truth框，其中r是ground truth的个数
    参数k: Anchor的个数
    参数dist: 距离函数
    返回值：(k, 2),k个Anchor框
    """
    i=0
    rows = boxes.shape[0]
    
    distances = np.empty((rows, k))

    last_clusters = np.zeros((rows,))

    np.random.seed()

    #随机挑选聚类中心
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while i<1000:
        # 各个聚类点和中心距离
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)
        # 每个gt记录下他属于的anchor_index
        nearest_clusters = np.argmin(distances, axis=1)

        # 结束条件；可同时加一个次数约束条件
        if (last_clusters == nearest_clusters).all():
            break
        # 更新簇中心
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        # 更新Anchor_index
        last_clusters = nearest_clusters

        i+=1

    return clusters

def clusterBoxes(box_list, k=9):
    box_list=np.array(box_list)
    result=kmeans(box_list, k)
    print("Accuracy: {:.2f}%".format(avg_iou(box_list, result) * 100))   
    # print(sorted(result.tolist(), key=lambda x:x[1]))
    print(sorted(result.tolist()))



if __name__ == '__main__':
    if len(sys.argv)>0:
        func=getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('wrong')
