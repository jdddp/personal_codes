import os, sys
import os.path as osp
import argparse
import numpy as np
from collections import Counter
import json

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


def cluster(featurePath, txtPath, jsonPath, clusterNum=30):
    clusterNum=int(clusterNum)
    feature=np.load(featurePath)
    kmeans=KMeans(n_clusters=clusterNum).fit(feature)
    all_labels=list(Counter(kmeans.labels_).keys())
    # print(dir(kmeans))
    ansDct={}

    for centri_label in all_labels:
        ansDct[str(centri_label)] = {
            'centroid_feature':kmeans.cluster_centers_[centri_label].tolist(),
            'image_detail':[]
        }
        
    with open(txtPath)as f:
        for idx,line in enumerate(f):
            imgPath = line.strip()
            ansDct[str(kmeans.labels_[idx])]['image_detail'].append({
                'imgpath':imgPath,
                'centroid_distance': euclidean_distances(feature[idx].reshape(1,-1), kmeans.cluster_centers_[kmeans.labels_[idx]].reshape(1,-1)).item(0)
            })

    for centri_cate in ansDct:
        ansDct[centri_cate]['image_detail'].sort(key=lambda x:x['centroid_distance'])
        
    with open(jsonPath, 'w', encoding='utf-8')as f:
        f.write(json.dumps(ansDct))

    return ansDct

if __name__=='__main__':
    if len(sys.argv)>1:
        func=getattr(sys.modules[__name__], sys.argv[1])
        func(*(sys.argv[2:]))
    else:
        print('wrong!')

#python path/to/k_means.py cluster *args

