#-*- coding: utf-8 -*-
# from __future__ import print_function
import pandas as pd
import kmeans_numpy
import importlib as imp
imp.reload(kmeans_numpy)
from kmeans_numpy import *

'''二维数据聚类过程演示'''
print(u'聚类过程展示...\n')
path = "K-means/"
X = pd.read_csv(path+"data.csv").values
K = 3   # 总类数
initial_centroids = np.array([[3,3],[6,2],[8,5]])   # 初始化类中心
max_iters = 10
mat = kmeans(X,K,initial_centroids,max_iters,True)  # 调用 kmeans 模块
centroids, idx = mat.runKMeans()                    # 执行K-Means聚类算法
















# 初始化类中心--随机取K个点作为聚类中心
# def kMeansInitCentroids(X,K):
#     m = X.shape[0]
#     m_arr = np.arange(0,m)      # 生成0-m-1
#     centroids = np.zeros((K,X.shape[1]))
#     np.random.shuffle(m_arr)    # 打乱m_arr顺序    
#     rand_indices = m_arr[:K]    # 取前K个
#     centroids = X[rand_indices,:]
#     return centroids
