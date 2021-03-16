import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class kmeans:
    def __init__(self, X, K, initial_centroids, max_iters, plot_process=True):
        self.X = X                                  # 数据
        self.m, self.n = X.shape                    # 数据条数和维度
        self.K = K                                  # 类数
        self.initial_centroids = initial_centroids  # 初始化
        self.max_iters = max_iters                  # 最大迭代次数
        self.plot_process = plot_process            # 是否画图


    def findClosestCentroids(self, initial_centroids):
        dis = np.zeros((self.m,self.K))           # 存储计算每个点分别到K个类的距离
        idx = np.zeros((self.m,1))           # 要返回的每条数据属于哪个类
        '''计算每个点到每个类中心的距离'''
        for i in range(self.m):
            for j in range(self.K):
                dis[i,j] = np.dot((self.X[i,:]-initial_centroids[j,:]).reshape(1,-1),(self.X[i,:]-initial_centroids[j,:]).reshape(-1,1))
        '''返回dis每一行的最小值对应的列号，即为对应的类别
        - np.min(dis, axis=1)返回每一行的最小值
        - np.where(dis == np.min(dis, axis=1).reshape(-1,1)) 返回对应最小值的坐标
        - 注意：可能最小值对应的坐标有多个，where都会找出来，所以返回时返回前m个需要的即可（因为对于多个最小值，属于哪个类别都可以）
        '''  
        dummy,idx = np.where(dis == np.min(dis, axis=1).reshape(-1,1))
        return idx[:dis.shape[0]]  # 注意截取一下


    # 计算类中心
    def computerCentroids(self,idx):
        centroids = np.zeros((self.K,self.n))
        for i in range(self.K):
            # 索引要是一维的,axis=0为每一列，idx==i一次找出属于哪一类的，然后计算均值
            centroids[i,:] = np.mean(self.X[np.ravel(idx==i),:], axis=0).reshape(1,-1)
        return centroids


    # 聚类算法
    def runKMeans(self):
        centroids = self.initial_centroids   # 记录当前类中心
        previous_centroids = centroids       # 记录上一次类中心
        # idx = np.zeros((self.m,1))           # 每条数据属于哪个类
        # 迭代法
        plt.ion()
        for i in range(self.max_iters):      # 迭代次数
            print(u'迭代计算次数：%d'%(i+1))
            idx = self.findClosestCentroids(centroids)
            if self.plot_process:    # 如果绘制图像
                plt1 = self.plotProcessKMeans(idx,centroids,previous_centroids) # 画聚类中心的移动过程
                previous_centroids = centroids  # 重置
            centroids = self.computerCentroids(idx)  # 重新计算类中心
        plt.ioff()
        if self.plot_process: plt1.show() # 显示最终的绘制结果
        return centroids, idx        # 返回聚类中心和数据属于哪个类


    def plotProcessKMeans(self,idx,centroids,previous_centroids):
        color = ['b','g','r','c','m','y']
        plt.cla()
        plt.scatter(self.X[:,0], self.X[:,1], c=idx)     # 原数据的散点图
        plt.plot(previous_centroids[:,0],previous_centroids[:,1],'rx',markersize=10,linewidth=5.0)  # 上一次聚类中心
        plt.plot(centroids[:,0],centroids[:,1],'rx',markersize=10,linewidth=5.0)                    # 当前聚类中心
        for j in range(centroids.shape[0]): # 遍历每个类，画类中心的移动直线
            p1 = centroids[j,:]
            p2 = previous_centroids[j,:]
            plt.plot([p1[0],p2[0]],[p1[1],p2[1]],"->",c=color[j],linewidth=2.0)
        plt.pause(1.0)
        return plt
