# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:21:49 2019

@author: Lenovo
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_all = np.loadtxt('HTRU_2_train.csv',delimiter=",")
data = data_all[:2000,:]


def sigmoid(z):
    return 1/(1+np.exp(-z))

#读取处理数据
def init_data():
    
    
    #取前两类和其前两个特征
    dataMatin = data[:,[0,1]]
    classLabels = data[:,[2]]
    
    #手动添加第三个特征，并将值置为1，为计算参数
    dataMatin = np.insert(dataMatin,0,1,axis=1)
    
    return dataMatin,classLabels

#梯度下降
def grad_descent(dataMatin,classLabels):
    #将类别与特征值变成矩阵
    dataMatrix = np.mat(dataMatin)
    labelMat = np.mat(classLabels)#.transpose()#转置
    m,n = np.shape(dataMatrix)#大小
    
    #随意生成一个3维向量
    weights = np.ones((n,1))
    alpha = 0.001#学习率
    maxCycle = 460#指定次数
    
    for i in range(maxCycle):
        #逻辑回归的预测函数
        h = sigmoid(dataMatrix*weights)
        
        #根据梯度下降原理，要求代价函数的最小值
        #根据矩阵相乘求得m个样本的偏导数之和
        weights = weights - alpha*dataMatrix.transpose()*(h-labelMat)
    
    return weights#最终确定的参数

#可视化
def look(weights):
    
    a_x = data[data[:,2] == 0,0]
    a_y = data[data[:,2] == 0,1]
    b_x = data[data[:,2] == 1,0]
    b_y = data[data[:,2] == 1,1]
    
    #可视化逻辑回归
    x = np.arange(0,175,1)
    y = - weights[1,0]/weights[2,0]*x - weights[0,0]/weights[2,0]
    
    plt.figure(1,figsize=(14,8))
    plt.scatter(a_x,a_y,s=5)
    plt.scatter(b_x,b_y,s=5)
    plt.plot(x,y)
    plt.show()
    
def test(weights):
    data_test = np.loadtxt('HTRU_2_test.csv',delimiter=",")
    #data = data_all
    #data_test = data_all[2000:,:]
    m,n = data_test.shape
    answer = weights[0,0] + weights[1,0]*data_test[:,0] + weights[2,0]*data_test[:,1]
    '''
    acc_count = 0
    for i in range(0,m):
        if(answer[i] >= 0):
            answer[i] = 1
        else:
            answer[i] = 0
        if(answer[i] == data_test[i,[2]]):
            acc_count = acc_count + 1
    acc = acc_count/m*100
    print(acc)
    '''
    for i in range(0,m):
        if(answer[i] >= 0):
            answer[i] = 1
        else:
            answer[i] = 0
    #print(answer)
    number = np.array(range(1,701))
    out = np.vstack((number, answer))
    out = out.T
    out = np.array(out,dtype=int)
    #print(out)
    out_title = np.array(['id','y'])
    
    import csv
    with open('out.csv','w',newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(out_title)
        for row in out:
            writer.writerow(row)
    csv_file.close()

dataMatin,classLabels = init_data()
weights = grad_descent(dataMatin,classLabels)
look(weights)
test(weights)

















