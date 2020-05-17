# -*- coding: utf-8 -*-
from numpy import *
import operator
from os import listdir

#分类器
#参数：inX（用于分类）dataSet（训练样本集）
#参数：labels（标签向量）k（选择最近邻居的数目）
#返回值：inX的标签    
def classify0(inX, dataSet, labels, k):
    #矩阵的行数，在这里即是训练样本集数据的个数
    dataSetSize = dataSet.shape[0]
    #构建一个将inX按行复制dataSetSize次的矩阵
    #并减去训练样本集构成的矩阵
    diffMat = tile(inX, (dataSetSize,1))-dataSet
    sqDiffMat = diffMat**2
    #矩阵每一行相加
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    #从小到大排序，返回下标
    sortedDistIndicies = distances.argsort()
    #初始化字典
    classCount = {}
    #取最近的k个数据点并计数
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    #排序，从大到小
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#将图像转化成向量
#即将32*32的图像矩阵为1*1024的向量
#参数：文件名
#返回值：1*1024向量
def img2vector(filename):
    #初始化返回向量
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

#手写数字识别系统
def handwritingClassTest():
    hwLabels = []
    #列出测试集目录里的文件名
    trainingFileList = listdir('trainingDigits')
    #文件名个数
    m = len(trainingFileList)
    #初始化m*1024矩阵
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        #形如0_0.txt得到0_0
        fileStr = fileNameStr.split('.')[0]
        #得到真实数值
        classNumStr = int(fileStr.split('_')[0])
        #拓展真实值数组
        hwLabels.append(classNumStr)
        #利用测试集构造矩阵
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        #调用img2vector函数将32*32图像矩阵转化为1*1024向量
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        #调用classify0函数获取标签
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("正确结果：%d 预测结果：%d" % (classNumStr, classifierResult))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\n预测错误数目: %d" % errorCount)
    print("\n错误率: %f" % (errorCount/float(mTest)))
    
if __name__ == '__main__':
    handwritingClassTest()
    