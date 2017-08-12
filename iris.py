from numpy import *
import operator


def main():
    features, labels = loadDataSet()

    testFeatures = zeros((30, 4))
    testLabels = zeros(30)
    trainFeatures = zeros((120, 4))
    trainLabels = zeros(120)

    # 从每一类中抽取一部分做测试集,剩余做训练集
    for i in range(3):
        testFeatures[i * 10: i * 10 + 10] = features[i * 50: i * 50 + 10]
        testLabels[i * 10: i * 10 + 10] = labels[i * 50: i * 50 + 10]
        trainFeatures[i * 10: i * 10 + 40] = features[i * 50: i * 50 + 40]
        trainLabels[i * 10: i * 10 + 40] = labels[i * 50: i * 50 + 40]

    errorCount = 0
    for i in range(30):
        classifierResult = classifierKNN(testFeatures[i], trainFeatures, trainLabels, 5)
        print("the classifier came back with: %d, the real answer is : %d" % (classifierResult, testLabels[i]))
        if (classifierResult != testLabels[i]):
            errorCount += 1

    print("the total error rate is: %f" % (errorCount / float(30)))


def loadDataSet():
    iris = loadtxt(open("./Iris.csv", "rb"), delimiter=",", skiprows=1)

    features = iris[:, 1:5]
    labels = iris[:, 5].astype(int)

    return features, labels


def classifierKNN(inX, dataSet, labels, k):
    # 距离计算
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance**0.5
    # argsort()函数的作用是升序排列，然后返回原位置的下标
    sortedDistIndicies = distance.argsort()

    # 选择距离最小的k个点
    classCount = {}

    for i in range(k):
        # 取相应的标签
        voteIlabel = labels[sortedDistIndicies[i]]

        # dict.get()在dict中查找键对应的值，没有返回0(默认是none）
        # 通过加一来实现投票算法进行统计
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 排序 sorted()对iterable对象做排序，key指定排序对象，reverse=True代表降序，另有一个cmp参数，指定排序方法
    # operator.itemgetter()函数定义了一个函数，表示取值
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]
