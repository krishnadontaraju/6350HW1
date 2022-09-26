# -*- coding: utf-8 -*-
"""DT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kQpPxAzbxWcQ3byqADvMBbxY9kXtBQfC
"""

from collections import Counter

import numpy as np


def entropy(y):
    uniqueValues, uniqueCounts = np.unique(y, return_counts=True)
    ps = uniqueCounts / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

def majorityError(data,featureIndex):
    mjSum = 0
    yColumnUniques = np.unique(data[: , -1])
    indices = np.argwhere()
    for yUnique in range(yColumnUniques):
      tempIndices = np.argwhere(data[: , featureIndex,-1][: , -1] == yColumnUniques[yUnique])
      tempData = data[tempIndices: , featureIndex]

      uniqueValues, uniqueCounts = np.unique(tempData, return_counts=True)
      ps = uniqueCounts.sort()
      ps = ps / len(tempData)
      mjSum = mjSum + np.sum([p])
    return np.sum()


class Node:
    def __init__(
        self, attribute=None, splitValue=None, left=None, right=None, *, value=None
    ):
        self.attribute = attribute
        self.splitValue = splitValue
        self.left = left
        self.right = right
        self.value = value

    def leafNodeCheck(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, maxDepth=100, nAttributes=None):
        self.maxDepth = maxDepth
        self.nAttributes = nAttributes
        self.root = None

    def fit(self, X, y):
        self.nAttributes = X.shape[1] if not self.nAttributes else min(self.nAttributes, X.shape[1])
        self.root = self.buildTree(X, y)

    def predict(self, X):
        return np.array([self.treeTravel(x, self.root) for x in X])

    def buildTree(self, X, y, depth=0):
        examples, attributes = X.shape
        uniqueLabels = len(np.unique(y))

        if (
            depth >= self.maxDepth
            or uniqueLabels == 1
        ):
            leafValue = self.mostCommonLabels(y)
            return Node(value=leafValue)
        
        selectedAttribute, selectedValue = self.getBestSplit(X, y, attributes)

        leftInd, rightInd = self.splitData(X[:, selectedAttribute], selectedValue)
        left = self.buildTree(X[leftInd, :], y[leftInd], depth + 1)
        right = self.buildTree(X[rightInd, :], y[rightInd], depth + 1)
        return Node(selectedAttribute, selectedValue, left, right)

    def getBestSplit(self, X, y, attrInds):
        bestGain = -1
        splitInd, splitVal = None, None
        for attrInd in range(attrInds):
            xColumn = X[:, attrInd]
            values = np.unique(xColumn)
            for value in values:
                gain = self.informationGain(y, xColumn, value)

                if gain > bestGain:
                    bestGain = gain
                    splitInd = attrInd
                    splitVal = value

        return splitInd, splitVal

    def informationGain(self, y, xColumn, splitValue):
        parentEnt = entropy(y)
        leftBranch, rightBranch = self.splitData(xColumn, splitValue)
        if len(leftBranch) == 0 or len(rightBranch) == 0:
            return 0

        n = len(y)
        nl, nr = len(leftBranch), len(rightBranch)
        el, er = entropy(y[leftBranch]), entropy(y[rightBranch])
        childEnt = (nl / n) * el + (nr / n) * er
        ig = parentEnt - childEnt
        return ig

    def splitData(self, attributeColumn, splitValue):

        leftBranch = np.argwhere(attributeColumn <= splitValue).flatten()
        rightBranch = np.argwhere(attributeColumn > splitValue).flatten()
        return leftBranch, rightBranch

    def treeTravel(self, x, node):
        if node.leafNodeCheck():
            return node.value

        if x[node.attribute] <= node.splitValue:
            return self.treeTravel(x, node.left)
        return self.treeTravel(x, node.right)

    def mostCommonLabels(self, y):
        counter = Counter(y[:, 0])
        mostCommon = counter.most_common(1)[0][0]
        return mostCommon


if __name__ == "__main__":
    import pandas as pd

    def accuracy(trueLabels, predictedLabels):
        trueLabels = trueLabels.flatten()           
        accuracy = np.sum(trueLabels == predictedLabels) / len(trueLabels)
        return accuracy
    #for i in range(17):
    ''' Usually takes a lot of time to run in the loop. Best thing to do is to pass the desired depth'''
    data = pd.read_csv('bank_train.csv')
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values.reshape(-1,1)

    clf = DecisionTree(maxDepth=i)
    clf.fit(X, Y)

    data = pd.read_csv('bank_test.csv')
    xTest = data.iloc[:, :-1].values
    yTest = data.iloc[:, -1].values.reshape(-1,1)
    yPred = clf.predict(xTest) 


    acc = accuracy(yTest, yPred)

    print("Accuracy:", acc)