import numpy as np
import pandas as pd


def addBias(X):
    return np.insert(X, 0, [1] * len(X), axis=1)


def StartPerceptron(featureColumns, labelColumn, epochs: int = 10, learnRate: float = 0.1):
    featureColumns = addBias(featureColumns)
    weightMatrix = np.zeros(len(featureColumns[0]))
    for epoch in range(epochs):
        indexes = np.arange(len(featureColumns))
        for index in indexes:
            if labelColumn[index] * np.dot(weightMatrix, featureColumns[index]) <= 0:
                weightMatrix += learnRate * (featureColumns[index] * labelColumn[index])
        if epoch + 1 == epochs:
            return weightMatrix


def VotedPerceptron(featureColumns, labelColumn, epochs: int = 10, learnRate: float = 0.1):
    featureColumns = addBias(featureColumns)
    corrects = 0
    unifiedWeightsArray = []
    weightMatrix = np.zeros(len(featureColumns[0]))
    for epoch in range(epochs):
        indexes = np.arange(len(featureColumns))
        np.random.shuffle(indexes)
        for index in indexes:
            if labelColumn[index] * np.dot(weightMatrix, featureColumns[index]) <= 0:
                weightMatrix += learnRate * (featureColumns[index] * labelColumn[index])
                unifiedWeightsArray.append([weightMatrix, corrects])
                corrects = 1
            else:
                corrects += 1
    return unifiedWeightsArray


def predictData(weights, labels):
    labels = addBias(labels)
    pred = lambda d: np.sign(np.dot(weights, d))
    return np.array([pred(xi) for xi in labels])


def AverageVotedPerceptron(featureColumns, labelColumn, epochs: int = 10, learnRate: float = 0.1):
    featureColumns = addBias(featureColumns)
    weightMatrix = np.zeros(len(featureColumns[0]))
    weightsFinalMatrix = np.zeros_like(featureColumns[0])

    for e in range(epochs):
        idxs = np.arange(len(featureColumns))
        np.random.shuffle(idxs)
        for i in idxs:
            if labelColumn[i] * np.dot(weightMatrix, featureColumns[i]) <= 0:
                weightsFinalMatrix += learnRate * (labelColumn[i] * featureColumns[i])
            weightsFinalMatrix = weightsFinalMatrix + weightMatrix
    return weightsFinalMatrix


def PredictVotedPerceptron(uWeights, features):
    features = addBias(features)
    predictionArray = np.ones(len(features))
    for feature in range(len(features)):
        sumOf = 0
        for weight in uWeights:
            sumOf += weight[-1] * np.sign(np.dot(weight[0], features[feature]))
        predictionArray[feature] = np.sign(sumOf)
    return predictionArray


def importData(fileLocation):
    data = pd.read_csv(fileLocation)
    data.loc[data.iloc[:, -1] == 0] = -1
    return np.array(data.iloc[:, :-1]), np.array(data.iloc[:, -1])


if __name__ == "__main__":
    xTrain, yTrain = importData("train.csv")
    xTest, yTest = importData("test.csv")
    weightMatrix = StartPerceptron(xTrain, yTrain, 10, 0.1)
    testAccuracyTrain = np.mean(yTrain == predictData(weightMatrix, xTrain))
    testAccuracy = np.mean(yTest == predictData(weightMatrix, xTest))
    print("=================================Perceptron Train Accuracy================================")
    print(testAccuracy)
    print("\n")
    print("=================================Perceptron Test Accuracy================================")
    print(testAccuracy)
    print("\n")
    votedWeights = VotedPerceptron(xTrain, yTrain, 10, 1e-3)
    votedTestAccuracyTrain = np.mean(yTrain == PredictVotedPerceptron(votedWeights, xTrain))
    votedTestAccuracy = np.mean(yTest == PredictVotedPerceptron(votedWeights, xTest))
    # print("=================================Voted Perceptron Weights================================")
    # print(votedWeights)
    # print("\n")
    print("=================================Voted Perceptron Train Accuracy================================")
    print(votedTestAccuracyTrain)
    print("\n")
    print("=================================Voted Perceptron Test Accuracy================================")
    print(votedTestAccuracy)
    print("\n")
    averageWeights = AverageVotedPerceptron(xTrain, yTrain, 10, 1e-3)
    averageAccuracyTrain = np.mean(yTrain == predictData(averageWeights, xTrain))
    averageAccuracy = np.mean(yTest == predictData(averageWeights, xTest))
    print("=================================Average Perceptron Train Accuracy================================")
    print(averageAccuracyTrain)
    print("\n")
    print("=================================Average Perceptron Test Accuracy================================")
    print(averageAccuracy)
    print("\n")
