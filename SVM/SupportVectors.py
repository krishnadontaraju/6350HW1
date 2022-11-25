from math import exp
import numpy as np
import scipy.optimize
import pandas as pd


def trainPrimalSVM(X, y, learnSchedule, C, epochs=10):
    X = np.insert(X, 0, [1] * len(X), axis=1)
    weights = np.zeros_like(X[0])

    for e in range(epochs):
        learn = learnSchedule(e)
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        for i in indices:
            if y[i] * np.dot(weights, X[i]) <= 1:
                zeroBias = weights.copy()
                zeroBias[0] = 0
                weights = weights - learn * zeroBias + learn * C * y[i] * X[i]
            else:
                weights = (1 - learn) * weights
    return weights


def predictPrimalSVM(weights, X) -> np.ndarray:
    X = np.insert(X, 0, [1] * len(X), axis=1)
    pred = lambda d: np.sign(np.dot(weights, d))
    return np.array([pred(xi) for xi in X])


def gaussian(x, y, gamma):
    return exp(-(np.linalg.norm(x - y, ord=2) ** 2) / gamma)


def trainDualSVM(self, X, y, C, kernel="dot", gamma=None):
    wstar = np.ndarray
    bstar = float
    support = []

    def reducer(a, X, y):
        yMatrix = y * np.ones((len(y), len(y)))
        aMtrix = a * np.ones((len(a), len(a)))

        if kernel == 'dot':
            xvals = (X @ X.T)
        if kernel == 'gaussian':
            xvals = X ** 2 @ np.ones_like(X.T) - 2 * X @ X.T + np.ones_like(X) @ X.T ** 2
            xvals = np.exp(-(xvals / gamma))

        vals = (yMatrix * yMatrix.T) * (aMtrix * aMtrix.T) * xvals
        return 0.5 * np.sum(vals) - np.sum(a)

    constraints = [
        {
            'type': 'ineq',
            'fun': lambda a: a
        },
        {
            'type': 'ineq',
            'fun': lambda a: C - a
        },
        {
            'type': 'eq',
            'fun': lambda a: np.dot(a, y)
        }
    ]

    res = scipy.optimize.minimize(reducer, x0=np.zeros(shape=(len(X),)), args=(X, y), method='SLSQP',
                                  constraints=constraints, tol=0.01)

    wstar = np.zeros_like(X[0])
    for i in range(len(X)):
        wstar += res['x'][i] * y[i] * X[i]

    bstar = 0
    if kernel == 'dot':
        for j in range(len(X)):
            bstar += y[j] - np.dot(wstar, X[j])
    if kernel == 'gaussian':
        for j in range(len(X)):
            bstar += y[j] - gaussian(wstar, X[j], gamma)
    self.bstar /= len(X)

    THRESH = 1e-10
    for i, a in enumerate(res['x']):
        if a > THRESH:
            support.append(X[i])
    return wstar, bstar, support


def predict(wstar, bstar, X, kernel="dot", gamma=None) -> np.ndarray:
    if kernel == 'dot':
        pred = lambda d: np.sign(np.dot(wstar, d) + bstar)
    if kernel == 'gaussian':
        pred = lambda d: np.sign(gaussian(wstar, d, gamma) + bstar)
    return np.array([pred(xi) for xi in X])


def runPrimalSVM(Cs, primalOrDual):
    if primalOrDual == 'Primal':
        for C in Cs:
            print(f"Current C = {C}")
            lnot, a = 1, 1
            learnSchedule = lambda e: lnot / (1 + (lnot / a) * e)
            primalWeights = trainPrimalSVM(xTrain, yTrain, lr_schedule=learnSchedule, C=C, epochs=100)
            print(f"weights: {primalWeights[1:]}")
            print(f"bias: {primalWeights[0]}")
            print(f"train accuracy: {np.mean(yTrain == predictPrimalSVM(primalWeights, xTrain))}")
            print(f"test accuracy: {np.mean(yTest == predictPrimalSVM(primalWeights, xTest))}")
    if primalOrDual == 'Dual':
        for C in Cs:
            print(f"Current C = {C}")
            weightsStar, biasStar, support = trainDualSVM(xTrain, yTrain, C=C)
            print(f"weights: {weightsStar}")
            print(f"bias: {biasStar}")
            print(f"train accuracy: {np.mean(yTrain == predict(weightsStar, biasStar, xTrain))}")
            print(f"test accuracy: {np.mean(yTest == predict(weightsStar, biasStar, xTest))}")


def importData(fileLocation):
    data = pd.read_csv(fileLocation)
    data.loc[data.iloc[:, -1] == 0] = -1
    return np.array(data.iloc[:, :-1]), np.array(data.iloc[:, -1])


if __name__ == "__main__":
    xTrain, yTrain = importData("train.csv")
    xTest, yTest = importData("test.csv")

    print("==== 2a ====")
    Cs = [100 / 873, 500 / 873, 700 / 873]
    runPrimalSVM(Cs, 'Primal')

    print("==== 2b ====")
    Cs = [100 / 873, 500 / 873, 700 / 873]
    runPrimalSVM(Cs, 'Primal')

    print("==== 3a ====")
    Cs = [100 / 873, 500 / 873, 700 / 873]
    runPrimalSVM(Cs, 'Dual')

    print("==== 3b ====")
    Cs = [100 / 873, 500 / 873, 700 / 873]
    gammas = [0.1, 0.5, 1, 5, 100]
    sv = []
    for C in Cs:
        for gamma in gammas:
            print(f"current C = {C}")
            print(f"current gamma = {gamma}")
            weightStar, biasStar, support = trainDualSVM(xTrain, yTrain, C=C, kernel='gaussian', gamma=gamma)
            print(f"weights: {weightStar}")
            print(f"bias: {biasStar}")
            print(f"# SVs: {len(support)}")
            if C == 500 / 873: sv.append(support)
            print(f"train accuracy: {np.mean(yTrain == predict(weightStar, biasStar, xTrain, kernel='gaussian', gamma=gamma))}")
            print(f"test accuracy: {np.mean(yTest == predict(weightStar, biasStar, xTest, kernel='gaussian', gamma=gamma))}")

    for i in range(4):
        count = 0
        for v in np.array(sv[i]):
            if v in np.array(sv[i + 1]):
                count += 1
        print(f"overlap of G = {gammas[i]} to {gammas[i + 1]}: {count}")
