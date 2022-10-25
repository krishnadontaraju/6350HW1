import numpy as np
from os import makedirs
import matplotlib.pyplot as plt

def MSE(pred, target) -> float:
    assert len(pred) == len(target)
    pred, target = np.array(pred), np.array(target)
    return np.sum((target - pred) ** 2) / 2


class LMSWeights:
    def __init__(self, weights: list):
        self.weights = weights

    def __str__(self) -> str:
        return str(self.weights)

    def predict(self, x) -> list:
        return list(map(lambda d: np.dot(self.weights, d), x))


def BatchGradientDescent(x, y, lr: float = 1, epochs: int = 10, threshold=1e-6):
    w = np.ones_like(x[0])

    losses, lastloss, diff = [], 9999, 1
    for ep in range(epochs):
        if diff <= threshold: break
        delJ = np.zeros(w.shape[0])

        for j in range(len(delJ)):
            for xi, yi in zip(x, y):
                delJ[j] -= (yi - np.dot(w, xi)) * xi[j]

        w = w - lr * delJ

        loss = 0
        for xi, yi in zip(x, y):
            loss += (yi - np.dot(w, xi)) ** 2
        loss /= 2

        diff = abs(loss - lastloss)
        lastloss = loss
        losses.append(loss)

    print(f"converged at epoch {ep} to {diff}")
    return LMSWeights(w), losses


def StochasticGradientDescent(x, y, lr: float = 1, epochs: int = 10, threshold=1e-6):

    w = np.ones_like(x[0])

    losses, lastloss, diff = [], 9999, 1
    for ep in range(epochs):
        if diff <= threshold: break

        for xi, yi in zip(x, y):
            for j in range(len(w)):
                w[j] += lr * (yi - np.dot(w, xi)) * xi[j]

            loss = 0
            for xi, yi in zip(x, y):
                loss += (yi - np.dot(w, xi)) ** 2
            loss /= 2

            diff = abs(loss - lastloss)
            lastloss = loss
            losses.append(loss)

    print(f"converged at epoch {ep} to {diff}")
    return LMSWeights(w), losses


def LMSRegression(x, y):
    x = np.transpose(np.array(x))
    y = np.array(y)

    w = np.linalg.inv(x @ np.transpose(x)) @ (x @ y)
    return LMSWeights(w)





cc_train_x = []
cc_train_y = []
with open("concrete/train.csv", "r") as f:
    for line in f:
        terms = line.strip().split(",")
        terms_flt = list(map(lambda x : float(x), terms))
        cc_train_x.append(terms_flt[:-1])
        cc_train_y.append(terms_flt[-1])

cc_test_x = []
cc_test_y = []
with open("concrete/test.csv", "r") as f:
    for line in f:
        terms = line.strip().split(",")
        terms_flt = list(map(lambda x : float(x), terms))
        cc_test_x.append(terms_flt[:-1])
        cc_test_y.append(terms_flt[-1])

print("LMS with Batch Gradient Descent")
bgd, loss_bgd = BatchGradientDescent(cc_train_x, cc_train_y, lr = 1e-3, epochs = 500)

print(f"weight vect: {bgd}")

print("LMS with Stochastic Gradient Descent")
sgd, loss_sgd = StochasticGradientDescent(cc_train_x, cc_train_y, lr = 1e-3, epochs = 500)

print(f"weight vect: {sgd}")

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(np.array(range(len(loss_bgd))) * len(cc_train_x), loss_bgd, color = 'tab:blue', label = "batch")
ax.plot(loss_sgd, color = 'tab:orange', label = "stochastic")
ax.legend()
ax.set_title("Gradient Descent")
ax.set_xlabel("No. of iterations")
ax.set_ylabel("Mean Squared Error")

plt.savefig("MSE.png")
plt.clf()


print("LMS Analytic Method")
lms = LMSRegression(cc_train_x, cc_train_y)

print(f"weight vect: {lms}")
