import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from os import makedirs
import matplotlib.pyplot as plt



class sigmoid:
    def __call__(self, x: float) -> float:
        z = np.exp(-x)
        sig = 1 / (1 + z)
        return sig

    def deriv(self, x):
        z = np.exp(-x)
        sig = 1 / (1 + z)
        return sig * (1 - sig)


class identity:
    def __call__(self, x: float) -> float:
        return x

    def deriv(self, x):
        return 1


class FCLayer:
    def __init__(self, in_channels, out_channels, activation_function, weight_init, include_bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        if activation_function == 'sigmoid':
            self.activation_function = sigmoid()
        elif activation_function == 'identity':
            self.activation_function = identity()
        else:
            raise NotImplementedError

        if include_bias:
            shape = (self.in_channels + 1, self.out_channels + 1)
        else:
            shape = (self.in_channels + 1, self.out_channels)

        if weight_init == 'zeroes':
            self.layer_weights = np.zeros(shape, dtype=np.float16)
        elif weight_init == 'random':
            self.layer_weights = np.random.standard_normal(shape)
        else:
            raise NotImplementedError

    def __str__(self) -> str:
        return str(self.layer_weights)

    def eval(self, x):
        return self.activation_function(np.dot(x, self.layer_weights))

    def backwards(self, zs, partials):
        delta = np.dot(partials[-1], self.layer_weights.T)
        delta *= self.activation_function.deriv(zs)
        return delta

    def update_ws(self, lr, zs, partials):
        grad = np.dot(zs.T, partials)
        self.layer_weights += -lr * grad
        return grad


class NeuralNetworkPrimary:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        x = np.append(1, x)
        zs = [np.atleast_2d(x)]

        for l in range(len(self.layers)):
            out = self.layers[l].eval(zs[l])
            zs.append(out)

        return float(zs[-1]), zs

    def backward(self, zs, y, lr=0.1, display=False):

        partials = [zs[-1] - y]

        for l in range(len(zs) - 2, 0, -1):
            delta = self.layers[l].backwards(zs[l], partials)
            partials.append(delta)

        partials = partials[::-1]

        for l in range(len(self.layers)):
            grad = self.layers[l].update_ws(lr, zs[l], partials[l])
            if display: print(f"gradient of layer {l + 1}: \n{grad}")


class RegressionDataset(Dataset):
    def __init__(self, datafile):
        xs = []
        ys = []
        with open(datafile, "r") as f:
            for line in f:
                terms = line.strip().split(",")
                terms_flt = list(map(lambda x: np.float32(x), terms))
                xs.append(terms_flt[:-1])
                ys.append(terms_flt[-1])

        self.xs = np.array(xs)
        self.ys = np.array(ys)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y


train_data = RegressionDataset('train.csv')
test_data = RegressionDataset('train.csv')

batch_size = 10
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for x, y in test_dataloader:
    print("x:", x)
    print("y:", y)
    break

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(dataloader, model, loss_fn, optimizer, bool):
    model.train()
    train_loss = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(torch.reshape(pred, y.shape), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            train_loss.append(loss.item())
    if bool:
        print(f"training error: {np.mean(train_loss):>8f}")
    return train_loss


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)

            test_loss += loss_fn(torch.reshape(pred, y.shape), y).item()
    test_loss /= num_batches
    print(f"test error: {test_loss:>8f} \n")


def init_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)


def init_he(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)


widths = [5, 10, 25, 50, 100]
depths = [3, 5, 9]
activations = [(nn.ReLU(), init_he, "ReLU"), (nn.Tanh(), init_xavier, "Tanh")]

for ac_fn, init_fn, ac_name in activations:
    print(f"using activation function {ac_name}")
    for width in widths:
        for depth in depths:

            print(f"{depth}-deep, {width}-wide network:\n-------------------------------")

            class NeuralNetwork(nn.Module):
                def __init__(self):
                    super(NeuralNetwork, self).__init__()
                    self.input = nn.Sequential(nn.Linear(4, width), ac_fn)
                    self.body = nn.ModuleList([])
                    for i in range(depth - 2):
                        self.body.append(nn.Sequential(nn.Linear(width, width), ac_fn))
                    self.out = nn.Linear(width, 1)

                def forward(self, x):
                    x = self.input(x)
                    for layer in self.body:
                        x = layer(x)
                    res = self.out(x)
                    return res


            model = NeuralNetwork().to(device)
            model.apply(init_fn)

            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            train_losses = np.array([])
            epochs = 15
            for t in range(epochs):
                epoch_losses = train(train_dataloader, model, loss_fn, optimizer, t + 1 == epochs)
                train_losses = np.append(train_losses, epoch_losses)

            test(test_dataloader, model, loss_fn)

print("Done!\nPlots saved in './out/'")



def square_loss(pred, target):
    return 0.5 * (pred - target) ** 2


train_x = []
train_y = []
with open("train.csv", "r") as f:
    for line in f:
        terms = line.strip().split(",")
        terms_flt = list(map(lambda x: np.float16(x), terms))
        train_x.append(terms_flt[:-1])
        train_y.append(terms_flt[-1])

train_x = np.array(train_x)
train_y = np.array(train_y)

test_x = []
test_y = []
with open("test.csv", "r") as f:
    for line in f:
        terms = line.strip().split(",")
        terms_flt = list(map(lambda x: np.float16(x), terms))
        test_x.append(terms_flt[:-1])
        test_y.append(terms_flt[-1])

test_x = np.array(test_x)
test_y = np.array(test_y)


def train(num_epochs, net, train_x, train_y, lr_0=0.5, d=1):
    all_losses = []

    for e in range(num_epochs):
        losses = []
        idxs = np.arange(len(train_x))
        np.random.shuffle(idxs)
        for i in idxs:
            y, zs = net.forward(train_x[i])
            losses.append(square_loss(y, train_y[i]))

            lr = lr_0 / (1 + (lr_0 / d) * e)
            net.backward(zs, train_y[i], lr)
        if e + 1 == num_epochs:
            print(f"training error: {np.mean(losses):>8f}")
        all_losses.append(np.mean(losses))

    return all_losses


def test(net, test_x, test_y):
    losses = []
    for i in range(len(test_x)):
        y, _ = net.forward(test_x[i])
        losses.append(square_loss(y, test_y[i]))
    print(f"testing error: {np.mean(losses):>8f}\n")

    return np.mean(losses)


if __name__ == "__main__":
    print("5-wide network:\n-------------------------------")
    newArray = [
        FCLayer(in_channels=4, out_channels=5, activation_function='sigmoid', weight_init='random'),
        FCLayer(in_channels=5, out_channels=5, activation_function='sigmoid', weight_init='random'),
        FCLayer(in_channels=5, out_channels=1, activation_function='identity', weight_init='random', include_bias=False)
    ]
    net = NeuralNetworkPrimary(newArray)

    training_acc = train(35, net, train_x, train_y, lr_0=0.5, d=1)
    testing_acc = test(net, test_x, test_y)

    print("10-wide network:\n-------------------------------")
    net = NeuralNetworkPrimary([
        FCLayer(in_channels=4, out_channels=10, activation_function='sigmoid', weight_init='random'),
        FCLayer(in_channels=10, out_channels=10, activation_function='sigmoid', weight_init='random'),
        FCLayer(in_channels=10, out_channels=1, activation_function='identity', weight_init='random',
                include_bias=False)
    ])

    training_acc = train(35, net, train_x, train_y, lr_0=0.5, d=1)
    testing_acc = test(net, test_x, test_y)

    print("25-wide network:\n-------------------------------")
    net = NeuralNetworkPrimary([
        FCLayer(in_channels=4, out_channels=25, activation_function='sigmoid', weight_init='random'),
        FCLayer(in_channels=25, out_channels=25, activation_function='sigmoid', weight_init='random'),
        FCLayer(in_channels=25, out_channels=1, activation_function='identity', weight_init='random',
                include_bias=False)
    ])

    training_acc = train(35, net, train_x, train_y, lr_0=0.05, d=1)
    testing_acc = test(net, test_x, test_y)

    print("50-wide network:\n-------------------------------")
    net = NeuralNetworkPrimary([
        FCLayer(in_channels=4, out_channels=50, activation_function='sigmoid', weight_init='random'),
        FCLayer(in_channels=50, out_channels=50, activation_function='sigmoid', weight_init='random'),
        FCLayer(in_channels=50, out_channels=1, activation_function='identity', weight_init='random',
                include_bias=False)
    ])

    training_acc = train(35, net, train_x, train_y, lr_0=0.1, d=1)
    testing_acc = test(net, test_x, test_y)

    print("100-wide network:\n-------------------------------")
    net = NeuralNetworkPrimary([
        FCLayer(in_channels=4, out_channels=100, activation_function='sigmoid', weight_init='random'),
        FCLayer(in_channels=100, out_channels=100, activation_function='sigmoid', weight_init='random'),
        FCLayer(in_channels=100, out_channels=1, activation_function='identity', weight_init='random',
                include_bias=False)
    ])

    training_acc = train(35, net, train_x, train_y, lr_0=0.01, d=2)
    testing_acc = test(net, test_x, test_y)

    plt.show()

    print("testing net")
    net = NeuralNetworkPrimary([
        FCLayer(in_channels=4, out_channels=5, activation_function='sigmoid', weight_init='random'),
        FCLayer(in_channels=5, out_channels=5, activation_function='sigmoid', weight_init='random'),
        FCLayer(in_channels=5, out_channels=1, activation_function='identity', weight_init='random', include_bias=False)
    ])

    x = np.array([1, 1, 1, 1])
    ystar = 1
    y, A = net.forward(x)
    net.backward(A, ystar, display=True)


    def square_loss(pred, target):
        return 0.5 * (pred - target) ** 2


    train_x = []
    train_y = []
    with open("train.csv", "r") as f:
        for line in f:
            terms = line.strip().split(",")
            terms_flt = list(map(lambda x: np.float16(x), terms))
            train_x.append(terms_flt[:-1])
            train_y.append(terms_flt[-1])

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    test_x = []
    test_y = []
    with open("test.csv", "r") as f:
        for line in f:
            terms = line.strip().split(",")
            terms_flt = list(map(lambda x: np.float16(x), terms))
            test_x.append(terms_flt[:-1])
            test_y.append(terms_flt[-1])

    test_x = np.array(test_x)
    test_y = np.array(test_y)


    def train(num_epochs, net, train_x, train_y, lr_0=0.5, d=1):
        all_losses = []

        for e in range(num_epochs):
            losses = []
            idxs = np.arange(len(train_x))
            np.random.shuffle(idxs)
            for i in idxs:
                y, zs = net.forward(train_x[i])
                losses.append(square_loss(y, train_y[i]))

                lr = lr_0 / (1 + (lr_0 / d) * e)
                net.backward(zs, train_y[i], lr)
            if e + 1 == num_epochs:
                print(f"training error: {np.mean(losses):>8f}")
            all_losses.append(np.mean(losses))

        return all_losses


    def test(net, test_x, test_y):
        losses = []
        for i in range(len(test_x)):
            y, _ = net.forward(test_x[i])
            losses.append(square_loss(y, test_y[i]))
        print(f"testing error: {np.mean(losses):>8f}\n")

        return np.mean(losses)


    print("5-wide network:\n-------------------------------")
    net = NeuralNetworkPrimary([
        FCLayer(in_channels=4, out_channels=5, activation_function='sigmoid', weight_init='zeroes'),
        FCLayer(in_channels=5, out_channels=5, activation_function='sigmoid', weight_init='zeroes'),
        FCLayer(in_channels=5, out_channels=1, activation_function='identity', weight_init='zeroes', include_bias=False)
    ])

    training_acc = train(35, net, train_x, train_y, lr_0=0.5, d=1)
    testing_acc = test(net, test_x, test_y)

    print("10-wide network:\n-------------------------------")
    net = NeuralNetworkPrimary([
        FCLayer(in_channels=4, out_channels=10, activation_function='sigmoid', weight_init='zeroes'),
        FCLayer(in_channels=10, out_channels=10, activation_function='sigmoid', weight_init='zeroes'),
        FCLayer(in_channels=10, out_channels=1, activation_function='identity', weight_init='zeroes',
                include_bias=False)
    ])

    training_acc = train(35, net, train_x, train_y, lr_0=0.5, d=1)
    testing_acc = test(net, test_x, test_y)

    print("25-wide network:\n-------------------------------")
    net = NeuralNetworkPrimary([
        FCLayer(in_channels=4, out_channels=25, activation_function='sigmoid', weight_init='zeroes'),
        FCLayer(in_channels=25, out_channels=25, activation_function='sigmoid', weight_init='zeroes'),
        FCLayer(in_channels=25, out_channels=1, activation_function='identity', weight_init='zeroes',
                include_bias=False)
    ])

    training_acc = train(35, net, train_x, train_y, lr_0=0.05, d=1)
    testing_acc = test(net, test_x, test_y)

    print("50-wide network:\n-------------------------------")
    net = NeuralNetworkPrimary([
        FCLayer(in_channels=4, out_channels=50, activation_function='sigmoid', weight_init='zeroes'),
        FCLayer(in_channels=50, out_channels=50, activation_function='sigmoid', weight_init='zeroes'),
        FCLayer(in_channels=50, out_channels=1, activation_function='identity', weight_init='zeroes',
                include_bias=False)
    ])

    training_acc = train(35, net, train_x, train_y, lr_0=0.1, d=1)
    testing_acc = test(net, test_x, test_y)

    print("100-wide network:\n-------------------------------")
    net = NeuralNetworkPrimary([
        FCLayer(in_channels=4, out_channels=100, activation_function='sigmoid', weight_init='zeroes'),
        FCLayer(in_channels=100, out_channels=100, activation_function='sigmoid', weight_init='zeroes'),
        FCLayer(in_channels=100, out_channels=1, activation_function='identity', weight_init='zeroes',
                include_bias=False)
    ])

    training_acc = train(35, net, train_x, train_y, lr_0=0.01, d=2)
    testing_acc = test(net, test_x, test_y)
