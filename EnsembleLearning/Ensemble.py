import matplotlib.pyplot as plt

import numpy as np
import multiprocessing as mp
from math import log, exp
from statistics import mode
import random



def error(pred: list, target: list):
    assert len(pred) == len(target)
    mistakes = 0
    for i in range(len(pred)):
        if pred[i] != target[i]: mistakes += 1
    return mistakes / len(pred)


def HandleLine(line):
    terms = line.strip().split(",")
    t_dict = {
        "age": int(terms[0]),
        "job": terms[1],
        "marital": terms[2],
        "education": terms[3],
        "default": terms[4],
        "balance": int(terms[5]),
        "housing": terms[6],
        "loan": terms[7],
        "contact": terms[8],
        "day": int(terms[9]),
        "month": terms[10],
        "duration": int(terms[11]),
        "campaign": int(terms[12]),
        "pdays": int(terms[13]),
        "previous": int(terms[14]),
        "poutcome": terms[15],
        "label": terms[16]
    }
    return t_dict


class AdaBoost:
    def __init__(self):
        self.stumps = list
        self.sample_weights = list
        self.stump_say = list

    def error(self, pred: list, target: list):
        assert len(pred) == len(target)
        err = 0
        for i in range(len(pred)):
            if pred[i] != target[i]: err += self.sample_weights[i]
        return err

    def train(self, data, epochs: int = 100):
        self.sample_weights = [1 / len(data)] * len(data)
        self.stumps = [None] * epochs
        self.stump_say = [None] * epochs

        for i in range(epochs):
            self.stumps[i] = DecisionTree()
            self.stumps[i].makeTree(data=data, weights=list(self.sample_weights), max_depth=1)

            pred = []
            for d in data:
                pred.append(self.stumps[i].predict(d))
            err = self.error(pred, [d['label'] for d in data])

            self.stump_say[i] = 0.5 * log((1 - err) / err)
            # print(self.stump_say[i])

            for j in range(len(self.sample_weights)):
                v = 1 if data[j]['label'] == self.stumps[i].predict(data[j]) else -1
                self.sample_weights[j] = self.sample_weights[j] * exp(-self.stump_say[i] * v)

            self.sample_weights = np.divide(self.sample_weights, np.sum(self.sample_weights))

    def predict(self, data, true_false_values=('yes', 'no')):
        pred = np.zeros_like(data)

        for i, d in enumerate(data):
            Hx = 0
            for j, stump in enumerate(self.stumps):
                Hx += self.stump_say[j] * (1 if stump.predict(d) == true_false_values[0] else -1)
            pred[i] = true_false_values[0] if np.sign(Hx) == 1 else true_false_values[1]

        return pred


def bagAndMakeTree(data, num_samples):
    bag = []
    for _ in range(num_samples):
        x = random.randrange(0, len(data))
        bag.append(data[x])

    tree = DecisionTree()
    tree.makeTree(bag)
    return tree


class BaggedTrees:
    def __init__(self):
        self.trees = list

    def train(self, data: list, num_trees: int = 100, num_samples: int = 1000, num_workers=None):
        mult_data = [data] * num_trees
        mult_samp = [num_samples] * num_trees

        with mp.Pool(num_workers) as pool:
            self.trees = pool.starmap(bagAndMakeTree, zip(mult_data, mult_samp))

    def getFirstTree(self):
        return self.trees[0]

    def predict(self, data, num_workers=4):
        pred = np.zeros_like(data)

        for i, d in enumerate(data):
            pred[i] = mode(map(lambda tree: tree.predict(d), self.trees))

        return pred


def rfBagTree(data, num_samples, num_attributes):
    bag = []
    for _ in range(num_samples):
        x = random.randrange(0, len(data))
        bag.append(data[x])

    tree = RandomForestTree()
    tree.makeTree(bag, num_attributes=num_attributes)
    return tree


class RandomForest:
    def __init__(self):
        self.trees = list

    def train(self, data: list, num_trees: int = 100, num_samples: int = 1000, num_attributes: int = 4,
              num_workers=None):
        mult_data = [data] * num_trees
        mult_samp = [num_samples] * num_trees
        mult_attr = [num_attributes] * num_trees

        with mp.Pool(num_workers) as pool:
            self.trees = pool.starmap(rfBagTree, zip(mult_data, mult_samp, mult_attr))

    def getFirstTree(self):
        return self.trees[0]

    def predict(self, data, num_workers=4):
        pred = np.zeros_like(data)

        for i, d in enumerate(data):
            pred[i] = mode(map(lambda tree: tree.predict(d), self.trees))

        return pred


class TreeNode(object):
    def __init__(self, nodetype=None, attr=None, value=None, finalclass=None):
        self.type = nodetype
        self.attr = attr
        self.value = value
        self.finalclass = finalclass
        self.children = []

    def toJSON(self):
        dict = {
            "type": self.type,
            "attr": self.attr,
            "value": self.value,
            "finalclass": self.finalclass,
            "children": []
        }

        for c in self.children:
            dict["children"].append(c.toJSON())

        return dict


def mostCommon(data, attribute="label"):
    values = list(filter(lambda x: x != "unknown", [d[attribute] for d in data]))
    return mode(values)


def splitAtMedian(data, attribute):
    values = [d[attribute] for d, w in data]
    median = np.median(values)
    lower = []
    upper = []

    for d, w in data:
        if d[attribute] < median:
            lower.append((d, w))
        else:
            upper.append((d, w))

    return lower, upper, median


def GiniIndex(data: list):
    counter = {}
    weight_sum = np.sum([w for d, w in data])

    for d, w in data:
        if counter.get(d["label"]) == None:
            counter[d["label"]] = w
        else:
            counter[d["label"]] += w

    gini = 0
    for v in counter.values():
        gini += (v / weight_sum) ** 2

    return 1 - gini


def InformationGain(data: list, attribute: str, purity=GiniIndex):
    gain = 0
    weight_sum = np.sum([w for d, w in data])
    if type(data[0][0][attribute]) == str:
        unique_vals = np.unique(np.array([d[attribute] for d, w in data]))
        for val in unique_vals:
            subset = []
            for d, w in data:
                if d[attribute] == val:
                    subset.append((d, w))
            gain += (np.sum([w for d, w in subset]) / weight_sum) * purity(subset)

    elif type(data[0][0][attribute] == int):
        lower, upper, _ = splitAtMedian(data, attribute)
        gain = ((np.sum([w for d, w in lower]) / weight_sum) * purity(lower)) + (
                (np.sum([w for d, w in upper]) / weight_sum) * purity(upper))

    return (purity(data) - gain)


def allSame(data):
    return len(np.unique(np.array([d["label"] for d in data]))) == 1


class DecisionTree:
    def __init__(self, purity_function=InformationGain, ):
        self.root = TreeNode(nodetype="root")
        self.purity_function = purity_function
        self.max_depth = 9999
        self.mostLabel = "na"

    def makeTree(self, data: list, weights=None, max_depth: int = None):
        if max_depth != None: self.max_depth = max_depth
        if weights == None: weights = [1 / len(data)] * len(data)
        self.mostLabel = mostCommon(data)
        self.root = self.buildTree(data, weights, self.root, 0, ["label"])

    def buildTree(self, data: list, weights: list, node, depth, used_attrs: list):
        if len(data) == 0:
            node.type = "leaf"
            node.finalclass = self.mostLabel
            return node
        if allSame(data):
            node.type = "leaf"
            node.finalclass = data[0]["label"]
            return node
        if depth >= self.max_depth:
            node.type = "leaf"
            node.finalclass = mostCommon(data)
            return node


        max = {"val": -np.inf, "attr": "none_found"}
        for attr in data[0].keys():
            if attr in used_attrs:
                continue
            purity = self.purity_function(list(zip(data, weights)), attr)

            if purity > max["val"]:
                max["val"] = purity
                max["attr"] = attr

        new_attrs = used_attrs.copy()
        new_attrs.append(max["attr"])


        if max["attr"] == "none_found":
            node.type = "leaf"
            node.finalclass = mostCommon(data)
            return node


        if type(data[0][max["attr"]]) == str:
            unique_vals = np.unique(np.array([d[max["attr"]] for d in data]))
            for val in unique_vals:
                childNode = TreeNode(nodetype="split", attr=max["attr"], value=val)
                new_data = [d for d, w in zip(data, weights) if d[max["attr"]] == val]
                new_weights = [w for d, w in zip(data, weights) if d[max["attr"]] == val]
                node.children.append(self.buildTree(new_data, new_weights, childNode, depth + 1, new_attrs))

        elif type(data[0][max["attr"]]) == int:
            lower, upper, median = splitAtMedian(list(zip(data, weights)), max["attr"])

            lower_data = [d for d, w in lower]
            upper_data = [d for d, w in upper]

            lower_weights = [w for d, w in lower]
            upper_weights = [w for d, w in upper]

            child_lower = TreeNode(nodetype="split", attr=max["attr"], value=(-np.inf, median))
            child_upper = TreeNode(nodetype="split", attr=max["attr"], value=(median, np.inf))

            node.children.append(self.buildTree(lower_data, lower_weights, child_lower, depth + 1, new_attrs))
            node.children.append(self.buildTree(upper_data, upper_weights, child_upper, depth + 1, new_attrs))
        return node


    def toJSON(self):
        return self.root.toJSON()

    def predict(self, value):
        return self._predict(value, self.root)

    def _predict(self, value, node):
        if node.type == "leaf":
            return node.finalclass

        for child in node.children:
            attr = child.attr
            if type(value[attr]) == str:
                if value[attr] == child.value:
                    return self._predict(value, child)
            elif type(value[attr]) == int:
                if (value[attr] >= child.value[0]) & (value[attr] < child.value[1]):
                    return self._predict(value, child)


x_pts = list(range(1, 25)) + list(range(25, 100, 5)) + list(range(100, 550, 50))


class RandomForestTree:
    def __init__(self, purity_function=InformationGain):
        self.root = TreeNode(nodetype="root")
        self.purity_function = purity_function
        self.max_depth = 9999
        self.mostLabel = "na"

    def makeTree(self, data: list, num_attributes: int = 4, max_depth: int = None):
        if max_depth != None:
            self.max_depth = max_depth
        weights = [1 / len(data)] * len(data)
        self.mostLabel = mostCommon(data)
        self.root = self.buildTree(data, weights, num_attributes, self.root, 0, ["label"])

    def buildTree(self, data: list, weights: list, num_attributes, node, depth, used_attrs: list):
        if len(data) == 0:
            node.type = "leaf"
            node.finalclass = self.mostLabel
            return node
        if allSame(data):
            node.type = "leaf"
            node.finalclass = data[0]["label"]
            return node
        if depth >= self.max_depth:
            node.type = "leaf"
            node.finalclass = mostCommon(data)
            return node

        max = {"val": -np.inf, "attr": "none_found"}
        A = list(data[0].keys())
        G = []
        if len(A) - len(used_attrs) <= num_attributes:
            G = A
        else:
            i = 0
            while i < num_attributes:
                idx = random.randrange(0, len(A))
                if A[idx] not in G:
                    G.append(A[idx])
                    i += 1

        for attr in G:
            if attr in used_attrs:
                continue
            purity = self.purity_function(list(zip(data, weights)), attr)
            if purity > max["val"]:
                max["val"] = purity
                max["attr"] = attr

        new_attrs = used_attrs.copy()
        new_attrs.append(max["attr"])

        if max["attr"] == "none_found":
            node.type = "leaf"
            node.finalclass = mostCommon(data)
            return node

        if type(data[0][max["attr"]]) == str:
            unique_vals = np.unique(np.array([d[max["attr"]] for d in data]))
            for val in unique_vals:
                childNode = TreeNode(nodetype="split", attr=max["attr"], value=val)
                new_data = [d for d, w in zip(data, weights) if d[max["attr"]] == val]
                new_weights = [w for d, w in zip(data, weights) if d[max["attr"]] == val]
                node.children.append(
                    self.buildTree(new_data, new_weights, num_attributes, childNode, depth + 1, new_attrs))

        elif type(data[0][max["attr"]]) == int:
            lower, upper, median = splitAtMedian(list(zip(data, weights)), max["attr"])

            lower_data = [d for d, w in lower]
            upper_data = [d for d, w in upper]

            lower_weights = [w for d, w in lower]
            upper_weights = [w for d, w in upper]

            child_lower = TreeNode(nodetype="split", attr=max["attr"], value=(-np.inf, median))
            child_upper = TreeNode(nodetype="split", attr=max["attr"], value=(median, np.inf))

            node.children.append(
                self.buildTree(lower_data, lower_weights, num_attributes, child_lower, depth + 1, new_attrs))
            node.children.append(
                self.buildTree(upper_data, upper_weights, num_attributes, child_upper, depth + 1, new_attrs))
        return node

    def toJSON(self):
        return self.root.toJSON()

    def predict(self, value):
        return self._predict(value, self.root)

    def _predict(self, value, node):
        if node.type == "leaf":
            return node.finalclass

        for child in node.children:
            attr = child.attr
            if type(value[attr]) == str:
                if value[attr] == child.value:
                    return self._predict(value, child)
            elif type(value[attr]) == int:
                if (value[attr] >= child.value[0]) & (value[attr] < child.value[1]):
                    return self._predict(value, child)


def str2Num(value):
    if value == 'no':
        return 0
    else:
        return 1


def array2Dict(data, header):
    out = [None] * len(data)

    for i, d in enumerate(data):
        out[i] = {}
        for j, label in enumerate(header):
            try:
                val = int(d[j])
            except ValueError:
                val = d[j]
            out[i][label] = val

    return out


def RunBoosting(dataset_loc):
    train_bank = []
    with open(dataset_loc + "train.csv", "r") as f:
        for line in f:
            train_bank.append(HandleLine(line))

    test_bank = []
    with open(dataset_loc + "test.csv", "r") as f:
        for line in f:
            test_bank.append(HandleLine(line))

    print("datasets loaded")

    print("adaboost")
    ada = AdaBoost()
    ada.train(train_bank, 5)

    trainPred = ada.predict(train_bank)
    print(error(trainPred, [d['label'] for d in train_bank]))

    testPred = ada.predict(test_bank)
    print(error(testPred, [d['label'] for d in test_bank]))

    print("running bagged trees...")
    train_err = []
    test_err = []

    for x in x_pts:
        print(f"# trees: {x}")

        bag = BaggedTrees()
        bag.train(train_bank, num_trees=x, num_samples=1000)

        trainPred = bag.predict(train_bank)
        train_err.append(error(trainPred, [d['label'] for d in train_bank]))

        testPred = bag.predict(test_bank)
        test_err.append(error(testPred, [d['label'] for d in test_bank]))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(x_pts, train_err, color='tab:blue', label="training")
    ax.plot(x_pts, test_err, color='tab:orange', label="testing")
    ax.legend()
    ax.set_title("Bagged Trees")
    ax.set_xlabel("# of trees")
    ax.set_ylabel("Misclassification Error")

    plt.savefig("bagged.png")
    plt.clf()


def RunRandomForest(dataset_loc):
    train_bank = []
    with open(dataset_loc + "train.csv", "r") as f:
        for line in f:
            train_bank.append(HandleLine(line))

    test_bank = []
    with open(dataset_loc + "test.csv", "r") as f:
        for line in f:
            test_bank.append(HandleLine(line))

    print("datasets loaded")

    print("running random forest...")
    train_err_2 = []
    train_err_4 = []
    train_err_6 = []
    test_err_2 = []
    test_err_4 = []
    test_err_6 = []

    for x in x_pts:
        print(f"# trees: {x}")

        rf_2 = RandomForest()
        rf_2.train(train_bank, num_trees=x, num_samples=1000, num_attributes=2)

        trainPred = rf_2.predict(train_bank)
        train_err_2.append(error(trainPred, [d['label'] for d in train_bank]))

        testPred = rf_2.predict(test_bank)
        test_err_2.append(error(testPred, [d['label'] for d in test_bank]))

        rf_4 = RandomForest()
        rf_4.train(train_bank, num_trees=x, num_samples=1000, num_attributes=4)

        trainPred = rf_4.predict(train_bank)
        train_err_4.append(error(trainPred, [d['label'] for d in train_bank]))

        testPred = rf_4.predict(test_bank)
        test_err_4.append(error(testPred, [d['label'] for d in test_bank]))

        rf_6 = RandomForest()
        rf_6.train(train_bank, num_trees=x, num_samples=1000, num_attributes=6)

        trainPred = rf_6.predict(train_bank)
        train_err_6.append(error(trainPred, [d['label'] for d in train_bank]))

        testPred = rf_6.predict(test_bank)
        test_err_6.append(error(testPred, [d['label'] for d in test_bank]))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x_pts, train_err_2, label="training, |G| = 2")
    ax.plot(x_pts, test_err_2, label="testing, |G| = 2")
    ax.plot(x_pts, train_err_4, label="training, |G| = 4")
    ax.plot(x_pts, test_err_4, label="testing, |G| = 4")
    ax.plot(x_pts, train_err_6, label="training, |G| = 6")
    ax.plot(x_pts, test_err_6, label="testing, |G| = 6")
    ax.legend()
    ax.set_title("Random Forest")
    ax.set_xlabel("# of trees")
    ax.set_ylabel("Misclassification Error")

    plt.savefig("RF.png")
    plt.clf()


def RunBiasAndVariance(dataset_loc):
    bank_raw_train = [["age", "job", "marital", "education",
                       "default", "balance", "housing", "loan",
                       "contact", "day", "month", "duration",
                       "campaign", "pdays", "previous", "poutcome", "label"]]

    with open(dataset_loc + "train.csv", "r") as f:
        for line in f:
            terms = line.strip().split(",")
            bank_raw_train.append(terms)

    bank_raw_test = []
    with open(dataset_loc + "test.csv", "r") as f:
        for line in f:
            terms = line.strip().split(",")
            bank_raw_test.append(terms)

    bank = np.array(array2Dict(bank_raw_train[1:], bank_raw_train[0]))
    test_bank = np.array(array2Dict(bank_raw_test, bank_raw_train[0]))
    idx = list(range(len(bank)))

    bagged_trees = []
    single_trees = []
    for i in range(1):
        random.shuffle(idx)
        train_bank = bank[idx[:1000]]
        bag = BaggedTrees()
        bag.train(train_bank, num_trees=500, num_samples=500)
        bagged_trees.append(bag)
        single_trees.append(bag.getFirstTree())

    bias_single, bias_bagged, var_single, var_bagged = [], [], [], []
    for d in test_bank:
        bagged = list(map(lambda t: str2Num(t.predict([d])[0]), bagged_trees))
        single = list(map(lambda t: str2Num(t.predict(d)), single_trees))
        lab = str2Num(d['label'])

        bias_single.append((lab - np.mean(single)) ** 2)
        bias_bagged.append((lab - np.mean(bagged)) ** 2)
        var_single.append(np.std(single) ** 2)
        var_bagged.append(np.std(bagged) ** 2)

    print(
        f"single tree: \n    bias: {np.mean(bias_single)}\n    variance: {np.mean(var_single)}\n    GSE: {np.mean(bias_single) + np.mean(var_single)}")
    print(
        f"bagged trees: \n    bias: {np.mean(bias_bagged)}\n    variance: {np.mean(var_bagged)}\n    GSE: {np.mean(bias_bagged) + np.mean(var_bagged)}")

    randomforests = []
    single_trees = []
    for i in range(100):
        random.shuffle(idx)
        train_bank = bank[idx[:1000]]
        rf = RandomForest()
        rf.train(train_bank, num_trees=500, num_samples=500)
        randomforests.append(rf)
        single_trees.append(rf.getFirstTree())

    bias_single, bias_randfor, var_single, var_randfor = [], [], [], []
    for d in test_bank:
        randfor = list(map(lambda t: str2Num(t.predict([d])[0]), randomforests))
        single = list(map(lambda t: str2Num(t.predict(d)), single_trees))
        lab = str2Num(d['label'])

        bias_single.append((lab - np.mean(single)) ** 2)
        bias_randfor.append((lab - np.mean(randfor)) ** 2)
        var_single.append(np.std(single) ** 2)
        var_randfor.append(np.std(randfor) ** 2)

    print(
        f"single tree: \n    bias: {np.mean(bias_single)}\n    variance: {np.mean(var_single)}\n    GSE: {np.mean(bias_single) + np.mean(var_single)}")
    print(
        f"random forest: \n    bias: {np.mean(bias_randfor)}\n    variance: {np.mean(var_randfor)}\n    GSE: {np.mean(bias_randfor) + np.mean(var_randfor)}")


def RunDefaultPrediction(dataset_loc):
    default_raw = []
    with open(dataset_loc + "default.csv", "r") as f:
        for line in f:
            terms = line.strip().split(",")
            default_raw.append(terms)

    default = np.array(array2Dict(default_raw[1:], default_raw[0]))
    idx = list(range(len(default)))
    random.shuffle(idx)

    default_train = default[idx[:24000]]
    default_test = default[idx[24000:]]

    print("training a tree...")
    tree = DecisionTree()
    tree.makeTree(default_train)

    pred_train = []
    for d in default_train:
        pred_train.append(tree.predict(d))
    train_tree = error(pred_train, [d['label'] for d in default_train])

    pred_test = []
    for d in default_test:
        pred_test.append(tree.predict(d))
    test_tree = error(pred_test, [d['label'] for d in default_test])

    print(f"tree: training error = {train_tree}, testing error = {test_tree}")

    print("running bagged trees...")
    train_bag = []
    test_bag = []

    for x in x_pts:
        print(f"# trees: {x}")

        bag = BaggedTrees()
        bag.train(default_train, num_trees=x, num_samples=1000)

        trainPred = bag.predict(default_train)
        train_bag.append(error(trainPred, [d['label'] for d in default_train]))

        testPred = bag.predict(default_test)
        test_bag.append(error(testPred, [d['label'] for d in default_test]))

    print("running random forests..")
    train_rf = []
    test_rf = []

    for x in x_pts:
        print(f"# trees: {x}")

        bag = BaggedTrees()
        bag.train(default_train, num_trees=x, num_samples=1000)

        trainPred = bag.predict(default_train)
        train_rf.append(error(trainPred, [d['label'] for d in default_train]))

        testPred = bag.predict(default_test)
        test_rf.append(error(testPred, [d['label'] for d in default_test]))

    print(f"bagged trees: training error = {train_bag[-1]}, testing error = {test_bag[-1]}")
    print(f"random forest: training error = {train_rf[-1]}, testing error = {test_rf[-1]}")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x_pts, train_bag, color='tab:green', label="training, bagged trees")
    ax.plot(x_pts, test_bag, color='tab:blue', label="testing, bagged trees")
    ax.plot(x_pts, train_rf, color='tab:red', label="training, random forest")
    ax.plot(x_pts, test_rf, color='tab:orange', labxel="testing, random forest")
    ax.legend()
    ax.set_xlabel("# of trees")
    ax.set_ylabel("Misclassification Error")
    print("Printed Figure")
    plt.savefig("Bonus.png")


if __name__ == '__main__':
    RunBoosting("bank/")
    RunRandomForest("bank/")
    RunBiasAndVariance("bank/")
    RunDefaultPrediction("default/")