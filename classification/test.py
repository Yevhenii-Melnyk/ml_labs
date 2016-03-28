from pandas import pandas

from classification.bayes import bayes_fit
from classification.bayes import bayes_predict
from classification.knn import getNeighbors, get_knn_class
from classification.tree import decisionTree, classify, printBinaryDecisionTree


def accuracy(y_test, predictions):
    correct = 0
    for i in range(len(y_test)):
        if y_test[i] == predictions[i]:
            correct += 1
    return (correct / float(len(y_test))) * 100.0


# data = pandas.read_csv("iris.csv", header=None)
data = pandas.read_csv("diabetes.csv", header=None)
data = data.sample(frac=1).reset_index(drop=True)

split_count = data.shape[0] * 0.67

data_train = data.ix[0:split_count, :]
X = data.ix[0:split_count, 0: (data.shape[1] - 2)]
y = data.ix[0:split_count, data.shape[1] - 1]

data_test = data.ix[split_count:, :]
X_test = data.ix[split_count:, 0: (data.shape[1] - 2)]
y_test = data.ix[split_count:, data.shape[1] - 1]

# Bayes
summary = bayes_fit(X, y)
predictions = bayes_predict(summary, X_test)
print "Bayes accuracy : ", accuracy(y_test.tolist(), predictions)

# Tree
tree_train = [(row[1].tolist()[0:4], row[1].tolist()[4]) for row in data_train.iterrows()]
tree_test = [row[1].tolist()[0:4] for row in data_test.iterrows()]
tree = decisionTree(tree_train)
#printBinaryDecisionTree(tree, ["sepal l", "sepal w", "petal l", "petal w"])
predictions = []
for row in tree_test:
    result = classify(tree, row)
    predictions.append(result)
print "Tree accuracy : ", accuracy(y_test.tolist(), predictions)

# KNN
knn_train = [row[1].tolist() for row in data_train.iterrows()]
knn_test = [row[1].tolist() for row in data_test.iterrows()]

k = 20
predictions = []
for row in knn_test:
    neighbors = getNeighbors(knn_train, row, k)
    result = get_knn_class(neighbors)
    predictions.append(result)

print "KNN accuracy : ", accuracy(y_test.tolist(), predictions)
