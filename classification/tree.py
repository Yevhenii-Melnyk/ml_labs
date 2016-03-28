import math

from numpy import mean


class Tree:
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []
        self.label = None
        self.classCounts = None
        self.splitFeatureValue = None
        self.splitFeature = None


def dataToDistribution(data):
    allLabels = [label for (point, label) in data]
    numEntries = len(allLabels)
    possibleLabels = set(allLabels)

    dist = []
    for aLabel in possibleLabels:
        dist.append(float(allLabels.count(aLabel)) / numEntries)

    return dist


def entropy(dist):
    return -sum([p * math.log(p, 2) for p in dist])


def splitData(data, featureIndex):
    attrValues = [float(point[featureIndex]) for (point, label) in data]
    meanV = mean(attrValues)
    yield [(point, label) for (point, label) in data if float(point[featureIndex]) <= meanV]
    yield [(point, label) for (point, label) in data if float(point[featureIndex]) > meanV]


def gain(data, featureIndex):
    entropyGain = entropy(dataToDistribution(data))
    for dataSubset in splitData(data, featureIndex):
        entropyGain -= entropy(dataToDistribution(dataSubset))
    return entropyGain


def homogeneous(data):
    return len(set([label for (point, label) in data])) <= 1


def majorityVote(data, node):
    labels = [label for (pt, label) in data]
    choice = max(set(labels), key=labels.count)
    node.label = choice
    node.classCounts = dict([(label, labels.count(label)) for label in set(labels)])

    return node


def buildDecisionTree(data, root, remainingFeatures):
    if homogeneous(data):
        root.label = data[0][1]
        root.classCounts = {root.label: len(data)}
        return root

    if len(remainingFeatures) == 0:
        return majorityVote(data, root)

    bestFeature = max(remainingFeatures, key=lambda index: gain(data, index))

    if gain(data, bestFeature) == 0:
        return majorityVote(data, root)

    root.splitFeature = bestFeature
    root.splitFeatureValue = mean([float(point[bestFeature]) for (point, label) in data])

    for dataSubset in splitData(data, bestFeature):
        aChild = Tree(parent=root)

        root.children.append(aChild)

        buildDecisionTree(dataSubset, aChild, remainingFeatures - set([bestFeature]))

    return root


def decisionTree(data):
    return buildDecisionTree(data, Tree(), set(range(len(data[0][0]))))


def classify(tree, point):
    if tree.children == []:
        return tree.label
    else:
        if point[tree.splitFeature] <= tree.splitFeatureValue:
            return classify(tree.children[0], point)
        else:
            return classify(tree.children[1], point)


def printBinaryDecisionTree(root, labels, indentation=""):
    if root.children == []:
        print "%s %s %s" % (indentation, root.label, root.classCounts)
    else:
        printBinaryDecisionTree(root.children[1], labels, indentation + "\t")
        print "%s%s, %s" % (indentation, root.splitFeatureValue, labels[root.splitFeature])
        printBinaryDecisionTree(root.children[0], labels, indentation + "\t")


if __name__ == '__main__':
    with open('iris.csv', 'r') as inputFile:
        lines = inputFile.readlines()

    data = [line.strip().split(',') for line in lines]
    data = [(x[0:4], x[4]) for x in data]

    cleanData = [x for x in data if '?' not in x[0]]

    tree = decisionTree(cleanData)
    # printBinaryDecisionTree(tree, ["sepal l", "sepal w", "petal l", "petal w"])
    print classify(tree, [6.9, 3.2, 5.7, 2.3])


    # 1. sepal length in cm
    # 2. sepal width in cm
    # 3. petal length in cm
    # 4. petal width in cm
    # 5. class:
