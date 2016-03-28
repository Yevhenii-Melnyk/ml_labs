import scipy.stats


def separateByClass(X, y):
    classes = y.unique()
    separated = {}
    for clazz in classes:
        separated[clazz] = X[y == clazz]
    return separated


def summarize(separated):
    summaries = {}
    for clazz, data in separated.iteritems():
        summaries[clazz] = [(column[1].describe()['mean'], column[1].describe()['std']) for column in data.iteritems()]
    return summaries


def bayes_fit(X, y):
    separated = separateByClass(X, y)
    return summarize(separated)


def calculateProbability(x, mean, stdev):
    return scipy.stats.norm(mean, stdev).pdf(x)


def calculateClassProbabilities(summaries, row):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1.
        for idx, (mean, stdev) in enumerate(classSummaries):
            probabilities[classValue] *= calculateProbability(row[idx], mean, stdev)
    return probabilities


def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def bayes_predict(summaries, testSet):
    predictions = []
    for row in testSet.iterrows():
        result = predict(summaries, row[1])
        predictions.append(result)
    return predictions
