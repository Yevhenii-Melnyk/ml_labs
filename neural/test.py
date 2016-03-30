import numpy
import pandas

from neural.nn import train_nn, predict_nn

data = pandas.read_csv("diabetes.csv", header=None)
data = data.sample(frac=1).reset_index(drop=True)

split_count = data.shape[0] * 0.67

# train
data_train = data.ix[0:split_count, :]
X = data.ix[0:split_count, 0: (data.shape[1] - 2)]
y = data.ix[0:split_count, data.shape[1] - 1]

min, max, mean = X.min(), X.max(), X.mean()
X_train = (X - mean) / (max - min)
X_train = X_train.as_matrix()
y_train = y.as_matrix().reshape(y.size, 1)

thetas = train_nn(X_train, y_train, X_train.shape[1])

# test
data_test = data.ix[split_count:, :]
X_test = data.ix[split_count:, 0: (data.shape[1] - 2)]
y_test = data.ix[split_count:, data.shape[1] - 1]

X_test = (X_test - mean) / (max - min)
X_test = X_test.as_matrix()
y_test = y_test.as_matrix().reshape(y_test.size, 1)

for theta_0, theta_1 in thetas:
    y_predicted = predict_nn(theta_0, theta_1, X_test)
    y_predicted = (y_predicted > 0.5).astype(int)

    total_count = y_predicted.size
    print "total", total_count
    correct_count = (y_predicted == y_test).sum()
    print "correct", correct_count
    print float(correct_count) / total_count
    print "################"
