import numpy as np

# alphas = [0.001, 0.01, 0.1, 1, 10]
alphas = [0.01]
hiddenSize = 32


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


def sigmoid_output_to_derivative(output):
    return output * (1 - output)


def train_nn(X, y, n):
    iterations = 60000
    thetas = []
    for alpha in alphas:
        print "Training With Alpha:" + str(alpha)
        np.random.seed(1)

        theta_1 = 2 * np.random.random((n + 1, hiddenSize)) - 1
        theta_2 = 2 * np.random.random((hiddenSize + 1, 1)) - 1

        for j in xrange(iterations + 1):
            layer_0 = np.insert(X, 0, 1, axis=1)
            layer_1 = sigmoid(np.dot(layer_0, theta_1))
            layer_1 = np.insert(layer_1, 0, 1, axis=1)
            layer_2 = sigmoid(np.dot(layer_1, theta_2))

            layer_2_error = layer_2 - y
            layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

            layer_1_error = layer_2_delta.dot(theta_2.T)
            layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
            layer_1_delta = layer_1_delta[:, 1:]

            layer_2_delta = layer_1.T.dot(layer_2_delta)
            layer_2_delta[:, 0] = 0

            layer_1_delta = layer_0.T.dot(layer_1_delta)
            layer_1_delta[:, 0] = 0

            theta_2 -= alpha * layer_2_delta
            theta_1 -= alpha * layer_1_delta

            if (j % iterations) == 0:
                print "Error after " + str(j) + " iterations:" + str(np.mean(np.abs(layer_2_error)))
        thetas.append((theta_1, theta_2))
    return thetas


def predict_nn(theta_0, theta_1, X_test):
    X = np.insert(X_test, 0, 1, axis=1)
    layer_1 = sigmoid(np.dot(X, theta_0))
    layer_1 = np.insert(layer_1, 0, 1, axis=1)
    return sigmoid(np.dot(layer_1, theta_1))
