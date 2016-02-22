import random
import time

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

black = (1, 0, 0, 1)
facecolors = [cm.jet(x) for x in np.random.rand(20)]


def show_plot(x, y, centroids, clusters, n):
    plt.figure(n, figsize=(10, 8))
    plt.clf()
    plt.scatter(x[:, 0], x[:, 1], c=y, s=33)
    plt.scatter(centroids[:, 0], centroids[:, 1], c=clusters, s=500, marker="*", linewidths=(3,), edgecolors=(black,),
                cmap='gray')
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')


def lin_init(n):
    X = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(n)])
    return X


def gauss_init(N, k):
    n = float(N) / k
    X = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(0.05, 0.5)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a, b])
        X.extend(x)
    X = np.array(X)[:N]
    return X


class Profiler(object):
    def __enter__(self):
        self._startTime = time.time()

    def __exit__(self, type, value, traceback):
        print "Elapsed time: {:.3f} sec".format(time.time() - self._startTime)
