import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

from MYKMeans import MYKMeans
from util import show_plot

iris = datasets.load_iris()
X = iris.data
y = iris.target

mykmc = MYKMeans(n_clusters=3)
clusters = mykmc.fit_predict(X)

fig, plots = plt.subplots(2, 2, figsize=(10, 10))

# original
plots[0][0].scatter(X[:, 0], X[:, 1], c=y, s=33)
plots[0][0].set_xlabel('Sepal length')
plots[0][0].set_ylabel('Sepal width')
plots[0][1].scatter(X[:, 2], X[:, 3], c=y, s=33)
plots[0][1].set_xlabel('Petal length')
plots[0][1].set_ylabel('Petal width')

# clustered
plots[1][0].scatter(X[:, 0], X[:, 1], c=clusters, s=33)
plots[1][0].set_xlabel('Sepal length')
plots[1][0].set_ylabel('Sepal width')
plots[1][1].scatter(X[:, 2], X[:, 3], c=clusters, s=33)
plots[1][1].set_xlabel('Petal length')
plots[1][1].set_ylabel('Petal width')

plt.show()
