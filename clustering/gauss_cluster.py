import matplotlib.pyplot as plt

from MYKMeans import MYKMeans
from clustering.weight_functions import euclid_dist, squared_dist
from util import gauss_init, Profiler, show_plot

data = gauss_init(200, 5)

inertia = []
iterations = []
n_range = range(2, 11)
for n_clusters in n_range:
    mykmc = MYKMeans(n_clusters=n_clusters, dist_func=euclid_dist)
    mykmc.fit_predict(data)
    inertia.append(mykmc.inertia)
    iterations.append(mykmc.iterations)

fig, plots = plt.subplots(2, figsize=(10, 8))
plots[0].plot(n_range, inertia)
plots[1].plot(n_range, iterations)
plt.show()

n_clusters = 4
data = gauss_init(2000, n_clusters)

mykmc = MYKMeans(n_clusters=n_clusters, dist_func=squared_dist)
with Profiler() as p:
    clusters = mykmc.fit_predict(data)
print mykmc.iterations

show_plot(data, clusters, mykmc.centroids, [10] * n_clusters, 0)
plt.show()
