from clustering.MYKMeans import MYKMeans
from clustering.MYKMedoids import MYKMedoids
from clustering.util import gauss_init, Profiler, show_plot
from clustering.weight_functions import squared_dist
import matplotlib.pyplot as plt

n_clusters = 4
data = gauss_init(2000, n_clusters)

mykmc = MYKMeans(n_clusters=n_clusters, dist_func=squared_dist, init="default")
with Profiler():
    clusters = mykmc.fit_predict(data)
print mykmc.iterations

show_plot(data, clusters, mykmc.centroids, [10] * n_clusters, 0)

mykmc = MYKMedoids(n_clusters=n_clusters, dist_func=squared_dist)
with Profiler():
    clusters = mykmc.fit_predict(data)
print mykmc.iterations

show_plot(data, clusters, mykmc.medoids, [10] * n_clusters, 1)

plt.show()
