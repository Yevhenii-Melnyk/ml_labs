import matplotlib.pyplot as plt
from numpy import array

from MYKMeans import MYKMeans, cosine_dist, euclid_dist, manhattan_dist, squared_dist
from util import show_plot, lin_init, Profiler

data = lin_init(2000)

n_clusters = 8
mykmc = MYKMeans(n_clusters=n_clusters, dist_func=squared_dist)
with Profiler() as p:
    clusters = mykmc.fit_predict(data)
print mykmc.iterations
show_plot(data, clusters, mykmc.centroids, [10] * n_clusters, 0)

plt.show()
