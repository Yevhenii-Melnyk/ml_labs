import matplotlib.pyplot as plt

from clustering.MYKMeans import MYKMeans
from clustering.weight_functions import cosine_dist, squared_dist
from util import show_plot, lin_init, Profiler

data = lin_init(1000)

n_clusters = 8
mykmc = MYKMeans(n_clusters=n_clusters, dist_func=cosine_dist)
with Profiler() as p:
    clusters = mykmc.fit_predict(data)
print mykmc.iterations

mykmc = MYKMeans(n_clusters=n_clusters, dist_func=squared_dist, init='default')
with Profiler() as p:
    clusters = mykmc.fit_predict(data)
print mykmc.iterations

show_plot(data, clusters, mykmc.centroids, [10] * n_clusters, 0)
plt.show()
