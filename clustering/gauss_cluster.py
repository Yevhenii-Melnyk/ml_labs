import matplotlib.pyplot as plt

from MYKMeans import MYKMeans, euclid_dist
from util import gauss_init

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


