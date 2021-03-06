import operator
import random
import numpy as np
from numpy import ndarray, mean
from numpy.ma import array

from clustering.util import weighted_choice
from clustering.weight_functions import euclid_dist


class MYKMeans:
    def __init__(self, n_clusters=8, max_iter=100, tol=1e-4, dist_func=euclid_dist, init='kmeans++'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.iterations = max_iter
        self.dist_func = dist_func
        self.centroids = None
        self.inertia = None
        self.init = init

    def default_seed_centroid(self, data):
        return data[random.sample(range(data.shape[0]), self.n_clusters)]

    def kmeans_plus_plus(self, data):
        centroids = []
        first = random.sample(range(data.shape[0]), 1)[0]
        centroids.append(first)
        for index in range(1, self.n_clusters):
            weights = ndarray((data.shape[0],), float)
            for idx, entry in enumerate(data):
                weights[idx] = min([self.dist_func(entry, data[centroid]) for centroid in centroids]) ** 2
            weights[centroids] = 0
            new_random = weighted_choice(weights)
            centroids.append(new_random)
        return data[centroids]

    def seed_centroid(self, data):
        if self.init == 'kmeans++':
            return self.kmeans_plus_plus(data)
        else:
            return self.default_seed_centroid(data)

    def cluster_assign(self, data, centroids):
        clusters = ndarray((data.shape[0],), int)
        self.inertia = 0
        for idx, entry in enumerate(data):
            assigned_centroid = min(enumerate([self.dist_func(entry, centroid) for centroid in centroids]),
                                    key=operator.itemgetter(1))
            self.inertia += assigned_centroid[1]
            clusters[idx] = assigned_centroid[0]
        return clusters

    def move_centroids(self, data, clusters):
        new_centroids = []
        for cluster in range(self.n_clusters):
            cluster_entries = data[clusters == cluster]
            new_centroids.append(mean(cluster_entries, axis=0))
        return new_centroids

    def fit_predict(self, data):
        old_centroids = None
        clusters = None
        centroids = self.seed_centroid(data)
        iteration = 1

        # main loop
        while not self.should_stop(old_centroids, centroids, iteration):
            clusters = self.cluster_assign(data, centroids)
            old_centroids = centroids
            centroids = self.move_centroids(data, clusters)
            iteration += 1

        self.iterations = iteration - 1
        self.centroids = array(centroids)
        return clusters

    def has_converged(self, old_centroids, centroids):
        if old_centroids is None:
            return False
        return np.allclose(old_centroids, centroids, atol=self.tol)

    def should_stop(self, old_centroids, centroids, iteration):
        if iteration > self.max_iter:
            return True
        return self.has_converged(old_centroids, centroids)
