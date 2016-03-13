import random
import copy

from numpy.ma import array

from clustering.weight_functions import euclid_dist
import numpy as np


class MYKMedoids:
    def __init__(self, n_clusters=8, max_iter=100, tol=1e-4, dist_func=euclid_dist):
        self.n_clusters = n_clusters

        self.max_iter = max_iter
        self.tol = tol
        self.iterations = max_iter
        self.dist_func = dist_func
        self.medoids = None
        self.inertia = None

    def compute_distances(self, data):
        distance_matrix = []
        for entry in data:
            distance_matrix.append([self.dist_func(entry, anotherEntry) for anotherEntry in data])
        return array(distance_matrix)

    def assign_points_to_clusters(self, medoids, distances):
        distances_to_medoids = distances[:, medoids]
        return medoids[np.argmin(distances_to_medoids, axis=1)]

    def compute_new_medoid(self, cluster, distances):
        mask = np.ones(distances.shape)
        mask[np.ix_(cluster, cluster)] = 0.
        cluster_distances = np.ma.masked_array(data=distances, mask=mask, fill_value=10e9)
        costs = cluster_distances.sum(axis=1)
        return costs.argmin(axis=0, fill_value=10e9)

    def fit_predict(self, data):
        distances = self.compute_distances(data)
        clusters = None

        old_medoids = np.array([-1] * self.n_clusters)
        medoids = array(random.sample(range(data.shape[0]), self.n_clusters))

        iteration = 1
        while not self.should_stop(old_medoids, medoids, iteration):
            clusters = self.assign_points_to_clusters(medoids, distances)

            old_medoids[:] = medoids[:]
            for idx, curr_medoid in enumerate(medoids):
                cluster = np.where(clusters == curr_medoid)[0]
                medoids[idx] = self.compute_new_medoid(cluster, distances)
                iteration += 1

        self.iterations = iteration - 1
        self.medoids = data[medoids.data]
        return clusters.data

    def should_stop(self, old_medoids, medoids, iteration):
        if iteration > self.max_iter:
            return True
        return (old_medoids == medoids).all()
