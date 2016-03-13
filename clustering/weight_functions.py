import numpy as np


def squared_dist(entry, centroid):
    diff = centroid - entry
    dist = diff.dot(diff)
    return dist


def euclid_dist(entry, centroid):
    diff = centroid - entry
    dist = np.sqrt(diff.dot(diff))
    return dist


def manhattan_dist(entry, centroid):
    diff = centroid - entry
    dist = np.sum(np.absolute(diff))
    return dist


def cosine_dist(entry, centroid):
    numerator = entry.dot(centroid)
    denominator = np.sqrt(entry.dot(entry)) * (np.sqrt(centroid.dot(centroid)))
    dist = numerator / denominator
    return 1 - dist
