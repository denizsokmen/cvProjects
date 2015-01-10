import numpy as np
from sklearn.neighbors import BallTree
from sklearn.utils import extmath

# For the full-blown implementation, see www.scikit-learn.org

def mean_shift(X, bandwidth, seeds, kernel_update_function, max_iterations=10):
    n_points, n_features = X.shape
    stop_thresh = 1e-3 * bandwidth  # when mean has converged                                                                                                               
    cluster_centers = []
    ball_tree = BallTree(X)  # to efficiently look up nearby points

    # For each seed, climb gradient until convergence or max_iterations                                                                                                     
    for weighted_mean in seeds:
         completed_iterations = 0
         while True:
             points_within = X[ball_tree.query_radius([weighted_mean], bandwidth*3)[0]]
             old_mean = weighted_mean  # save the old mean                                                                                                                  
             weighted_mean = kernel_update_function(old_mean, points_within, bandwidth)
             converged = extmath.norm(weighted_mean - old_mean) < stop_thresh
             if converged or completed_iterations == max_iterations:
                 cluster_centers.append(weighted_mean)
                 break
             completed_iterations += 1

    return cluster_centers

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def gaussian_kernel_update(x, points, bandwidth):
    distances = euclidean_distances(points, x)
    weights = np.exp(-1 * (distances ** 2 / bandwidth ** 2))
    return np.sum(points * weights, axis=0) / np.sum(weights)

def flat_kernel_update(x, points, bandwidth):
    return np.mean(points, axis=0)

import numpy as np
from collections import defaultdict 

def bin_points(X, bin_size, min_bin_freq):
    bin_sizes = defaultdict(int)
    for point in X:
        binned_point = np.cast[np.int32](point / bin_size)
        bin_sizes[tuple(binned_point)] += 1

    bin_seeds = np.array([point for point, freq in bin_sizes.iteritems() if freq >= min_bin_freq], dtype=np.float32)
    bin_seeds = bin_seeds * bin_size
    return bin_seeds

from numpy import genfromtxt
from sklearn.cluster import get_bin_seeds

from sklearn.datasets.samples_generator import make_blobs

from sklearn.cluster import MeanShift, estimate_bandwidth

X = genfromtxt('results.csv', delimiter=',')


seeds = get_bin_seeds(X, 1, 1)
print seeds
print mean_shift(X, 0.01, seeds, gaussian_kernel_update)



