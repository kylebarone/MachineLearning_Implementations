
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm
# Load image
import imageio


# Set random seed so output is all same
np.random.seed(1)


class KMeans(object):

    def __init__(self):  # No need to implement
        pass

    def pairwise_dist(self, x, y):  # [5 pts]
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
                dist: N x M array, where dist2[i, j] is the euclidean distance between 
                x[i, :] and y[j, :]
                """
        arr = x[:, None, :] - y[None, :, :]
        return np.linalg.norm(arr, axis=-1)

    def _init_centers(self, points, K, **kwargs):  # [5 pts]
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            kwargs: any additional arguments you want
        Return:
            centers: K x D numpy array, the centers. 
        """
        return np.random.random((K, points.shape[1])) * points[:K, :]

    def _update_assignment(self, centers, points):  # [10 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            points: NxD numpy array, the observations
        Return:
            cluster_idx: numpy array of length N, the cluster assignment for each point

        Hint: You could call pairwise_dist() function.
        """
        return np.argmin(self.pairwise_dist(points, centers), axis=1)

    def _update_centers(self, old_centers, cluster_idx, points):  # [10 pts]
        """
        Args:
            old_centers: old centers KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            centers: new centers, K x D numpy array, where K is the number of clusters, and D is the dimension.
        """

        n = points.shape[0]
        k = old_centers.shape[0]
        d = points.shape[1]
        cluster_asn = np.zeros((k, n, d))
        cluster_asn[cluster_idx[:], range(n)] = np.ones((d,))
        b = cluster_asn * points
        c = np.sum(b, axis=1)
        dummy = np.zeros((n, k))
        dummy[range(n), cluster_idx[:]] = 1
        count = np.sum(dummy, axis=0).reshape(k, 1)
        count = np.where(count == 0, 1, count)
        final = c / count
        return final

    def _get_loss(self, centers, cluster_idx, points):  # [5 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            loss: a single float number, which is the objective function of KMeans. 
        """
        pairwise = self.pairwise_dist(points, centers)
        dummy = np.zeros(pairwise.shape)
        dummy[range(pairwise.shape[0]), cluster_idx[:]] = 1
        dummy = (dummy * pairwise)
        dummy = np.sum(dummy, axis=1)
        return dummy.dot(dummy.T)

    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, verbose=False, **kwargs):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            verbose: boolean to set whether method should print loss (Hint: helpful for debugging)
            kwargs: any additional arguments you want
        Return:
            cluster assignments: Nx1 int numpy array
            cluster centers: K x D numpy array, the centers
            loss: final loss value of the objective function of KMeans
        """
        centers = self._init_centers(points, K, **kwargs)
        for it in range(max_iters):
            cluster_idx = self._update_assignment(centers, points)
            centers = self._update_centers(centers, cluster_idx, points)
            loss = self._get_loss(centers, cluster_idx, points)
            K = centers.shape[0]
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            if verbose:
                print('iter %d, loss: %.4f' % (it, loss))
        return cluster_idx, centers, loss

    def find_optimal_num_clusters(self, data, max_K=15):  # [10 pts]
        """Plots loss values for different number of clusters in K-Means

        Args:
            image: input image of shape(H, W, 3)
            max_K: number of clusters
        Return:
            None (plot loss values against number of clusters)
        """

        final = []
        for k in range(1, max_K + 1):
            call = self.__call__(data, k)
            if call is not None: final.append(call[2])
        plt.plot(range(1, 16), final)
        plt.show()
        return final


def intra_cluster_dist(cluster_idx, data, labels):  # [4 pts]
    """
    Calculates the average distance from a point to other points within the same cluster

    Args:
        cluster_idx: the cluster index (label) for which we want to find the intra cluster distance
        data: NxD numpy array, where N is # points and D is the dimensionality
        labels: 1D array of length N where each number indicates of cluster assignement for that point
    Return:
        intra_dist_cluster: 1D array where the i_th entry denotes the average distance from point i 
                            in cluster denoted by cluster_idx to other points within the same cluster
    """
    clstr_data = data[labels == cluster_idx, :]
    arr = clstr_data[:, None, :] - clstr_data[None, :, :]
    dist_arr = np.linalg.norm(arr, axis=-1)
    sum_arr = np.sum(dist_arr, axis=1)
    avg = sum_arr / ((sum_arr.shape[0] - 1) if (sum_arr.shape[0] - 1) != 0 else 1)
    return avg


def inter_cluster_dist(cluster_idx, data, labels):  # [4 pts]
    """
    Calculates the average distance from one cluster to the nearest cluster
    Args:
        cluster_idx: the cluster index (label) for which we want to find the intra cluster distance
        data: NxD numpy array, where N is # points and D is the dimensionality
        labels: 1D array of length N where each number indicates of cluster assignement for that point
    Return:
        inter_dist_cluster: 1D array where the i-th entry denotes the average distance from point i in cluster
                            denoted by cluster_idx to the nearest neighboring cluster
    """
    k = np.unique(labels).size
    n_idx = np.unique(labels, return_counts=True)[1][cluster_idx]
    clstr_arr = []
    for i in range(k):
        clstr_arr.append(data[labels == i, :])
    iShouldReallySpreadThisProjOut = np.zeros((k - 1, n_idx))
    for i, cluster in enumerate(clstr_arr):
        if i == cluster_idx: continue
        arr = clstr_arr[cluster_idx][:, None, :] - clstr_arr[i][None, :, :]
        dist_arr = np.linalg.norm(arr, axis=-1)
        sum_arr = np.sum(dist_arr, axis=1)
        avg = sum_arr / (clstr_arr[i].shape[0])
        iShouldReallySpreadThisProjOut[i if i < k - 1 else cluster_idx] = avg
    iShouldReallySpreadThisProjOut = np.min(iShouldReallySpreadThisProjOut.T, axis=1)
    return iShouldReallySpreadThisProjOut


def silhouette_coefficient(data, labels):  # [2 pts]
    """
    Finds the silhouette coefficient of the current cluster assignment

    Args:
        data: NxD numpy array, where N is # points and D is the dimensionality
        labels: 1D array of length N where each number indicates of cluster assignement for that point
    Return:
        silhouette_coefficient: Silhouette coefficient of the current cluster assignment
    """
    n = labels.size
    k = np.unique(labels).size
    si = 0
    for i in range(k):
        mew_out = inter_cluster_dist(i, data, labels)
        mew_in = intra_cluster_dist(i, data, labels)
        s = np.sum((mew_out - mew_in)/ np.maximum(mew_in, mew_out))
        si += s
    return si/n