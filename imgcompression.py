from matplotlib import pyplot as plt
import numpy as np


class ImgCompression(object):
    def __init__(self):
        pass

    def svd(self, X):  # [5pts]
        """
        Do SVD. You could use numpy SVD.
        Your function should be able to handle black and white
        images (N*D arrays) as well as color images (N*D*3 arrays)
        In the image compression, we assume that each colum of the image is a feature. Image is the matrix X.
        Args:
            X: N * D array corresponding to an image (N * D * 3 if color image)
        Return:
            U: N * N for black and white images / N * N * 3 for color images
            S: min(N, D) * 1 for black and white images / min(N, D) * 3 for color images
            V: D * D for black and white images / D * D * 3 for color images
        """
        N = X.shape[0]
        D = X.shape[1]
        Sdim = D if D < N else N
        if len(X.shape) == 3:
            U = np.zeros((N, N, 3))
            S = np.zeros((Sdim, 3))
            V = np.zeros((D, D, 3))
            for i in range(3):
                Ui, Si, Vi = np.linalg.svd(X[..., i], full_matrices=True)
                U[..., i] = Ui
                S[..., i] = Si
                V[..., i] = Vi
        else:
            U, S, V = np.linalg.svd(X, full_matrices=True)
        return U, S, V


    def rebuild_svd(self, U, S, V, k): # [5pts]
        """
        Rebuild SVD by k componments.
        Args:
            U: N*N (*3 for color images)
            S: min(N, D)*1 (*3 for color images)
            V: D*D (*3 for color images)
            k: int corresponding to number of components
        Return:
            Xrebuild: N*D array of reconstructed image (N*D*3 if color image)

        Hint: numpy.matmul may be helpful for reconstructing color images
        """

        N = U.shape[0]
        D = V.shape[1]
        #print("D:",D)
        Sdim = D if D < N else N
        if len(U.shape) == 3:
            image = np.zeros((N, D,  3))
            for i in range(3):
                Uk = np.zeros((N,N))
                #print(U[:,:k,i].shape)
                Uk[...,:k] = U[:,:k,i]
                Sk = np.zeros((N, D))
                Sk[:k,:k] = np.diag(S[:k, i])
                Vk = np.zeros((D,D))
                Vk[:k,...] = V[:k,:,i]
                #print("Uk.shape:", Uk.shape, " Sk.shape:", Sk.shape, " i", i)
                us = np.matmul(Uk, Sk)
                svdi = np.matmul(us, Vk)
                image[...,i] = svdi
        else:
            image = np.zeros((N,D))
            Uk = np.zeros(U.shape)
            Uk[...,:k] = U[:,:k]
            Sk = np.zeros((N, D))
            Sk[:k,:k] = np.diag(S[:k])
            Vk = np.zeros(V.shape)
            Vk[:k,...] = V[:k,:]
            us = np.matmul(Uk, Sk)
            svdi = np.matmul(us, Vk)
            image[...] = svdi
        return image

    def compression_ratio(self, X, k): # [5pts]
        """
        Compute compression of an image: (num stored values in original)/(num stored values in compressed)
        Args:
            X: N * D array corresponding to an image (N * D * 3 if color image)
            k: int corresponding to number of components
        Return:
            compression_ratio: float of proportion of storage used by compressed image
        """
        N = X.shape[0]
        D = X.shape[1]
        return (k * (1 + N + D))/(N*D)

    def recovered_variance_proportion(self, S, k): # [5pts]
        """
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation

        Args:
           S: min(N, D)*1 (*3 for color images) of singular values for the image
           k: int, rank of approximation
        Return:
           recovered_var: int (array of 3 ints for color image) corresponding to proportion of recovered variance
        """
        if len(S.shape) == 2:
            arr = []
            for i in range(3):
                # print(S[:k,i].shape)
                m = np.dot(S[:k, i].T, S[:k, i]) / np.dot(S[:, i].T, S[:, i])
                # print(m, "l")
                arr.append(m)
        else:
            # print('d')
            arr = (np.dot(S[:k].T, S[:k])) / (np.dot(S.T, S))
        return arr