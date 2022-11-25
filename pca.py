import numpy as np


class PCA(object):

    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X):
        """
        Decompose dataset into principal components.
        You may use your SVD function from the previous part in your implementation or numpy.linalg.svd function.

        Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
        corresponding values from PCA.

        Args:
            X: N*D array corresponding to a dataset
        Return:
            None
        """
        X_hat = X - np.mean(X, axis=0).reshape(1, X.shape[1])
        self.U, self.S, self.V = np.linalg.svd(X_hat, full_matrices=False)

    def transform(self, data, K=2):
        """
        Transform data to reduce the number of features such that final data has given number of columns

        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: N*D array corresponding to a dataset
            K: Int value for number of columns to be kept
        Return:
            X_new: N*K array corresponding to data obtained by applying PCA on data
        """
        n = data.shape[0]
        s = np.dot(self.S, self.S) / n
        # print("k: ", K, " data :", data.shape, " self.V.T: ", self.V.T.shape)
        v = np.matmul(data, self.V.T)[..., :K]
        return v

    def transform_rv(self, data, retained_variance=0.99):  # 3 pts
        """
        Transform data to reduce the number of features such that a given variance is retained

        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: N*D array corresponding to a dataset
            retained_variance: Float value for amount of variance to be retained
        Return:
            X_new: N*K array corresponding to data obtained by applying PCA on data
        """

        s_sqr = np.dot(self.S, self.S)
        s_i = self.S * self.S
        s_i_cum = np.cumsum(s_i)
        s_var = s_i_cum / s_sqr
        k = np.where(s_var == retained_variance)[0]
        if (len(k) == 0):
            k = np.argmin(s_var - retained_variance * ((s_var - retained_variance) > 0))
        else:
            k = k[0]
        k += 1
        # print(k)
        return self.transform(data, k)

    def get_V(self):
        """ Getter function for value of V """

        return self.V

    def get_V(self):
        """ Getter function for value of V """

        return self.V