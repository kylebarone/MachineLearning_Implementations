import numpy as np


class Regression(object):

    def __init__(self):
        pass

    def rmse(self, pred, label):  # [5pts]
        '''
        This is the root mean square error.
        Args:
            pred: numpy array of length N * 1, the prediction of labels
            label: numpy array of length N * 1, the ground truth of labels
        Return:
            a float value
        '''

        y = label - pred
        return pow(np.dot(y.T, y) / pred.size, 1/2)

    def construct_polynomial_feats(self, x, degree):  # [5pts]
        """
        Args:
            x: numpy array of length N, the 1-D observations
            degree: the max polynomial degree
        Return:
            feat: numpy array of shape Nx(degree+1), remember to include
            the bias term. feat is in the format of:
            [[1.0, x1, x1^2, x1^3, ....,],
             [1.0, x2, x2^2, x2^3, ....,],
             ......0
            ]
        """
        n = x.size
        d = degree + 1
        ds = np.arange(d)
        x_new = x.reshape(n, 1) * np.ones((n, d))
        return np.power(x_new, ds)

    def predict(self, xtest, weight):  # [5pts]
        """
        Args:
            xtest: NxD numpy array, where N is number
                   of instances and D is the dimensionality of each
                   instance
            weight: Dx1 numpy array, the weights of linear regression model
        Return:
            prediction: Nx1 numpy array, the predicted labels
        """

        return np.matmul(xtest, weight)

    # =================
    # LINEAR REGRESSION
    # Hints: in the fit function, use close form solution of the linear regression to get weights.
    # For inverse, you can use numpy linear algebra function
    # For the predict, you need to use linear combination of data points and their weights (y = theta0*1+theta1*X1+...)

    def linear_fit_closed(self, xtrain, ytrain):  # [5pts]
        """
        Args:
            xtrain: N x D numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: N x 1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """

        xTx = np.matmul(xtrain.T, xtrain)
        xTxInv = np.linalg.pinv(xTx)
        xTy = np.matmul(xtrain.T, ytrain)
        return np.matmul(xTxInv, xTy)

    def linear_fit_GD(self, xtrain, ytrain, epochs=5, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """

        n = xtrain.shape[0]
        weights = np.zeros((xtrain.shape[1], 1))
        for i in range(epochs):
            pred = np.dot(xtrain, weights)
            a = ytrain - pred
            summy = np.dot(xtrain.T, a)
            summy = (learning_rate / n) * summy
            weights = weights + summy
        return weights

    def linear_fit_SGD(self, xtrain, ytrain, epochs=100, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """

        n = xtrain.shape[0]
        d = xtrain.shape[1]
        weights = np.zeros((d, 1))
        rand = np.random.randint(n, size=epochs)
        for i in rand:
            pred = np.dot(xtrain[i, :], weights)
            a = ytrain[i] - pred
            summy = np.dot(xtrain[i, :].T.reshape((d, 1)), a.reshape((1, 1)))
            summy = (learning_rate) * summy
            weights = weights + summy
        return weights

    # =================
    # RIDGE REGRESSION

    def ridge_fit_closed(self, xtrain, ytrain, c_lambda):  # [5pts]
        """
        Args:
            xtrain: N x D numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: N x 1 numpy array, the true labels
            c_lambda: floating number
        Return:
            weight: Dx1 numpy array, the weights of ridge regression model
        """

        xTx = np.matmul(xtrain.T, xtrain)
        xTx = xTx + np.eye(xTx.shape[0], xTx.shape[1]) * c_lambda
        xTxInv = np.linalg.inv(xTx)
        xTy = np.matmul(xtrain.T, ytrain)
        return np.matmul(xTxInv, xTy)

    def ridge_fit_GD(self, xtrain, ytrain, c_lambda, epochs=500, learning_rate=1e-7):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
            c_lambda: floating number
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """

        n = xtrain.shape[0]
        weights = np.zeros((xtrain.shape[1], 1))
        for i in range(epochs):
            # print("weights pre: ", weights)
            pred = np.dot(xtrain, weights)
            a = ytrain - pred
            summy = np.dot(xtrain.T, a)
            # print("summy: ", summy, " regularlization factor: ", 2*c_lambda*weights)
            summy = (learning_rate / n) * (summy + 2 * c_lambda * weights)
            weights = weights + summy
        return weights

    def ridge_fit_SGD(self, xtrain, ytrain, c_lambda, epochs=100, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """

        n = xtrain.shape[0]
        d = xtrain.shape[1]
        weights = np.zeros((d, 1))
        rand = np.random.randint(n, size=epochs)
        for i in rand:
            pred = np.dot(xtrain[i, :], weights)
            a = ytrain[i] - pred
            summy = np.dot(xtrain[i, :].T.reshape((d, 1)), a.reshape((1, 1)))
            summy = (learning_rate) * (summy + 2 * c_lambda * weights)
            weights = weights + summy
        return weights

    def ridge_cross_validation(self, X, y, kfold=10, c_lambda=100):  # [5 pts]
        """
        Args:
            X : NxD numpy array, where N is the number of instances and D is the dimensionality of each instance
            y : Nx1 numpy array, true labels
            kfold: Number of folds you should take while implementing cross validation.
            c_lambda: Value of regularization constant
        Returns:
            meanErrors: Float average rmse error
        Hint: np.concatenate might be helpful.
        Look at 3.5 to see how this function is being used.
        # For cross validation, use 10-fold method and only use it for your training data (you already have the train_indices to get training data).
        # For the training data, split them in 10 folds which means that use 10 percent of training data for test and 90 percent for training.
        """
        n = y.size
        x_split = np.split(X, kfold)
        y_split = np.split(y, kfold)
        # print(y_split)
        rsmeTot = 0
        for i in range(kfold):
            x_test = x_split[i]
            y_test = y_split[i]
            if i + 1 < len(x_split):
                x_train = np.concatenate(x_split[:i] + x_split[i + 1:])
                y_train = np.concatenate(y_split[:i] + y_split[i + 1:])
            else:
                x_train = np.concatenate(x_split[:i])
                y_train = np.concatenate(y_split[:i])
            w_i = self.ridge_fit_closed(x_train, y_train, c_lambda)
            predi = self.predict(x_test, w_i)
            rsmei = self.rmse(predi, y_test)
            rsmeTot += rsmei
        return rsmeTot / kfold