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



class GMM(object):
    def __init__(self, X, K, max_iters = 100): # No need to change
        """
        Args: 
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters
        
        self.N = self.points.shape[0]        #number of observations
        self.D = self.points.shape[1]        #number of features
        self.K = K                           #number of components/clusters

    #Helper function for you to implement
    def softmax(self, logit): # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        """
        n = logit.shape[0]
        log_max = np.amax(logit, axis=1).reshape((n, 1))
        logit_minus = logit - log_max
        log_exp = np.exp(logit_minus)
        return log_exp / np.sum(log_exp, axis=1).reshape((n, 1))

    def logsumexp(self, logit): # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        """
        n = logit.shape[0]
        logit_exp = np.exp((logit - np.amax(logit, axis=1).reshape((n, 1))))
        log_exp_sum = np.sum(logit_exp, axis=1)
        final = np.log(log_exp_sum) +  np.amax(logit, axis=1)
        return final.reshape((final.size, 1))
        #return final

    #for undergraduate student
    def normalPDF(self, logit, mu_i, sigma_i): #[5pts]
        """
        Args:
            logit: N x D numpy array
            mu_i: 1xD numpy array (or array of lenth D), the center for the ith gaussian.
            sigma_i: 1xDxD 3-D numpy array (or DxD 2-D numpy array), the covariance matrix of the ith gaussian.
        Return:
            pdf: 1xN numpy array (or array of length N), the probability distribution of N data for the ith gaussian

        Hint:
            np.diagonal() should be handy.
        """

        x_sub_mu = logit - mu_i
        x_sub_mu_sqr = x_sub_mu * x_sub_mu
        sigma_ii = 2 * np.diagonal(sigma_i)
        right_sig = (np.ones(sigma_ii.shape) * -1) / sigma_ii
        right_paran = right_sig.reshape((1, right_sig.size)) * x_sub_mu_sqr
        right_exp = np.exp(right_paran)
        sig_dig = 2 * np.pi * np.diagonal(sigma_i)
        left_paran = np.ones((sig_dig.size,)) / np.sqrt(sig_dig)
        left = left_paran.reshape((1, left_paran.size))
        bruh =  left * right_exp
        final = np.prod(bruh, axis=1)
        return final

    #for grad students
    def multinormalPDF(self, logits, mu_i, sigma_i):  #[5pts]
        """
        Args: 
            logit: N x D numpy array
            mu_i: 1xD numpy array (or array of lenth D), the center for the ith gaussian.
            sigma_i: 1xDxD 3-D numpy array (or DxD 2-D numpy array), the covariance matrix of the ith gaussian.  
        Return:
            pdf: 1xN numpy array (or array of length N), the probability distribution of N data for the ith gaussian
         
        Hint: 
            np.linalg.det() and np.linalg.inv() should be handy.
        """
        raise NotImplementedError
    
    
    def _init_components(self, **kwargs): # [5pts]
        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. 
                You will have KxDxD numpy array for full covariance matrix case
        """
        comps = np.ones((self.K,)) / self.K

        mu = self.points[np.random.randint(self.N-1, size=self.K)]
        sigma = np.zeros((self.K, self.D, self.D))
        for i in range(self.K):
            sigma[i] = np.eye(self.D, self.D)
        return comps, mu, sigma

    
    def _ll_joint(self, pi, mu, sigma, **kwargs): # [10 pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            
        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
        """

        log_comp = np.log(pi + 1e-32)
        bruh = np.ones((self.K, self.N))
        for j in range(self.K):
            bruh[j] = np.log(self.normalPDF(self.points, mu[j], sigma[j]) + 1e-32)
        #print(bruh.shape)
        final = bruh.T + log_comp.reshape((1, log_comp.size))
        return final

    def _E_step(self, pi, mu, sigma, **kwargs): # [5pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            
        Hint: 
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above. 
        """

        blurp = np.zeros((self.K, self.N))
        for i in range(self.K):
            blurp[i] = pi[i] * self.normalPDF(self.points, mu[i], sigma[i])
        blurpy = blurp.T
        blurpy_sum = np.sum(blurpy, axis=1)
        blurpies = blurpy / blurpy_sum.reshape(blurpy_sum.size, 1)
        return blurpies

    def _M_step(self, gamma, **kwargs): # [10pts]
        """
        Args:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            
        Hint:  
            There are formulas in the slide and in the above description box.
        """

        top = np.ones((self.K, self.D))
        for i in range(self.K):
            t_mu = gamma.T[i].reshape((self.N, 1)) * self.points
            top[i] = np.sum(t_mu, axis=0)
        mu = top / np.sum(gamma, axis=0).reshape((self.K, 1))

        sigma = np.zeros((self.K, self.D, self.D))
        for i in range(self.K):
            x_sub_mu = self.points - mu[i].reshape((1, self.D))
            right = (gamma.T[i] * x_sub_mu.T).dot(x_sub_mu)
            sigma[i] = np.diag(np.diagonal(right / np.sum(gamma.T[i])))
        pi = np.sum(gamma, axis=0) / self.N

        return pi, mu, sigma
    
    
    def __call__(self, abs_tol=1e-16, rel_tol=1e-16, **kwargs): # No need to change
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want
        
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)       
        
        Hint: 
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters. 
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))
        
        for it in pbar:
            # E-step
            gamma = self._E_step(pi, mu, sigma)
            
            # M-step
            pi, mu, sigma = self._M_step(gamma)
            
            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)
