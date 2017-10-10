import time
import numpy as np

def pca_naive(X, K):
    """
    PCA -- naive version

    Inputs:
    - X: (float) A numpy array of shape (N, D) where N is the number of samples,
         D is the number of features
    - K: (int) indicates the number of features you are going to keep after
         dimensionality reduction

    Returns a tuple of:
    - P: (float) A numpy array of shape (K, D), representing the top K
         principal components
    - T: (float) A numpy vector of length K, showing the score of each
         component vector
    """

    ###############################################
    #TODO: Implement PCA by extracting eigenvector#
    ###############################################

    X_cov = X.transpose().dot(X)
    X_cov /= (X.shape[0] -1)
    eig_values, eig_vectors = np.linalg.eig(X_cov)
    eiginds = eig_values.argsort()
    sorted_eig_values = eig_values[eiginds[::-1]]
    sorted_eig_vecs = (eig_vectors.transpose()[eiginds[::-1]]).transpose()
    req_dim = sorted_eig_vecs[:, :K]
    #P = X.dot(req_dim)
    #T = np.mean(P, axis=0)
    P = req_dim.transpose()
    T = np.mean(P, axis=1)
    #P = P.dot(req_dim.transpose())

    ###############################################
    #              End of your code               #
    ###############################################
    
    return (P, T)
