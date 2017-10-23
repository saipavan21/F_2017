import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wst W, an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
      scores = X[i].dot(W)
      scores -= np.max(scores)
      exp_scores = np.exp(scores)
      sigma_score = exp_scores/np.sum(exp_scores)
      loss -= np.log(sigma_score[y[i]])
      sigma_score[y[i]] -= 1
      for j in range(num_classes):
        dW[:,j] += (X[i] * sigma_score[j])
    
    loss /= num_train
    loss += reg* np.sum(W*W)
    
    dW /= num_train
    dW += reg * 2 * W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    
    #computing the loss

    num_train = X.shape[0]
    scores = X.dot(W)
    scores -= np.amax(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    sigma_scores = exp_scores/ np.sum(exp_scores, axis=1, keepdims=True)
    req_sigma_scores = sigma_scores[np.arange(num_train), y]
    loss = np.sum(-scores[np.arange(num_train), y] + np.log(np.sum(exp_scores, axis=1)))
    loss /= num_train
    loss += reg * np.sum(W * W)

    #computing the gradient

    Z_gradient = sigma_scores
    Z_gradient[np.arange(num_train), y] -= 1
    dW = X.transpose().dot(Z_gradient)
    dW /= num_train
    dW += reg * 2 * W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
