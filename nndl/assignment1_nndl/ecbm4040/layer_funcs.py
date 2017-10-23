from builtins import range
import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine function.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: a numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: a numpy array of weights, of shape (D, M)
    - b: a numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    """
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    x_shape = x.shape
    new_shape = [x_shape[0], np.prod(x_shape[1:])]
    reshaped_x = np.reshape(x, new_shape)
    out = reshaped_x.dot(w) + b


    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out


def affine_backward(dout, x, w, b):
    """
    Computes the backward pass of an affine function.

    Inputs:
    - dout: upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: input data, of shape (N, d_1, ... d_k)
      - w: weights, of shape (D, M)
      - b: bias, of shape (M,)

    Returns a tuple of:
    - dx: gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: gradient with respect to w, of shape (D, M)
    - db: gradient with respect to b, of shape (M,)
    """
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    x_shape = x.shape
    new_shape = [x_shape[0], np.prod(x_shape[1:])]
    reshape_x = np.reshape(x, new_shape)

    dx = dout.dot(w.transpose())
    dx = np.reshape(dx, x_shape)
    dw = reshape_x.transpose().dot(dout)
    db = np.sum(dout, axis=0)


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for rectified linear units (ReLUs).

    Input:
    - x: inputs, of any shape

    Returns a tuple of:
    - out: output, of the same shape as x
    """
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################

    out = x.clip(min=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out


def relu_backward(dout, x):
    """
    Computes the backward pass for rectified linear units (ReLUs).

    Input:
    - dout: upstream derivatives, of any shape

    Returns:
    - dx: gradient with respect to x
    """
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    x = x.clip(min=0)
    binary_x = x
    binary_x[x>0] =1
    dx = binary_x * dout

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """
    Softmax loss function, vectorized version.
    y_prediction = argmax(softmax(x))
    
    Inputs:
    - X: (float) a tensor of shape (N, #classes)
    - y: (int) ground truth label, a array of length N

    Returns:
    - loss: the cross-entropy loss
    - dx: gradients wrt input x
    """
    # Initialize the loss.
    loss = 0.0
    dx = np.zeros_like(x)
    #############################################################################
    # TODO: You can use the previous softmax loss function here.                #
    #############################################################################

    num_train = x.shape[0]
    scores = x
    scores -= np.amax(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    sigma_scores = exp_scores/ np.sum(exp_scores, axis=1, keepdims=True)
    #req_sigma_scores = sigma_scores[np.arange(num_train), y]
    loss = np.sum(-scores[np.arange(num_train), y] + np.log(np.sum(exp_scores, axis=1)))
    loss /= num_train

    #computing the gradient

    dx = sigma_scores
    dx[np.arange(num_train), y] -= 1
    dx /= num_train
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dx