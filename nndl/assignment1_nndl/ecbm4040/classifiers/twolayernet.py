from __future__ import print_function

import numpy as np

from ecbm4040.layer_funcs import *
from ecbm4040.layer_utils import *

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:
    input -> DenseLayer -> AffineLayer -> softmax loss -> output
    Or more detailed,
    input -> affine transform -> ReLU -> affine transform -> softmax -> output

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_dim=3072, hidden_dim=200, num_classes=10, reg=0.0, weight_scale=1e-2):
        """
        Inputs:
        - reg: (float) L2 regularization
        - weight_scale: (float) for layer weight initialization
        """
        self.layer1 = DenseLayer(input_dim, hidden_dim, weight_scale=weight_scale)
        self.layer2 = AffineLayer(hidden_dim, num_classes, weight_scale=weight_scale)
        self.reg = reg
        self.momentum = 0.5
        params = self.layer1.params + self.layer2.params
        self.velocity = []
        for j in range(len(params)):
            self.velocity.append(np.zeros_like(params[j]))

        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_alpha = 0.001
        self.adam_eps = 1e-8
        self.adam_m = self.velocity
        self.adam_r = self.velocity
        self.adam_count = 0

    def loss(self, X, y):
        """
        Calculate the cross-entropy loss and then use backpropogation
        to get gradients wst W,b in each layer.
        
        Inputs:
        - X: input data
        - y: ground truth
        
        Return loss value(float)
        """
        loss = 0.0
        reg = self.reg
        ###################################################
        #TODO: Feedforward                                #
        ###################################################

        a1 = self.layer1.feedforward(X)
        a2 = self.layer2.feedforward(a1)
        loss, da2 = softmax_loss(a2, y)


        ###################################################
        #TODO: Backpropogation, here is just one dense    #
        #layer, it should be pretty easy                  #
        ###################################################
        da1 = self.layer2.backward(da2)
        dx = self.layer1.backward(da1)
        
        ###################################################
        #              END OF YOUR CODE                   #
        ###################################################
        # Add L2 regularization
        square_weights = np.sum(self.layer1.params[0]**2) + np.sum(self.layer2.params[0]**2)
        loss += 0.5*self.reg*square_weights
        return loss

    def step(self, learning_rate=1e-5):
        """
        Use SGD to implement a single-step update to each weight and bias.
        """
        # creates new lists with all parameters and gradients
        layer1, layer2 = self.layer1, self.layer2
        params = layer1.params + layer2.params
        grads = layer1.gradients + layer2.gradients
        
        # Add L2 regularization
        reg = self.reg
        grads = [grad + reg*params[i] for i, grad in enumerate(grads)]
        ###################################################
        #TODO: Use SGD or SGD with momentum to update     #
        #variables in layer1 and layer2                   #
        ###################################################

        #for k in range(len(params)):
        #    params[k] -= learning_rate * grads[k]
        #print(len(params))
        #print(superpo)
        #for k in range(len(params)):
        #    self.velocity[k] = (self.momentum * self.velocity[k]) - learning_rate * grads[k]
        #    params[k] += self.velocity[k]

        #for k in range(len(params)):
        #    self.velocity[k] = (self.momentum * self.velocity[k]) + learning_rate * grads[k]
        #    params[k] -= self.velocity[k]
        #self.velocity = (self.momentum * self.velocity) - learning_rate * grads
        #params += self.velocity

        
        self.adam_count += 1
        for k in range(len(params)):
            self.adam_m[k] = (self.adam_beta1 * self.adam_m[k]) + (1-self.adam_beta1) * grads[k]
            #self.adam_r[k] = np.maximum((self.adam_beta2 * self.adam_r[k]), np.abs(grads[k]))
            self.adam_r[k] = (self.adam_beta2 * self.adam_r[k]) + (1-self.adam_beta2) * grads[k]**2
            m_k_hat = self.adam_m[k]/(1-self.adam_beta1**self.adam_count)
            r_k_hat = self.adam_r[k]/(1-self.adam_beta2**self.adam_count)
            r_k_hat = np.abs(r_k_hat)
            #temp = (self.adam_alpha* m_k_hat)/(np.sqrt(r_k_hat) + self.adam_eps)
            #print(r_k_hat)
            params[k] -= (self.adam_alpha* m_k_hat)/(np.sqrt(r_k_hat) + self.adam_eps)
        
        #self.adam_m = (self.adam_beta1 * self.adam_m) + (1-self.adam_beta1) * grads
        #self.adam_r = (self.adam_beta2 * self.adam_r) + (1-self.adam_beta2) * grads * grads
        #m_k_hat = self.adam_m/(1-self.adam_beta1**iterat)
        #print(m_k_hat.shape)
        #print(self.velocity)
        #print(params.shape)
        #print(self.layer1.params.shape)
        #r_k_hat = np.sqrt(self.adam_r/(1-self.adam_beta2**iterat))
        #print(superpo)
        #params += (learning_rate* m_k_hat)/(r_k_hat + self.adam_eps)

        #print(np.allclose(temp[0], params[0]))
        #print(superpo)
        #params -= learning_rate * grads

        #params = np.subtract(params, learning_rate * grads)

        
        ###################################################
        #              END OF YOUR CODE                   #
        ###################################################
   
        # update parameters in layers
        layer1.update_layer(params[0:2])
        layer2.update_layer(params[2:4])
        

    def predict(self, X):
        """
        Return the label prediction of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        
        Returns: 
        - predictions: (int) an array of length N
        """
        predictions = None
        layer1, layer2 = self.layer1, self.layer2
        #######################################################
        #TODO: Remember to use functions in class SoftmaxLayer#
        #######################################################
        a1 = self.layer1.feedforward(X)
        a2 = self.layer2.feedforward(a1)
        a2 -= np.amax(a2, axis=1, keepdims=True)
        exp_a2 = np.exp(a2)
        sigma_a2 = exp_a2/ np.sum(exp_a2, axis=1, keepdims=True)
        predictions = np.argmax(sigma_a2, axis=1)
        
        #######################################################
        #                 END OF YOUR CODE                    #
        #######################################################
        
        return predictions
    
    def check_accuracy(self, X, y):
        """
        Return the classification accuracy of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        - y: (int) an array of length N. ground truth label 
        Returns: 
        - acc: (float) between 0 and 1
        """
        y_pred = self.predict(X)
        acc = np.mean(np.equal(y, y_pred))
        
        return acc
    
    def save_model(self):
        """
        Save model's parameters, including two layer's W and b and reg
        """
        return [self.layer1.params, self.layer2.params, self.reg]
    
    def update_model(self, new_params):
        """
        Update layers and reg with new parameters
        """
        layer1_params, layer2_params, reg = new_params
        
        self.layer1.update_layer(layer1_params)
        self.layer2.update_layer(layer2_params)
        self.reg = reg
        
        
        


