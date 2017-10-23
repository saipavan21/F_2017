from builtins import range
from builtins import object
import numpy as np

from ecbm4040.layer_funcs import *
from ecbm4040.layer_utils import *

class MLP(object):
    """
    MLP with an arbitrary number of dense hidden layers,
    and a softmax loss function. For a network with L layers,
    the architecture will be

    input >> DenseLayer x (L - 1) >> AffineLayer >> softmax_loss >> output

    Here "x (L - 1)" indicate to repeat L - 1 times. 
    """
    def __init__(self, input_dim=3072, hidden_dims=[200,200], num_classes=10, reg=0.0, weight_scale=1e-2):
        """
        Inputs:
        - reg: (float) L2 regularization
        - weight_scale: (float) for layer weight initialization
        """
        self.num_layers = len(hidden_dims) + 1
        self.reg = reg
        
        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims)-1):
            layers.append(DenseLayer(input_dim=dims[i], output_dim=dims[i+1], weight_scale=weight_scale))
        layers.append(AffineLayer(input_dim=dims[-1], output_dim=num_classes, weight_scale=weight_scale))
        
        self.layers = layers
        self.momentum = 0.5

        self.velocity = []
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i].params)):
                self.velocity.append(np.zeros_like(self.layers[i].params[j]))
            
        self.adam_m = self.velocity
        self.adam_r = self.velocity
        self.adam_count = 0
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_alpha = 0.001
        self.adam_eps = 1e-8

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
        num_layers = self.num_layers
        layers = self.layers
        ###################################################
        #TODO: Feedforward                                #
        ###################################################

        layer_input = X
        for layer in self.layers:
            layer_ouput = layer.feedforward(layer_input)
            layer_input = layer_ouput

        loss, layer_dinput = softmax_loss(layer_input, y)


        ###################################################
        #TODO: Backpropogation                            #
        ###################################################
        
        for rev_layer in list(reversed(self.layers)):
            rev_layer_output = rev_layer.backward(layer_dinput)
            layer_dinput = rev_layer_output

        
        ###################################################
        # TODO: Add L2 regularization                     #
        ###################################################
        square_weights = 0.0
        for layer in self.layers:
            square_weights += np.sum(layer.params[0]**2)

        loss += 0.5 * self.reg * square_weights
        
        ###################################################
        #              END OF YOUR CODE                   #
        ###################################################
        
        return loss

    def step(self, learning_rate=1e-5):
        """
        Use SGD to implement a single-step update to each weight and bias.
        """
        ###################################################
        #TODO: Use SGD or SGD with momentum to update     #
        #variables in layers                              #
        ###################################################
        num_layers = len(self.layers)
        params  = []
        grads = []
        for i in range(len(self.layers)):
            params += self.layers[i].params
            grads += self.layers[i].gradients

        for i in range(len(grads)):
            grads[i] += self.reg * params[i] 

        
        for i in range(len(params)):
            params[i] -= learning_rate * grads[i]

        #for i in range(len(params)):
        #    self.velocity[i] += (self.momentum * self.velocity[i]) + (learning_rate * grads[i])
        #    params[i] -= self.velocity[i]


        """
        self.adam_count += 1
        for k in range(len(params)):
            self.adam_m[k] = (self.adam_beta1 * self.adam_m[k]) + (1-self.adam_beta1) * grads[k]
            #self.adam_r[k] = np.maximum((self.adam_beta2 * self.adam_r[k]), np.abs(grads[k]))
            self.adam_r[k] = (self.adam_beta2 * self.adam_r[k]) + (1-self.adam_beta2) * grads[k]**2
            m_k_hat = self.adam_m[k]/(1-self.adam_beta1**self.adam_count)
            r_k_hat = self.adam_r[k]/(1-self.adam_beta2**self.adam_count)
            r_k_hat = np.abs(r_k_hat)
            params[k] -= (self.adam_alpha* m_k_hat)/(np.sqrt(r_k_hat) + self.adam_eps)

        """
        ###################################################
        #              END OF YOUR CODE                   #
        ###################################################
   
        # update parameters in layers
        for i in range(num_layers):
            self.layers[i].update_layer(params[2*i:2*(i+1)])
        

    def predict(self, X):
        """
        Return the label prediction of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        
        Returns: 
        - predictions: (int) an array of length N
        """
        predictions = None
        num_layers = self.num_layers
        layers = self.layers
        #######################################################
        #TODO: Remember to use functions in class SoftmaxLayer#
        #######################################################
        layer_input = X

        for layer in self.layers:
            layer_output = layer.feedforward(layer_input)
            layer_input = layer_output

        layer_output -= np.amax(layer_output, axis=1, keepdims=True)
        exp_layer_output = np.exp(layer_output)
        sigma_layer_output = exp_layer_output/ np.sum(exp_layer_output, axis=1, keepdims=True)
        predictions = np.argmax(sigma_layer_output, axis=1)
        

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
        
        


