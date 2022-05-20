from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        all_dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
          #shape of matrix W (in_dim, out_dim)
          #For W1 this will be d  X H1. 
          #For W2 this will be H1 X H2.
          #...
          #Fow WL this will be H_(L-1) X C
          in_dim = all_dims[i]
          out_dim = all_dims[i+1]

          #create objects
          self.params['W%i'%(i+1)] = weight_scale*np.random.randn(in_dim,out_dim)
          self.params['b%i'%(i+1)] = np.zeros(out_dim)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        cache = {} #store intermediate values needed for backprop
        N = X.shape[0]

        #'data' is the input to the each layer
        data = X.reshape(N,-1)
        for i in range(self.num_layers-1): #the last layer will be treated differently, no ReLu
          W = self.params['W%i'%(i+1)]
          b = self.params['b%i'%(i+1)]
          cache['indata_layer_%i'%(i+1)] = data

          #do Relu(h*W + b) to compute input for next layer (until finish)
          data = np.maximum(0, np.matmul(data,W) + b)

        #output from the last dense layer without ReLu
        WL = self.params['W%i'%self.num_layers]
        bL = self.params['b%i'%self.num_layers]
        cache['indata_layer_%i'%self.num_layers] = data        
        scores = np.matmul(data,WL) + bL

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        ####################################################################
        # FIRST DO DATA LOSS AND GRADIENTS. 
        # ADD REGULARIZATION CONTRIBUTIONS TO BOTH AT THE END
        ####################################################################

        #to avoid numerical problems with softmax, subtract out the max of the score for each row
        max_rowscore = np.max(scores, axis=1)
        scores = (scores.transpose() - max_rowscore).transpose()
        scores_exp = np.exp(scores)
        scores_exp_sum = np.sum(scores_exp, axis=1)

        #Due to normalization, p_ik can be interpreted as probabilities
        p_ik = (scores_exp.transpose()/scores_exp_sum).transpose() 
        L_i = -np.log(p_ik[np.arange(N),y])
        loss = 1./N*np.sum(L_i)        

        #now get ready for the backward pass

        #Compute gradient of softmax with respect to the scores
        index_mat = np.zeros_like(p_ik)
        index_mat[np.arange(N),y] = 1 #sparse mat would be more efficient ofc
        grad_soft = p_ik - index_mat
        grad_soft *= (1./N) #also need to divide by N, just like loss

        #Compute the gradient with respect to WL and bL
        i_n = np.ones((N,1))
        grads['b%i'%self.num_layers] = np.matmul(i_n.transpose(), grad_soft).flatten()

        data = cache['indata_layer_%i'%self.num_layers]
        grads['W%i'%self.num_layers] = np.matmul(data.transpose(),grad_soft)

        #compute the gradient with respect to data path of the graph
        #this will be used as input gradient for the next layer
        #initialize 'grad_prev' with this.
        WL = self.params['W%i'%self.num_layers]
        grad_prev = np.matmul(grad_soft,WL.transpose())

        #initialize data_prev, the output of the ReLu from the layer before
        data_prev = data

        #Now compute the gradient for each of the hidden layers 
        for i in reversed(range(self.num_layers-1)):
          #grab the input data from the forward pass stored for this layer
          data = cache['indata_layer_%i'%(i+1)]

          #grab the weights
          W = self.params['W%i'%(i+1)]

          #gradient with respect to the input of the ReLu
          #the derivation says this is the gradient with respect to the output of the relu,
          #with all the entries where the input is <= 0 set to 0
          #This is the same mask as where the output of the ReLu is 0 (with the behavior at 0.0 undefined)
          #By using the output of the ReLu (i.e. data_prev) we do not have to cache the input to the ReLu each iter
          grad_prev[data_prev==0] = 0.0

          #Same structure as gradients for bL and WL
          grads['b%i'%(i+1)] = np.matmul(i_n.transpose(), grad_prev).flatten()
          grads['W%i'%(i+1)] = np.matmul(data.transpose(),grad_prev)

          #prepare for next iterations (grad_prev and data_prev)
          grad_prev = np.matmul(grad_prev,W.transpose())
          data_prev = data

        ####################################################################
        # Add regularization to loss and to gradients. Include the 0.5
        ####################################################################
        reg = self.reg
        for i in reversed(range(self.num_layers)):                
          W = self.params['W%i'%(i+1)]
          b = self.params['b%i'%(i+1)]

          loss += 0.5*reg*(np.sum(W**2) + np.sum(b**2))
          grads['W%i'%(i+1)] += reg*W
          grads['b%i'%(i+1)] += reg*b

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
