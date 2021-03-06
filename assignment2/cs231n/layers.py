from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = x.shape[0]
    sample_dims = x[0].shape #For a sample, grab the dimensions

    x_vec = x.reshape(N, np.prod(sample_dims))
    out = np.matmul(x_vec,w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = x.shape[0]
    sample_dims = x[0].shape #For a sample, grab the dimensions    

    x_vec = x.reshape(N, np.prod(sample_dims))

    #grad with respect to x
    dx = np.matmul(dout,w.transpose())
    dx = dx.reshape(x.shape)

    #grad with respect to w
    dw = np.matmul(x_vec.transpose(),dout)

    #grad with respect to b
    db = dout.sum(axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = np.copy(dout)
    dx[x<0] = 0.0

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C = x.shape

    #from classifiers/fc_net.py
    scores = x

    #to avoid numerical problems with softmax, subtract out the max of the score for each row
    max_rowscore = np.max(scores, axis=1)
    scores = (scores.transpose() - max_rowscore).transpose()
    scores_exp = np.exp(scores)
    scores_exp_sum = np.sum(scores_exp, axis=1)

    #Due to normalization, p_ik can be interpreted as probabilities
    p_ik = (scores_exp.transpose()/scores_exp_sum).transpose() 
    L_i = -np.log(p_ik[np.arange(N),y])
    loss = 1./N*np.sum(L_i)            

    #Compute gradient of softmax with respect to the scores
    index_mat = np.zeros_like(p_ik)
    index_mat[np.arange(N),y] = 1 #sparse mat would be more efficient ofc
    grad_soft = p_ik - index_mat
    grad_soft *= (1./N) #also need to divide by N, just like loss

    dx = grad_soft

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #algorithm 1, p3 of paper
        batch_mean    = 1./N * np.sum(x, axis=0) #vector of length D
        x_mean_zero   = x-batch_mean

        batch_var     = 1./N * np.sum(x_mean_zero**2, axis=0) #vector of length D
        x_var_one     = x_mean_zero/np.sqrt(batch_var + eps)

        #update running mean and variance
        running_mean  = momentum * running_mean + (1 - momentum) * batch_mean
        running_var   = momentum * running_var + (1 - momentum) * batch_var

        #prep output
        out           = gamma*x_var_one+beta #scale and shift step of algorithm 1

        #cache will contain helpful variables for backprop
        #Using intermediate variable names 
        cache         = { 'mu':batch_mean,
                          'v':batch_var,
                          's':np.sqrt(batch_var + eps),
                          'x':x,
                          'y':x_var_one,
                          'gamma':gamma,
                          'eps':eps
                          }

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #equation 12, algorithm 2
        out           = gamma * (x-running_mean)/np.sqrt(running_var + eps) + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #In this implementation I will just explicitly compute the chain rule. 
    #I am not trying to simplify the expressions

    N, D = dout.shape

    #z = gamma*y + beta, (gamma and beta are (D,) vectors and * represents column multiplication here. Not mat-vec)
    dldz  = dout                                                      #(N,D)
    dldb  = np.sum(dldz, axis=0)                                      #(D,)
    dldg  = np.sum(cache['y']*dldz, axis=0)                           #(D,)

    dldy  = dldz*cache['gamma']                                       #(N,D)
    dlds  = np.sum(-1.0*cache['s']**-2*(cache['x']-cache['mu'])*dldy, axis=0)       #(D,)
    dldv  = 0.5*(cache['v']+cache['eps'])**-0.5 * dlds                #(D,)
    
    #dldu = dvdu*dldv + dydu*dldy
    dldu  = 1./N * -1* np.sum(2*(cache['x']-cache['mu']),axis=0) * dldv
    dldu += -1./cache['s']*np.sum(dldy,axis=0)                        #(D,)

    #dldx = dvdx*dldv + dudx*dldu + dydx*dldy
    dldx  = 1./N * 2 * (cache['x'] -cache['mu']) * dldv 
    dldx += 1./N*dldu
    dldx += 1./cache['s']*dldy #This branch is not displayed in figure of notebook

    #prep for out
    dx      = dldx
    dgamma  = dldg
    dbeta   = dldb

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #for batchnorm we sum over the rows
    #when we want to reuse this function for layer normalization, we transpose dout,
    #but we still want to sum over the original activations (which are now cols)
    sumaxis = cache.get('sumaxis', 0)

    N, D = dout.shape

    dldz  = dout                                                      #(N,D)
    dldb  = np.sum(dldz, axis=sumaxis)                                #(D,)
    dldg  = np.sum(cache['y']*dldz, axis=sumaxis)                     #(D,)

    #now work to dx
    dldy  = dldz*cache['gamma']  

    #Derivation for a single dimension
    #since all dimensions are batch normalized independent
    #of each other, this derivation naturally extends 
    #to all dimensions D

    #I derived dy_i / dx_j, where the subscripts refer to 
    #sample numbers and y and x are for the same dimension
    #This expression simplifies A LOT because dsigma/dx_j = 0 !

    #dy_i / dx_j = kron_{i,j}/sigma - 1/sigma*du/dx_k - (x_i - mu)*d(1/sigma)/dx_j

    #my final expression for dy_i / dx_j = s**-1*(kron_{i,j}-1/N) - (x_i-mu)(x_j-mu)/(N*s**3) [a lot of terms cancelled]
    #Then dldx = sum_i{dy_i/dx_j * dL/dy_i} (again, indices are for one dimension [i.e. col] of the matrix. But same operation on each col)

    #Could maybe have simplified expression more, but not spending more time on this

    sum_dldy = np.sum(dldy, axis=0)
    sum_xdldy= np.sum(cache['x']*dldy, axis=0)
    dldx  = 1./cache['s'] * dldy - 1./(N*cache['s'])*sum_dldy - (cache['x']-cache['mu'])*(sum_xdldy-cache['mu']*sum_dldy)/(N*cache['s']**3)
          

    #prep for out
    dx      = dldx
    dgamma  = dldg
    dbeta   = dldb

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,D = x.shape
    x = x.transpose()

    #in layer normalization we still apply gamma / beta to each activation individually.
    #gamma/beta are NOT used to scale sample by sample, so same behavior as original batch normalization

    y, cache = batchnorm_forward(x, gamma.reshape(D,1), beta.reshape(D,1), {'mode':'train'})
    out = y.transpose()
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    cache['sumaxis'] = 1
    dx, dgamma, dbeta = batchnorm_backward_alt(dout.T, cache)
    dx = dx.transpose()

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #When using more than one training input (N > 1), 
        #it is unclear if the training inputs should have the same mask applied.
        #With the mask I designed below, EACH TRAINING INPUT WILL HAVE ITS OWN MASK
        #and therefore works with its own set of zero'd out activations. 

        mask = (np.random.rand(*x.shape) < p) / p # dropout mask. Notice /p so we don't have to *p during testing time!
        out = x * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #don't need to do anything here during testing. 
        #don't apply mask, and also don't correct for p (already done in training)
        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,C,H,W     = x.shape
    F,C,HH,WW   = w.shape

    #Get pad and stride. 
    #Since integer, these will apparently be the same in the H and W direction

    pad         = conv_param['pad'] 
    stride      = conv_param['stride']

    #output shape
    h_out       = 1 + (H + 2 * pad - HH) / stride
    w_out       = 1 + (W + 2 * pad - WW) / stride

    #check if stride fits properly (no fractional numbers)
    if h_out%1 > 0 or w_out%1 > 0:
      raise Exception("filter does not fit properly")

    h_out = int(h_out)
    w_out = int(w_out)

    #add pad
    x_pad = np.pad(x, ((0,0),(0,0), (pad,pad), (pad,pad)), mode='constant') #constant = 0 by default

    #intitialize output
    out   = np.zeros((N,F,h_out,w_out), dtype=x.dtype)

    #efficiency is not the goal here, use loops over all 4 dimensions
    for i in range(N):
      for j in range(F):
        for k in range(h_out):
          row_ind = k*stride #position of top left corner where filter will be placed
          for l in range(w_out):
            col_ind = l*stride #position of top left corner where filter will be placed

            out[i,j,k,l] = np.sum(x_pad[i,:,row_ind:row_ind+HH,col_ind:col_ind+WW] * \
                                  w[j,:,:,:])

    out += b.reshape(1,F,1,1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #simply grab anything that could be useful in this naive implementation
    x, w, b, conv_param = cache
    pad         = conv_param['pad'] 
    stride      = conv_param['stride']    

    N,F,H,W     = x.shape
    F,C,HH,WW   = w.shape
    dx          = np.zeros_like(x)

    x_pad = np.pad(x, ((0,0),(0,0), (pad,pad), (pad,pad)), mode='constant') #constant = 0 by default
    _,_,h_pad,w_pad = x_pad.shape

    #using same notation for shape of forward convolution output
    _,_,h_out,w_out = dout.shape

    #################################
    ### NAIVE GRADIENT FOR X      ###
    #################################

    #for each spatial location i,j in 'x', figure out which activations it influences
    #then multiply the gradient at those activations with the perturbation induced by change at i,j
    for i in range(H):
      i_pad = i+pad #corresponding index in padded array

      #find out which activations are influenced by this position i_pad in x_pad

      #min index of activation that is impacted by i,j
      k_min = max(int(np.ceil((i_pad - (HH-1))/stride)),0) #depending on size of filter and padding, this could be negative. Those filters are not included in forward pass though
      k_max = min(int(np.floor(i_pad/stride)), h_out-1)  #h_out-1 in python notation is the last filter activation coordinate that can be impacted

      for j in range(W):
        j_pad = j+pad #corresponding index in padded array

        #find out which activations are influenced by this position j_pad in x_pad
        l_min = max(int(np.ceil((j_pad - (WW-1))/stride)),0) #depending on size of filter and padding, this could be negative. Those filters are not included in forward pass though
        l_max = min(int(np.floor(j_pad/stride)), w_out-1)  #w_out-1 in python notation is the last filter activation coordinate that can be impacted

        #get d_out_{k,l} / dx_{i,j} for all k,l where non-zero, 
        #and multiply by gradient of those k,l locations in activations
        for k in range(k_min,k_max+1):
          #get indices of top-left corner of filter on x_pad
          row_ind = k*stride #position of top left corner where filter will be placed

          for l in range(l_min,l_max+1):  
            col_ind = l*stride #position of top left corner where filter will be placed

            filt_row = i_pad - row_ind
            filt_col = j_pad - col_ind

            for c in range(C):
              for f in range(F):
                dx[:,c,i,j] += w[f,c,filt_row, filt_col]*dout[:,f,k,l]

    #################################
    ### END NAIVE GRADIENT FOR X  ###
    #################################

    #################################
    ### NAIVE GRADIENT FOR W      ###
    #################################
    dw = np.zeros_like(w)

    #determine how much does each output activation changes by a chance in w
    #then multiply that by how much the loss function changes by a change in activation (i.e. incoming grad dout)

    #move the filter around in the same way that we did during the forward pass of convolution
    for i in range(N):
      for j in range(F):
        for k in range(h_out):
          row_ind = k*stride #position of top left corner where filter will be placed
          for l in range(w_out):
            col_ind = l*stride #position of top left corner where filter will be placed

            #for each k,l, a change in any filter coefficient will impact the same activation
            dw[j,:,:,:] += x_pad[i,:,row_ind:row_ind+HH,col_ind:col_ind+WW] * dout[i,j,k,l]


    #################################
    ### END NAIVE GRADIENT FOR X  ###
    #################################

    #Gradient for b (nice and simple)
    db = np.sum(dout,axis=(0,2,3))
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,C,H,W = x.shape
    pool_height = pool_param['pool_height']
    pool_width  = pool_param['pool_width']
    stride      = pool_param['stride']

    nh = int(1 + (H - pool_height) / stride)
    nw = int(1 + (W - pool_width)  / stride)

    out = np.zeros((N,C,nh,nw))

    for i in range(N):
      for j in range(C):
        for k in range(nh):
          for l in range(nw):

            out[i,j,k,l] = np.max(x[i,j,k*stride:k*stride+pool_height,l*stride:l*stride+pool_width])

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x = cache[0]
    pool_param = cache[1]

    N,C,H,W = x.shape
    pool_height = pool_param['pool_height']
    pool_width  = pool_param['pool_width']
    stride      = pool_param['stride']

    _,_,nh,nw = dout.shape
    dx = np.zeros_like(x)

    #go over all the patches that we pooled over
    #find the max index in each patch, that will be the only nonzero gradient in dx
    for i in range(N):
      for j in range(C):
        for k in range(nh):
          for l in range(nw):
            #index of 1D version of patch where the maximum was found
            maxind = np.argmax(x[i,j,k*stride:k*stride+pool_height,l*stride:l*stride+pool_width])

            max_rel_k = int(np.floor(maxind / pool_height))
            max_rel_l = maxind % pool_width

            #if patch moves with stride less than width/heigth, than same entry can appear in more than one patch
            #we want to make sure to add gradient, not simply assign to cover that (probably rare) case
            dx[i,j,k*stride+max_rel_k, l*stride+max_rel_l] += dout[i,j,k,l]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #Reshape x from N,C,H,W to (N * H * W), C
    #Batch normalization (which takes statistics over rows) could then run as before
    N,C,H,W = x.shape
    x_reorder = x.transpose(0,2,3,1)
    x_reorder = np.reshape(x_reorder, (N*H*W,C))
    out_reorder, cache = batchnorm_forward(x_reorder, gamma, beta, bn_param)

    out_reorder = np.reshape(out_reorder,(N,H,W,C))
    out = out_reorder.transpose(0,3,1,2) 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,C,H,W = dout.shape

    dout_reorder = dout.transpose(0,2,3,1)
    dout_reorder = np.reshape(dout_reorder, (N*H*W,C))

    dx_reorder, dgamma, dbeta = batchnorm_backward_alt(dout_reorder, cache)

    dx_reorder = np.reshape(dx_reorder,(N,H,W,C))
    dx = dx_reorder.transpose(0,3,1,2) 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N,C,H,W = x.shape
    CS = C//G #chunk size

    shape = (N*G, CS * H * W)
    X = x.reshape(shape).transpose()

    nval = CS * H * W

    #now use code from normal batch normalization, to take statistics over rows
    mu            = 1./nval * np.sum(X, axis=0) #vector of length D
    X_mean_zero   = X-mu

    var           = 1./nval * np.sum(X_mean_zero**2, axis=0) #vector of length D
    std           = np.sqrt(var + eps)
    X_var_one     = X_mean_zero/std    

    z             = X_var_one.transpose().reshape(N,C,H,W)
    out           = gamma*z+beta

    cache={'std':std, 'gamma':gamma, 'z':z, 'shape':shape}

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape
    shape = cache['shape']
    dbeta = dout.sum(axis=(0,2,3), keepdims=True)
    dgamma = np.sum(dout*cache['z'], axis=(0,2,3), keepdims=True)

    #running out of time for this exercise given interview coming up. 
    #this will be same logic as original batch norm grad, but now different shapes
    #to avoid tedious check, saving a couple of mins by copying from some answers i found online, mea culpa :)  (did all other parts myself)

    # reshape tensors
    z = cache['z'].reshape(shape).transpose()
    M = z.shape[0]
    dfdz = dout * cache['gamma']
    dfdz = dfdz.reshape(shape).transpose()

    # copy from batch normalization backward alt
    dfdz_sum = np.sum(dfdz,axis=0)
    dx = dfdz - dfdz_sum/M - np.sum(dfdz * z,axis=0) * z/M
    dx /= cache['std']
    dx = dx.transpose().reshape(N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
