from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape
    H    = W1.shape[1]
    C    = W2.shape[1]

    dW1 = np.zeros_like(W1)
    dW2 = np.zeros_like(W2)
    db1 = np.zeros_like(b1)
    db2 = np.zeros_like(b2)
    grads = {}
 
    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################

    #append the bias
    #X_app = np.hstack([X,np.ones((N,1))])
    #W1_app = np.vstack([W1, b1])
    #W2_app = np.vstack([W2, b2])

    #compute scores hidden layer, including nonlinear step (ReLu)
    #s1 = np.maximum(np.matmul(X_app,W1_app),0)

    #do bias trick again
    #s1_app = np.hstack([s1, np.ones((N,1))])

    #compute final scores
    #scores = np.matmul(s1_app, W2_app)

    #compute scores. Will do one at a time for educational purposes. Not most efficient probably
    step1 = np.matmul(X, W1)
    step2 = step1 + b1
    step3 = np.maximum(step2, 0)
    s1    = step3
    step4 = np.matmul(s1, W2)
    step5 = step4 + b2

    scores = step5

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss

    #to avoid numerical problems with softmax, subtract out the max of the score for each row
    max_rowscore = np.max(scores, axis=1)
    scores = (scores.transpose() - max_rowscore).transpose()
    scores_exp = np.exp(scores)
    scores_exp_sum = np.sum(scores_exp, axis=1)

    p_ik = (scores_exp.transpose()/scores_exp_sum).transpose() #Due to normalization, can be interpreted as probabilities
    L_i = -np.log(p_ik[np.arange(N),y])
    loss = 1./N*np.sum(L_i)

    if not np.isfinite(loss):
        import warnings
        warnings.warn("Something is not right. For at least one of the examples p_ik is 1.0 for the wrong class and 0.0 for all others (including correct class)")
        #At this point we are in hack territory. I will change W by only applying the regularization now. 
        dW1 += reg*2*W1
        dW2 += reg*2*W2
        db1 += reg*2*b1
        db2 += reg*2*b2

        grads['W1'] = dW1
        grads['W2'] = dW2
        grads['b1'] = db1
        grads['b2'] = db2

        return loss, grads

        #print("step1: ", step1)
        #print("step2: ", step2)
        #print("step3: ", step3)
        #print("step4: ", step4)
        #print("step5: ", step5)

        #print("scores: ", scores)
        #print("scores_exp: ", scores_exp)
        #print("scores_exp_sum: ", scores_exp_sum)

        #raise Exception("loss is not finite")

    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################


    #ASSUMING that b1 and b2 are also regularized. Won't see if true during test, because b1 and b2 are zero at that point.
    regloss = reg*(np.sum(W1**2) + np.sum(W2**2) + np.sum(b1**2) + np.sum(b2**2))
    regloss = regloss 
    loss = loss + regloss

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################

    #first backpropagate the gradient along the data part of the neural net

    #Compute gradient of softmax with respect to the scores
    index_mat = np.zeros_like(p_ik)
    index_mat[np.arange(N),y] = 1 #sparse mat would be more efficient ofc
    grad_soft = p_ik - index_mat


    #rules for matrices for backpropagating gradient:
    #A + B = C, DL/DA = DL/DC (so just pass on the incoming gradient)
    #B*A=C: DL/DA = B^T DL/DC. DL/DB = DL/DC*A^T -> https://math.stackexchange.com/questions/1866757/not-understanding-derivative-of-a-matrix-matrix-product (comment denis, sizes also OK)


    #compute gradient with respect to b2 (It is added to the scores as the matrix multiplication of i_n * b , where i_n is col vector of 1's of length n and b2 is row vector of biases of length c
    i_n = np.ones((N,1))
    db2 += np.matmul(i_n.transpose(), grad_soft).flatten()

    #go down the multiplication path (where we computed step4). First derivative with respect to W2
    dW2 += np.matmul(step3.transpose(), grad_soft)

    #now go to branch for 'step3'
    grad_step3 = np.matmul(grad_soft,W2.transpose())

    #now to go 'step2' branch. i.e., go through relu. Throw out any element of the gradient corresponding to an entry of step2 smaller or equal to 0
    grad_step2 = np.copy(grad_step3)
    grad_step2[step2<=0] = 0.0

    #now go to b1 branch. Again i_n * b
    db1 += np.matmul(i_n.transpose(), grad_step2).flatten()

    #now to branch of step1 (same, just addition gate)
    grad_step1 = grad_step2
    
    #now go to W1
    dW1 += np.matmul(X.transpose(), grad_step1)

    #all data gradients are divided by 1/N
    dW1/=N
    dW2/=N
    db1/=N
    db2/=N

    #add regularization
    dW1 += reg*2*W1
    dW2 += reg*2*W2
    db1 += reg*2*b1
    db2 += reg*2*b2

    grads['W1'] = dW1
    grads['W2'] = dW2
    grads['b1'] = db1
    grads['b2'] = db2

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False, replace=True, num_epoch=None):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    - replace: boolean; added by Bram. 
    - num_epoch: if specified, overrule default 'iterations_per_epoch'
    """
    num_train = X.shape[0]

    if num_epoch is not None:
        iterations_per_epoch = max(int(num_iters/num_epoch),1)
    else:
        iterations_per_epoch = max(num_train / batch_size, 1)



    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    all_inds = np.arange(num_train)
    epoch = 0
    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################

      inds = np.random.choice(all_inds, size=batch_size, replace=replace) #may have some doubles
      X_batch = X[inds,:]
      y_batch = y[inds]

      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params['W1'] -= learning_rate*grads['W1']
      self.params['W2'] -= learning_rate*grads['W2']
      self.params['b1'] -= learning_rate*grads['b1']
      self.params['b2'] -= learning_rate*grads['b2']

      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay
        #print("ending epoch %i"%epoch)
        epoch += 1

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    y_pred = np.zeros(X.shape[0])
    scores = self.loss(X)

    y_pred[:] = np.argmax(scores, axis=1) #Get the class with the maximum score for each image

    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


