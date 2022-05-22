import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    N,D = X.shape
    C   = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################


    #Compute scores and gradient

    for i in xrange(N): #for each image...
        scores = np.zeros(C)
        for j in xrange(C): # for each class
            scores[j] = np.dot(X[i,:], W[:,j]) #compute the score

        #subtract the max score (numerical stability, will not impact the final loss we compute. See http://cs231n.github.io/linear-classify/ for details
        scores = scores-np.max(scores)
        scores_exp = np.exp(scores)
        sum_scores_exp = np.sum(scores_exp)
        loss_i = - np.log(scores_exp[y[i]]/sum_scores_exp)

        #update gradient
        for j in xrange(C):
            dW[:,j] += X[i,:]*scores_exp[j]/sum_scores_exp

            if j == y[i]:
                dW[:,j] += -X[i,:]

        loss += loss_i

    loss = loss / N
    dW   = dW / N
    #print("Data loss is %e"%loss)
    #add regularization loss

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg*2*W



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
  
    N,D = X.shape
    C   = W.shape[1]

    scores = np.dot(X,W)
    scores = scores - np.max(scores) #not exactly the same as in naiive where I subtract the scores for each image with the max score of that image. But should give similar accuracy but is cheaper

    scores_exp = np.exp(scores) 
    scores_exp_sum = np.sum(scores_exp, axis=1)

    p_ik = (scores_exp.transpose()/scores_exp_sum).transpose() #Due to normalization, can be interpreted as probabilities
    L_i = -np.log(p_ik[np.arange(N),y])
    loss = 1./N*np.sum(L_i)

    index_mat = np.zeros_like(p_ik)
    index_mat[np.arange(N),y] = 1 #sparse mat would be more efficient ofc

    dW = np.dot(X.transpose(),(p_ik-index_mat))
    dW = 1./N*dW

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg*2*W


    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

