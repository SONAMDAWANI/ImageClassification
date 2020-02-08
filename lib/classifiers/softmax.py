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

  num_train = X.shape[0]
  countclass = W.shape[1]
  model_loss= 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores = scores-np.max(scores)
    scoreexp=np.exp(scores)
    summationj = np.sum(scoreexp)
    valp = lambda classi: np.exp(scores[classi])/summationj
    model_loss = model_loss-np.log(valp(y[i]))

    for classi in range(countclass):
      valp_classi = valp(classi)
      dW[:, classi] += (valp_classi - (classi == y[i])) * X[i]

  model_loss= model_loss/num_train
  model_loss = model_loss+0.5 * reg * np.sum(W * W)
  loss=model_loss
  dW = dW/num_train

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  scores = X.dot(W)
  print('scores ',scores)
  scores=scores - np.max(scores,axis=1,keepdims=True)
  summation_scores = np.sum(np.exp(scores),axis=1,keepdims=True)
  valp = np.exp(scores) /summation_scores
  tempval=np.log(valp[np.arange(num_train),y])
  loss =np.sum((tempval)*-1)
  vecind =np.zeros_like(valp)
  vecind[np.arange(num_train),y] = 1
  dW=X.T.dot(valp-vecind)

  loss = loss/num_train
  loss = loss + (0.5*reg*np.sum(W*W))
  dW =dW/num_train
  dW =dW + (reg*W)
  
  return loss, dW

