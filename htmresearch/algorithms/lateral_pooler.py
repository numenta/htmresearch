# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
import numpy as np
from htmresearch.support.lateral_pooler.utils import random_mini_batches



class LateralPooler(object):
  """
  A lightweight experimental spatial pooler implementation
  with learned lateral inhibitory connections.

  Example Usage
  -------------
  ```
  # Instantiate
  pooler = SpatialPooler(...)

  # Training
  X = load_training_data()
  pooler.fit(X, batch_size=32, num_epochs=10)

  ```
  """

  def __init__(self, 
               input_size           = 784, 
               output_size          = 128, 
               code_weight          = 4, 
               seed                = -1,
               learning_rate        = 0.01,
               smoothing_period     = 50., 
               boost_strength       = 100.,
               boost_strength_hidden = 100.,
               inc_dec_ratio         = 1.,
               permanence_threshold  = -1.0,
               learning_rate_hidden  = 1.
    ):
    """
    Args
    ----
    input_size: 
        Number of visible units
    output_size:
        Number of hidden units
    code_weight:
        Number of desired active output units
    seed:
        Random seed

    ...

    """
    self.seed = seed
    if seed != -1:
        self._random = np.random.RandomState(seed)
    else:
        self._random = np.random

    self.input_size  = input_size
    self.output_size = output_size
    self.shape       = (output_size, input_size)
    self.code_weight = code_weight
    self.sparsity    = float(code_weight)/float(output_size)

    # ---------------------
    #  Network connections 
    # ---------------------
    (n, m) = self.shape
    self.feedforward = self._random.rand(n, m)
    self.inhibitory  = np.ones((n, n))/float(n)
    np.fill_diagonal(self.inhibitory, 0.)
    self.boostfactor = np.ones((n, 1))

    # ---------------------
    #  Statistics
    # ---------------------
    self.avg_activity_units = np.zeros(n)
    self.avg_activity_pairs = 0.0000001*np.ones((n, n))

    # Placeholder. Variable used to store the ovelap scores
    # during encoding, in order to make the score accessible
    # for debugging and analyses.
    self._scores = None

    # ---------------------
    #  Learning parameters
    # ---------------------
    self.boost_strength        = boost_strength
    self.boost_strength_hidden = boost_strength_hidden
    self.permanence_threshold = float(permanence_threshold)
    self.smoothing_period = float(smoothing_period)
    self.learning_rate    = float(learning_rate)
    self.inc_dec_ratio    = float(inc_dec_ratio)
    self.learning_rate_hidden = learning_rate_hidden

  def get_connections(self):
    return self.feedforward, self.boostfactor, self.inhibitory


  def set_connections(self, W, b, H):
    self.feedforward[:,:] = W
    self.boostfactor[:,:] = b
    self.inhibitory[:,:]  = H
    return self


  def encode(self, X):
    """
    Encodes a batch of input vectors, where the inputs are 
    given as the columns (!!!) of the matrix X (not the rows).
    """ 
    W, boost, H = self.get_connections()
    n, m  = W.shape 

    
    # Optional thresholding of permanence values:
    # (note that the original SP does this)
    W_prime = W
    theta = self.permanence_threshold
    if theta > 0:
      W_prime = (W >= theta).astype(float)
    
    d = X.shape[1]
    Y = np.zeros((n,d))
    s = self.sparsity

    inh_signal   = np.zeros((n, d))
    scores       = boost * np.dot(W_prime, X)
    self._scores = scores 
    sorted_score_args = np.argsort(scores, axis=0, kind='mergesort')[::-1, :]

    
    for t in range(d):
      current_weight = 0
      for i in sorted_score_args[:, t]:

        too_strong = ( inh_signal[i,t] >= s )

        if not too_strong:
          current_weight += 1
          Y[i, t] = 1.
          inh_signal[:, t] += H[i,:]

        if self.learning_rate_hidden == 0.0 and current_weight == self.code_weight:
          print current_weight
          break

    return Y


  def compute_dW(self, X, Y):
    """
    Computes the weight update for the feedforward weights
    according to the Hebbian-like update rule in the paper.
    """
    n, m, d = Y.shape[0], X.shape[0], X.shape[1]
    r       = self.inc_dec_ratio

    Pos = np.mean(np.expand_dims(Y , axis=1) * np.expand_dims(    X, axis=0), axis=2)
    Neg = np.mean(np.expand_dims(Y , axis=1) * np.expand_dims(1 - X, axis=0), axis=2)
    dW  = Pos  -  (1.0/r) * Neg

    return dW

  def compute_dW_online(self, X, Y):
    """
    Computes the weight update for the feedforward weights
    according to the Hebbian-like update rule in the paper.
    """
    n, m, d = Y.shape[0], X.shape[0], X.shape[1]
    r       = self.inc_dec_ratio

    Pos = np.dot(Y,X.T)
    Neg = np.dot(Y ,(1.0 - X).T)
    dW  = Pos  -  (1.0/r) * Neg

    return Pos, Neg


  def update_feedforward(self, X, Y):
    alpha = self.learning_rate
    W  = self.feedforward
    dW = self.compute_dW(X, Y)

    W[:,:] = W  +  alpha * dW 
    W[np.where(W > 1.0)] = 1.0
    W[np.where(W < 0.0)] = 0.0

  def update_feedforward_online(self, X, Y):
    inc = self.learning_rate
    dec = inc/self.inc_dec_ratio
    W  = self.feedforward
    dW_pos, dW_neg = self.compute_dW_online(X, Y)

    W[:,:] = W  +  inc * dW_pos - dec*dW_neg
    W[np.where(W > 1.0)] = 1.0
    W[np.where(W < 0.0)] = 0.0


  def update_inhibitory(self):
    C = self.boost_strength_hidden
    H = self.inhibitory
    P = self.avg_activity_pairs
    epsilon = self.learning_rate_hidden
    H[:,:] = P[:,:]
    # H[:,:] = np.exp( C * P )
    np.fill_diagonal( H, 0.0)
    H[:,:] = (1.0 - epsilon)*H + epsilon*H/np.sum( H, axis=1, keepdims=True)

  def update_inhibitory_from_P(self, P):
    C = self.boost_strength_hidden
    H = self.inhibitory
    epsilon = self.learning_rate_hidden
    H[:,:] = P[:,:]
    # H[:,:] = np.exp( C * P )
    np.fill_diagonal( H, 0.0)
    H[:,:] = (1.0 - epsilon)*H + epsilon * H/np.sum( H, axis=1, keepdims=True)

  def update_boost(self):
    C = self.boost_strength
    p = self.avg_activity_units
    n = self.output_size
    s = self.sparsity
    self.boostfactor = np.exp(s - C * p).reshape((n,1))


  def update_connections(self, X, Y):
    """
    Method that updates the model parameters, i.e. feedforward connections, 
    lateral connections, and homeostatic boost factors, according to the
    update rules in the paper.
    """
    beta = 1 - 1/self.smoothing_period
    self.update_statistics(Y, beta)
    self.update_feedforward(X,Y)
    self.update_boost()
    self.update_inhibitory()

  def update_connections_online(self, X, Y):
    """
    Method that updates the model parameters, i.e. feedforward connections, 
    lateral connections, and homeostatic boost factors, according to the
    update rules in the paper.
    """
    beta = 1 - 1/self.smoothing_period
    self.update_statistics_online(Y, beta)
    self.update_feedforward_online(X,Y)
    self.update_boost()
    self.update_inhibitory()

  def fit(self, X, batch_size=32, num_epochs=10, initial_epoch=0, callbacks=[]):
    """
    Fits a model to a training set of inputs.
    """
    seed  = self.seed

    for callback in callbacks:
      callback.set_model(self)

    cache = {
      "num_epochs": num_epochs,
      "initial_epoch": initial_epoch,
      "batch_size": batch_size
    }

    for epoch in range(initial_epoch, num_epochs):


        for callback in callbacks:
          callback.on_epoch_begin(epoch, cache)

        minibatches = random_mini_batches(X, None, batch_size, seed)
        cache["num_batches"] = len(minibatches)

        num_batches = len(minibatches)
        for t, (X_t, _) in enumerate(minibatches):

            for callback in callbacks:
              callback.on_batch_begin((X_t, None), cache)

            Y_t = self.encode(X_t)
            self.update_connections(X_t, Y_t)

            for callback in callbacks:
              callback.on_batch_end((X_t, Y_t), cache)

        for callback in callbacks:
          callback.on_epoch_end(epoch, cache)


  def update_statistics(self, Y, beta=0.9):
      """
      Updates the exponential moving averages over pairwise and individual 
      cell activities. 
      """
      P_pairs = self.avg_activity_pairs 
      P_units = self.avg_activity_units

      A = np.expand_dims(Y, axis=1) * np.expand_dims(Y, axis=0)        
      Q = np.mean(A, axis=2)
      # Q[np.where(Q == 0.)] = 0.000001

      P_pairs[:,:] = beta*P_pairs + (1-beta)*Q
      P_units[:]   = P_pairs.diagonal()


  def update_statistics_online(self, Y, beta=0.9):
      """
      Updates the exponential moving averages over pairwise and individual 
      cell activities. 
      """
      P_pairs = self.avg_activity_pairs 
      P_units = self.avg_activity_units

      Q = Y * Y.T   

      P_pairs[:,:] = beta*P_pairs + (1-beta)*Q
      P_units[:]   = P_pairs.diagonal()









