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
import uuid 
from scipy.stats import entropy
import pickle
import os






def get_permanence_vals(sp):
  m = sp.getNumInputs()
  n = np.prod(sp.getColumnDimensions())
  W = np.zeros((n, m))
  for i in range(sp._numColumns):
      sp.getPermanence(i, W[i, :])
  return W


def update_statistics(Y, P, beta=0.9):
  """
  Args
  ----
  Y: 
      2d array whose columns encode the 
      activity of the output units
  P:
      2d array encoding the pairwise average 
      activity of the output units  

  Returns
  -------
      The updated average activities
  """

  (n, d) = Y.shape
  

  A = np.expand_dims(Y, axis=1) * np.expand_dims(Y, axis=0)
  assert(A.shape == (n, n, d))
  
  Q = np.mean(A, axis=2)
  Q[np.where(Q == 0.)] = 0.000001
  assert(P.shape == Q.shape)

  return beta*P + (1-beta)*Q


def compute_probabilities_from(Y):
  (n, d) = Y.shape
  A = np.expand_dims(Y, axis=1) * np.expand_dims(Y, axis=0)    
  P = np.mean(A, axis=2)
  return P

def compute_conditional_probabilies(Y):
  (n, d) = Y.shape
  
  A = np.expand_dims(Y, axis=1) * np.expand_dims(Y, axis=0)
  assert(A.shape == (n, n, d))
  
  Q = np.mean(A, axis=2)
  Q[np.where(Q == 0.)] = 0.000001
  assert(P.shape == Q.shape)

  Diag = Q.diagonal().reshape((n,1))

  Q = Q / Diag

  np.fill_diagonal(Q, 0.)
  Q = Q + np.diag(Diag.reshape(-1))

  return beta*P + (1-beta)*Q


def random_mini_batches(X, Y, minibatch_size, seed=None):
  """
  Compute a list of minibatches from inputs X and targets Y.
  A datapoint is expected to be represented as a column in 
  the data matrices X and Y.
  """
  d    = X.shape[1]
  size = minibatch_size
  minibatches = []

  if Y is None:
      Y = np.zeros((1, d))

  np.random.seed(seed)
  perm = np.random.permutation(d)

  for t in range(0, d, size):
      subset = perm[t: t+size]
      minibatches.append((X[:, subset], Y[:, subset]))

  return minibatches


def scalar_reconstruction(x):
  # x = (x>0.05).astype(float)
  v = [ x[i]*i  for i in range(len(x))]
  s = np.mean(v)
  s = s/len(x)
  return s


def trim_doc(docstring):
  """"
  Removes the indentation from a docstring.
  Credit goes to: 
      http://codedump.tumblr.com/post/94712647/handling-python-docstring-indentation
  """
  if not docstring:
      return ''
  lines = docstring.expandtabs().splitlines()

  # Determine minimum indentation (first line doesn't count):
  indent =  sys.maxsize
  for line in lines[1:]:
      stripped = line.lstrip()
      if stripped:
          indent = min(indent, len(line) - len(stripped))

  # Remove indentation (first line is special):
  trimmed = [lines[0].strip()]
  if indent < sys.maxsize:
      for line in lines[1:]:
          trimmed.append(line[indent:].rstrip())

  # Strip off trailing and leading blank lines:while trimmed and not trimmed[-1]:
      trimmed.pop()
  while trimmed and not trimmed[0]:
      trimmed.pop(0)
  return '\n'.join(trimmed)


def random_id(length):
  """Returns a random id of specified length."""
  assert(length < 10)
  x = str(uuid.uuid4())
  return x[:length]


def create_movie(fig, update_figure, filename, title, fps=15, dpi=100):
  """Helps us to create a movie."""
  FFMpegWriter = manimation.writers['ffmpeg']
  metadata     = dict(title=title)
  writer       = FFMpegWriter(fps=fps, metadata=metadata)

  with writer.saving(fig, filename, dpi):
      t = 0
      while True:
          if update_figure(t):
              writer.grab_frame()
              t += 1
          else:
              break


def add_noise(X, noise_level = 0.05):
  noisy_X = X.copy()
  noise = (np.random.sample(X.shape) < noise_level)
  mask  = np.where( noise == True ) 
  noisy_X[mask] = X[mask] + (-1.0)**X[mask] 
  assert(noisy_X.shape == X.shape)
  return noisy_X

def add_noisy_bits(X, noise_level = 0.05):
  noisy_X = X.copy()
  noise = (np.random.sample(X.shape) < noise_level)
  mask  = np.where( noise == True ) 
  noisy_X[mask] = 1.0
  assert(noisy_X.shape == X.shape)
  return noisy_X

def subtract_noisy_bits(X, noise_level = 0.05):
  noisy_X = X.copy()
  noise = (np.random.sample(X.shape) < noise_level)
  mask  = np.where( noise == True ) 
  noisy_X[mask] = 0.0
  assert(noisy_X.shape == X.shape)
  return noisy_X


