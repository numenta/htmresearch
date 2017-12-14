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
from scalar_encoder import ScalarEncoder 
from scipy.io import loadmat
from sklearn.datasets import fetch_mldata
import itertools

def encode_streams(data_streams, bits_per_axis, weight_per_axis):

    Y = data_streams
    num_streams = Y.shape[0]
    num_points  = Y.shape[1]
    enc = []
    for i in range(num_streams):
        enc.append(ScalarEncoder(0.,1.,bits_per_axis[i], weight_per_axis[i]))

    X = np.zeros((np.sum(bits_per_axis), num_points))

    
    for t in range(num_points):
        X[:,t] = np.concatenate([enc(Y[i,t]) for i in range(num_streams)])

    return X


def xy_biased(bits_per_axis=[50,150], weight_per_axis=[5,15], num_samples=100000):

    Y = np.random.sample((2, num_samples))
    X = encode_streams(Y, bits_per_axis, weight_per_axis)

    Y_test = np.array([[0.1 + x*0.2, 0.1 + y*0.2] for y,x in itertools.product(range(5), repeat=2)]).T
    X_test = encode_streams(Y_test, bits_per_axis, weight_per_axis)

    return (X, Y, X_test, Y_test)


def mnist(threshold=0.2):

    mnist = fetch_mldata('MNIST original')
    X     = mnist.data.T
    Y     = mnist.target.reshape((1,-1))

    perm = np.random.permutation(X.shape[1])
    X    = X[:,perm]
    Y    = Y[:,perm]

    X = X/255
    X = (X > threshold).astype(float)

    return (X[:,:60000], Y[:,:60000], X[:,-10000:], Y[:,-10000:])


def load_data(label):

  if label == "mnist":
    return mnist()

  elif label == "xy_biased":
    return xy_biased()
  
  else:
    raise "No data set with that label...."





