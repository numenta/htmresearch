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

def load_data(label, **args):

  X = None
  Y = None
  X_test = None
  Y_test = None

  if label == "mnist":
    mnist = fetch_mldata('MNIST original')
    X = mnist.data.T
    Y = mnist.target.reshape((1,-1))

    perm = np.random.permutation(X.shape[1])
    X = X[:,perm]
    Y = Y[:,perm]

    threshold = 0.2
    X = X/255
    X = (X > threshold).astype(float)

    X,Y, X_test, Y_test = X[:,:-10000], Y[:,:-10000], X[:,-10000:], Y[:,-10000:]



  elif label == "xy_biased":
    encoder_params = {
        "dimensions"      : 2, 
        "max_values"      : [[0.,1.]]*2,
        "bits_per_axis"   : [50, 150],
        "weight_per_axis" : [5, 15],
        "wrap_around"     : False
    }

    SDR = ScalarEncoder(**encoder_params)

    Y = np.random.sample(( 2, args["num_inputs"]))
    X = np.array([ SDR(y) for y in Y.T]).T

    Y_test = np.array([[0.1 + x*0.2, 0.1 + y*0.2] for y,x in itertools.product(range(5), repeat=2)]).T
    X_test = np.array([ SDR(y) for y in Y_test.T]).T


  elif label == "natural":

    mat = loadmat("../data/IMAGES.mat")
    img = mat["IMAGES"]
    
    size = 20
    d_test = 20
    d    = args["num_inputs"] + d_test

    X = np.zeros((size, size, d))
    x = np.random.randint(0, 512-size, d)
    y = np.random.randint(0, 512-size, d)
    i = np.random.randint(0, 10, d)
    for t in range(d):
        xs = range(x[t], x[t] + size)
        ys = range(y[t], y[t] + size)
        patch = img[x[t]:(x[t] + size), y[t]: (y[t] + size), i[t]]
        patch = (patch < 0.0).astype(float)
        X[:,:,t] = patch

    X = X.reshape((size**2, d))


    X_test = X[:,d-d_test:]
    X      = X[:,:d]

  elif label == "natural_inv":

    mat = loadmat("../data/IMAGES.mat")
    img = mat["IMAGES"]
    
    size = 20
    d_test = 20
    d    = args["num_inputs"] + d_test

    X = np.zeros((size, size, d))
    x = np.random.randint(0, 512-size, d)
    y = np.random.randint(0, 512-size, d)
    i = np.random.randint(0, 10, d)
    for t in range(d):
        xs = range(x[t], x[t] + size)
        ys = range(y[t], y[t] + size)
        patch = img[x[t]:(x[t] + size), y[t]: (y[t] + size), i[t]]
        patch = (patch < 0.0).astype(float)
        X[:,:,t] = patch

    X = X.reshape((size**2, d))


    X_test = 1. - X[:,d-d_test:]
    X      = 1. - X[:,:d]


  elif label == "xy_biased_dependent":
    encoder_params = {
        "dimensions"      : 2, 
        "max_values"      : [[0.,1.]]*2,
        "bits_per_axis"   : [50, 150],
        "weight_per_axis" : [5, 15],
        "wrap_around"     : False
    }

    SDR = ScalarEncoder(**encoder_params)

    T      = np.random.sample(( 1, args["num_inputs"]))
    F_of_T = np.cos(5*np.pi*T)
    Y = np.zeros((2, args["num_inputs"]))
    Y[0,:] = T
    Y[1,:] = F_of_T
    X = np.array([ SDR(y) for y in Y.T]).T

    # Y_test = np.array([[0.1 + x*0.2, 0.1 + y*0.2] for y,x in itertools.product(range(5), repeat=2)]).T
    # X_test = np.array([ SDR(y) for y in Y_test.T]).T


  elif label == "simply_x":
    encoder_params = {
        "dimensions"      : 1, 
        "max_values"      : [[0.,1.]],
        "bits_per_axis"   : [250],
        "weight_per_axis" : [15],
        "wrap_around"     : False
    }

    SDR = ScalarEncoder(**encoder_params)

    Y = np.random.sample((1, args["num_inputs"]))
    X = np.array([ SDR(y) for y in Y.T]).T

    Y_test = np.linspace(0, 1., num=10)
    X_test = np.array([ SDR(y.reshape(1,1)) for y in Y_test.T]).T


  elif label == "squares_biased":
    if "num_inputs" in args:
        d = args["num_inputs"] + 1000
    else:
        d = 20000 + 1000

    X = np.zeros((28,28, d))
    Y = np.zeros((2,2,d)).astype(int)
    Y[:,0,:] = np.random.randint(0,26, size=(2, d))
    Y[:,1,:] = np.random.randint(0,22, size=(2, d))
    for t in range(d):
        a = Y[:,0,t]
        b = Y[:,1,t]
        X[a[0]:(a[0]+2), a[1]:(a[1]+2), t] = 1.
        X[b[0]:(b[0]+6), b[1]:(b[1]+6), t] = 1.

    X = X.reshape(28**2, d)
    X_test = X[:,-1000:]


  elif label == "2d_space":

    d = args["num_inputs"] + 10000
    width, height = args["shape"]
    s = args["size"]

    X = np.zeros((x_size, y_size, d))
    Y = np.zeros((2,2,d)).astype(int
        )
    Y[:,0,:] = np.random.randint(0, width,  size=(2, d))
    Y[:,1,:] = np.random.randint(0, height, size=(2, d))

    for t in range(d):
        a = Y[:,0,t]
        b = Y[:,1,t]
        X[a[0]:(a[0]+s), a[1]:(a[1]+s), t] = 1.
        X[b[0]:(b[0]+s), b[1]:(b[1]+s), t] = 1.

    X = X.reshape(width*height, d)
    X_test = X[:,-1000:]

  return X,Y, X_test, Y_test




