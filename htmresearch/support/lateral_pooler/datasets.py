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
        X[:,t] = np.concatenate([enc[i](Y[i,t]) for i in range(num_streams)])

    return X


def xy_biased(bits_per_axis=[50,150], weight_per_axis=[5,15], num_samples=100000):

    Y = np.random.sample((2, num_samples))
    X = encode_streams(Y, bits_per_axis, weight_per_axis)

    Y_test = np.array([[0.1 + x*0.2, 0.1 + y*0.2] for y,x in itertools.product(range(5), repeat=2)]).T
    X_test = encode_streams(Y_test, bits_per_axis, weight_per_axis)

    return (X, Y, X_test, Y_test)


def mnist(threshold=0.3):

    mnist = fetch_mldata('MNIST original')
    X     = mnist.data.T
    Y     = mnist.target.reshape((1,-1))

    perm = np.random.permutation(X.shape[1])
    X    = X[:,perm]
    Y    = Y[:,perm]

    X = X/255.0
    X = (X > threshold).astype(float)

    return (X[:,:60000], Y[:,:60000], X[:,-10000:], Y[:,-10000:])

def mnist_two_channel(threshold=0.2):

    mnist = fetch_mldata('MNIST original')
    X     = mnist.data.T
    Y     = mnist.target.reshape((1,-1))

    perm = np.random.permutation(X.shape[1])
    X    = X[:,perm]
    Y    = Y[:,perm]

    X = X/255.0
    X = (X > threshold).astype(float)

    X2 = np.zeros((784,2,70000))
    X2[:,1,:] = X
    X2[:,0,:] = 1.0 - X
    X2 = X2.reshape((784*2,70000))

    return (X2[:,:60000], Y[:,:60000], X2[:,-10000:], Y[:,-10000:])


def uniform_2d(bits_per_axis=100, weight_per_axis=16, num_samples=60000):
    R = np.random.randint(bits_per_axis - weight_per_axis, size=(2,num_samples))
    Y = R/float(bits_per_axis - weight_per_axis)
    X = np.zeros((bits_per_axis, bits_per_axis, num_samples))
    C = np.zeros((2, num_samples))
    X_test = np.zeros((bits_per_axis, bits_per_axis, 400))
    C_test = np.zeros((2, 400))
    R_test = np.random.randint(30, 60 - weight_per_axis, size=(2,400))
    for t in range(num_samples):
        C[0,t] = R[0,t] + weight_per_axis//2
        C[1,t] = R[1,t] + weight_per_axis//2
        for i in range(R[0,t], R[0,t] + weight_per_axis):
            X[i, range(R[1,t], R[1,t] + weight_per_axis), t] = 1.0

    for t in range(400):
        C_test[0,t] = R_test[0,t] + weight_per_axis//2
        C_test[1,t] = R_test[1,t] + weight_per_axis//2
        for i in range(R_test[0,t], R_test[0,t] + weight_per_axis):
            X_test[i, range(R_test[1,t], R_test[1,t] + weight_per_axis), t] = 1.0


    X = X.reshape((bits_per_axis**2,-1))
    X_test = X_test.reshape((bits_per_axis**2,-1))

    return X[:,:80000], C[:,:80000], X_test[:,:], C_test[:,:]
    

def random_walk_2d(bits_per_axis=100, weight_per_axis=5, num_samples=500000):

    radius = 30
    steps = 100
    w = weight_per_axis
    bpa = bits_per_axis
    X = np.zeros((bpa, bpa, num_samples))

    for t in range(num_samples//steps):
        cx = np.random.randint(0,bits_per_axis - radius)
        cy = np.random.randint(0,bits_per_axis - radius)
        for s in range(steps):
            sx = np.random.randint(0, radius - w)
            sy = np.random.randint(0, radius - w)
            x = cx + sx
            y = cy + sy
            for i in range(x, x + w):
                X[i, range(y, y + w), t*steps + s] = 1.0

    X = X.reshape((bits_per_axis**2,-1))
    return X[:,:]
    

    
def load_data(label):

    if label == "mnist":
        return mnist()

    elif label == "mnist_two_channel":
        return mnist_two_channel()

    elif label == "xy_biased":
        return xy_biased()

    elif label == "xy_biased_big":
        return xy_biased(bits_per_axis=[200,600], weight_per_axis=[20,60], num_samples=100000)

    elif label == "uniform_2d":
        return uniform_2d()

    elif label == "random_walk_2d":
        return random_walk_2d()

    else:
        raise "No data set with that label...."





