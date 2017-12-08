# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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
import pytest
import numpy as np
from lateralpooler.lateral_pooler import SpatialPooler
import itertools

def test_whether_encoding_is_the_same_as_wmax_for_uniform_H():
	"""
	Naive reality check for the encoding function 
	of the lateral pooler implementation.
	"""
	n = 24
	m = 16
	d = 100
	w = 4

	X = np.random.rand(m,d)
	
	W = np.random.rand(n,m)
	b = np.exp( - np.random.rand(n,1) )
	H = np.ones((n,n))/n
	np.fill_diagonal(H, 0.)

	pooler = SpatialPooler(inputSize=m, outputSize=n, codeWeight=4, seed=1)
	pooler.set_connections(W,b,H)
	Result = pooler.encode(X)

	S    = b * np.dot(W, X)
	wmax = np.sort(S, axis=0)[::-1][[w],:]
	Expected = (S - wmax > 0).astype(float) 

	print(Expected)
	print(Result)
	assert(np.all(Expected == Result))


def test_feedforward_weight_update():
	"""
	"""
	X = np.array([
		[1, 0, 1],
		[1, 0, 0],
		[0, 1, 0],
		[0, 1, 0],
		[0, 1, 1]
	])
	Y = np.array([
		[1, 0, 1],
		[1, 0, 0],
		[0, 1, 0]
	])

	m = X.shape[0]
	n = Y.shape[0]
	d = X.shape[1]
	
	ratio = 2.
	incr  = 1.
	decr  = 1./ratio

	pooler = SpatialPooler(inputSize=m, outputSize=n, seed=1, incDecRatio = ratio)

	Result   = pooler.compute_dW(X,Y)
	Expected = np.zeros((n,m))
	for t in range(d):
		for i,j in itertools.product(range(n), range(m)):
			if  Y[i,t] + X[j,t] == 2:
				Expected[i,j] += incr
			elif Y[i,t] + X[j,t] == 1:
				Expected[i,j] -= decr
	Expected = Expected/d
	
	epsilon = 0.000000001
	assert(np.all(Expected - Result < epsilon))





