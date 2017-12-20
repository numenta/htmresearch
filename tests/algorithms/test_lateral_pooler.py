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
import unittest
import numpy as np
from htmresearch.algorithms.lateral_pooler import LateralPooler
from htmresearch.algorithms.lateral_pooler_wrapper import LateralPoolerWrapper
import itertools
from nupic.algorithms.spatial_pooler import SpatialPooler
from htmresearch.support.lateral_pooler.utils import get_permanence_vals as get_W



class LateralPoolerTest(unittest.TestCase):
  """
  Simplistic tests of the experimental lateral pooler implementation.
  """

  def test_whether_encoding_is_the_same_as_wmax_for_uniform_H(self):
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

    pooler = LateralPooler(input_size=m, output_size=n, code_weight=4, seed=1)
    pooler.set_connections(W,b,H)
    Result = pooler.encode(X)

    S    = b * np.dot(W, X)
    wmax = np.sort(S, axis=0)[::-1][[w],:]
    Expected = (S - wmax > 0).astype(float) 


    assert(np.all(Expected == Result))

  def test_whether_encoding_is_the_same_as_wmax_with_zero_learning_rate(self):
    """
    Naive reality check for the encoding function 
    of the lateral pooler implementation.
    """
    n = 60
    m = 30
    d = 100
    w = 7

    X = np.random.randint(0,2,size=(m,d))
    Y_nup = np.zeros((n,d))
    Y_lat = np.zeros((n,d))

    params_nup = {
        "inputDimensions": [m,1],
        "columnDimensions": [n,1],
        "potentialRadius": n,
        "potentialPct": 1.0,
        "globalInhibition": True,
        "localAreaDensity": -1.0,
        "numActiveColumnsPerInhArea": w,
        "stimulusThreshold": 0,
        "synPermInactiveDec": 0.05,
        "synPermActiveInc"  : 0.1,
        "synPermConnected"  : 0.5,
        "minPctOverlapDutyCycle": 0.001,
        "dutyCyclePeriod": 1000,
        "boostStrength"  : 100.0,
        "seed": 1936
    }
    params_lat = params_nup.copy()
    params_lat["learningRateHidden"] = 0.0
    sp_nup = SpatialPooler(**params_nup)
    sp_lat = LateralPoolerWrapper(**params_lat)
    for i in range(n):
      sp_nup.getPermanence(i,sp_lat.feedforward[i,:])


    inc = 0.1
    dec = 0.05

    assert(sp_nup._synPermActiveInc   == sp_lat.learning_rate)
    assert(sp_nup._synPermInactiveDec == sp_lat.learning_rate/sp_lat.inc_dec_ratio)


    t = 0
    sp_nup.compute(X[:,t], False, Y_nup[:,t])
    sp_lat.compute(X[:,t], False, Y_lat[:,t])
    assert(np.all(Y_nup[:,t] == Y_lat[:,t]))


    for t in range(10):
      sp_nup.compute(X[:,t], True, Y_nup[:,t])
      sp_lat.compute(X[:,t], True, Y_lat[:,t])

      W_nup = get_W(sp_nup)
      W_lat = get_W(sp_lat)
      # cond = np.all(get_W(sp_nup) == get_W(sp_lat))
      cond = np.all(Y_nup[:,t] == Y_lat[:,t])

      if cond == False:
        print "\n what t", t
        print Y_nup[:,t]
        print Y_lat[:,t]
        # print(np.sign(W_nup - get_W(sp_nup)) * W_nup - get_W(sp_nup) )
        print(np.amax(W_nup - W_lat))
        print(np.amin(W_nup - W_lat))
        # print(np.sign(W_lat - get_W(sp_lat)) * W_lat - get_W(sp_lat) )

      assert(cond == True)

        # print get_W(sp_nup) - get_W(sp_lat)
      # assert(np.all(Y_nup[:,t] == Y_lat[:,t]))
      










  def test_feedforward_weight_update(self):
    """
    Check if the Hebbian-like update works as expected.
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

    pooler = LateralPooler(input_size=m, output_size=n, seed=1, inc_dec_ratio = ratio)

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



if __name__ == "__main__":
  unittest.main()



