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
import itertools
from nupic.algorithms.spatial_pooler import SpatialPooler
from htmresearch.support.lateral_pooler.utils import get_permanence_vals as get_W



class LateralPoolerTest(unittest.TestCase):
  """
  Simplistic tests of the experimental lateral pooler implementation.
  """

  def test_whether_is_the_same_as_spatial_pooler(self):
    """
    Naive reality check for the encoding function 
    of the lateral pooler implementation.
    """
    n = 1024
    m = 784
    d = 100
    w = 20

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
        "seed": 1936 }

    params_lat = params_nup.copy()
    params_lat["lateralLearningRate"]  = 0.0
    params_lat["enforceDesiredWeight"] = False

    sp_nup = SpatialPooler(**params_nup)
    sp_lat = LateralPooler(**params_lat)


    for t in range(d):
      sp_nup.compute(X[:,t], False, Y_nup[:,t])
      sp_lat.compute(X[:,t], False, Y_lat[:,t])
    
    self.assertTrue(np.all(Y_nup == Y_lat), 
      "Produces wrong output even without learning.")


    for t in range(d):
      sp_nup.compute(X[:,t], True, Y_nup[:,t])
      sp_lat.compute(X[:,t], True, Y_lat[:,t])

    self.assertTrue(np.all(Y_nup == Y_lat), 
      "Wrong outputs, something diverges during learning.")

    W_nup = get_W(sp_nup)
    W_lat = get_W(sp_lat)
    self.assertTrue(np.all(W_nup == W_lat), 
      "Wrong synaptic weights, something diverges during learning.")

      


if __name__ == "__main__":
  unittest.main()



