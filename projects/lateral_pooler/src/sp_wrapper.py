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
from nupic.algorithms.spatial_pooler import SpatialPooler 
import numpy as np


class SpatialPoolerWrapper(SpatialPooler):
    """
    This is a wrapper for the spatial pooler. 
    It just collects more statistics, otherwise behaves the exact same.
    """
    def __init__(self, **args):
        super(SpatialPoolerWrapper, self).__init__(**args)

        n = self._numColumns

        inhibitionArea = ((2*self._inhibitionRadius + 1)** self._columnDimensions.size)
        inhibitionArea = min(n, inhibitionArea)
        density        = float(self._numActiveColumnsPerInhArea) / inhibitionArea
        self.sparsity = density
        self.avgActivityPairs     = np.ones((n,n))*(density**2)
        np.fill_diagonal(self.avgActivityPairs, density)


    def compute(self, inputVector, learn, activeArray):
        """
        This method resembles the primary public method of the SpatialPooler class. 
        """
        super(SpatialPoolerWrapper, self).compute(inputVector, learn, activeArray)

        self._updateAvgActivityPairs(activeArray)


    def encode(self, X):
        d = X.shape[1]
        n = self._numColumns
        Y = np.zeros((n,d))
        for t in range(d):
            self.compute(X[:,t], False, Y[:,t])
            
        return Y

    def _updateAvgActivityPairs(self, activeArray):
        n, m   = self.shape
        Y      = activeArray.reshape((n,1))
        beta = 1.0 - 1.0/self._dutyCyclePeriod

        # period = self._dutyCyclePeriod
        # if (period > self._iterationNum):
          # period = self._iterationNum
        Q = np.dot(Y, Y.T) 

        self.avgActivityPairs = beta*self.avgActivityPairs + (1-beta)*Q

        # self.avgActivityPairs = self._updateDutyCyclesHelper(
        #                           self.avgActivityPairs,
        #                           Q,
        #                           period)


    @property
    def code_weight(self):
        return self._numActiveColumnsPerInhArea

    @property
    def feedforward(self):
        m = self._numInputs
        n = self._numColumns
        W = np.zeros((n, m))
        for i in range(self._numColumns):
            self.getPermanence(i, W[i, :])

        return W

    @property
    def shape(self):
        return self._numColumns, self._numInputs

    @property
    def avg_activity_pairs(self):
        return self.avgActivityPairs

        