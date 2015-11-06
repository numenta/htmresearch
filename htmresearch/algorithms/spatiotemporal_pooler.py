# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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


import numpy
from nupic.research.spatial_pooler import SpatialPooler


REAL_DTYPE = numpy.float32
INT_DTYPE = numpy.int32



class SpatiotemporalPooler(SpatialPooler):
  """
  """


  def __init__(self, 
               historyLength = 5,
               **kwargs):
    """
    """
    super(SpatiotemporalPooler, self).__init__(**kwargs)

    self._historyLength = historyLength
    self._persistence = numpy.zeros(self._numInputs)
    self._activeColumns = numpy.ndarray(self._numColumns)


  def reset(self):
    """
    """
    # Reset Spatriotemporal Pooler fields
    self._persistence[:] = 0

    # Reset Spatial Pooler fields
    self._overlapDutyCycles = numpy.zeros(self._numColumns, dtype=REAL_DTYPE)
    self._activeDutyCycles = numpy.zeros(self._numColumns, dtype=REAL_DTYPE)
    self._minOverlapDutyCycles = numpy.zeros(self._numColumns, dtype=REAL_DTYPE)
    self._minActiveDutyCycles = numpy.zeros(self._numColumns, dtype=REAL_DTYPE)
    self._boostFactors = numpy.ones(self._numColumns, dtype=REAL_DTYPE)
    


  def compute(self, activeInput, predictedActiveInput, learn, unionedInputMonitor=None):
    """
    """
    self._persistence -= 1
    self._persistence = numpy.maximum(self._persistence, 0)

    # For now, all active cells participate in the union
    self._persistence[activeInput.astype(bool)] += self._historyLength

    unionSDR = self._persistence > 0
    
    if unionedInputMonitor is not None:
      unionedInputMonitor[:] = unionSDR
    
    # print unionSDR.shape, self._activeColumns.shape
    
    SpatialPooler.compute(self, unionSDR,
      learn, self._activeColumns)
          
    return self._activeColumns