# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""
Experimental Union Pooler Python implementation.
"""

import numpy

from nupic.bindings.math import (SM32 as SparseMatrix,
                                 SM_01_32_32 as SparseBinaryMatrix)

from nupic.research.spatial_pooler import SpatialPooler

# TODO: Look into this!
REAL_DTYPE = numpy.float32



class UnionPooler(SpatialPooler):
  """

  """

  def __init__(self,
               inputDimensions=[32,32],
               columnDimensions=[64,64],
               potentialRadius=16,
               potentialPct=0.9,
               globalInhibition=True,
               localAreaDensity=-1.0,
               numActiveColumnsPerInhArea=20.0, # Is this even used?
               stimulusThreshold=2,
               synPermInactiveDec=0.01,
               synPermActiveInc=0.03,
               synPermConnected=0.3,
               minPctOverlapDutyCycle=0.001,
               minPctActiveDutyCycle=0.001,
               dutyCyclePeriod=1000,
               maxBoost=1.0,
               seed=42,
               spVerbosity=0,
               wrapAround=True):

    # TODO: SDR Union length parameter
    # TODO: Strength of mini-burst parameter; regular input param

    super(UnionPooler, self).__init__(inputDimensions,
                                      columnDimensions,
                                      potentialRadius,
                                      potentialPct,
                                      globalInhibition,
                                      localAreaDensity,
                                      numActiveColumnsPerInhArea,
                                      stimulusThreshold,
                                      synPermInactiveDec,
                                      synPermActiveInc,
                                      synPermConnected,
                                      minPctOverlapDutyCycle,
                                      minPctActiveDutyCycle,
                                      dutyCyclePeriod,
                                      maxBoost,
                                      seed,
                                      spVerbosity,
                                      wrapAround)

    self._poolingActivation = numpy.zeros(self._numColumns, dtype="int32")


  def reset(self):
    """
    Reset the state of the Union Pooler.
    """
    pass


  def compute(self, learning):
    """
    Computes one cycle of the Union Pooler algorithm.

    """

    # Perform Spatial Pooling
    #
