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
               wrapAround=True,
               regularColumnActivationPeriod=100,
               miniBurstColumnActivationPeriod=500):
    """
    Please see spatial_pooler.py in NuPIC super class parameter descriptions.

    Class-specific parameters:
    -------------------------------------

    @param regularColumnActivationPeriod: The maximum number of timesteps that a
      column activated by only (unpredicted) active cell input will remain
      active in the Union SDR in the absence of any further active cell input.

    @param miniBurstColumnActivationPeriod: The maximum number of timesteps that
      a column activated by predicted active cell input will remain
      active in the Union SDR in the absence of any further predicted
      active cell input.
    """

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

    self._regularColumnActivationPeriod = regularColumnActivationPeriod
    self._miniBurstColumnActivationPeriod = miniBurstColumnActivationPeriod
    self._poolingActivation = numpy.zeros(self._numColumns, dtype="int32")
    self._unionSDR = []


  def reset(self):
    """
    Reset the state of the Union Pooler.
    """

    # Reset Union Pooler fields
    self._poolingActivation = numpy.zeros(self._numColumns, dtype="int32")
    self._unionSDR = []

    # Reset Spatial Pooler fields
    self._overlapDutyCycles = numpy.zeros(self._numColumns, dtype=REAL_DTYPE)
    self._activeDutyCycles = numpy.zeros(self._numColumns, dtype=REAL_DTYPE)
    self._minOverlapDutyCycles = numpy.zeros(self._numColumns, dtype=REAL_DTYPE)
    self._minActiveDutyCycles = numpy.zeros(self._numColumns, dtype=REAL_DTYPE)
    self._boostFactors = numpy.ones(self._numColumns, dtype=REAL_DTYPE)


  def compute(self, activeCellInput, predictedActiveCellInput, burstingColumns,
              learn):
    """
    Computes one cycle of the Union Pooler algorithm.
    """
    assert (numpy.size(activeCellInput) == self._numInputs)
    assert (numpy.size(predictedActiveCellInput) == self._numInputs)
    self._updateBookeepingVars(learn)

    overlapActiveCells = self._calculateOverlap(activeCellInput)
    overlapPredActiveCells = self._calculateOverlap(predictedActiveCellInput)
    combinedOverlap = overlapActiveCells + overlapPredActiveCells

    # Apply boosting when learning is on
    if learn:
      boostedOverlaps = self._boostFactors * combinedOverlap
    else:
      boostedOverlaps = combinedOverlap

    # Apply inhibition to determine the winning columns
    activeColumns = self._inhibitColumns(boostedOverlaps)

    if learn:
      # TODO Study TP _adaptSynapses to guide this one
      # self._adaptSynapses(activeCellInput, activeColumns)
      self._updateDutyCycles(combinedOverlap, activeColumns)
      self._bumpUpWeakColumns()
      self._updateBoostFactors()
      if self._isUpdateRound():
        self._updateInhibitionRadius()
        self._updateMinDutyCycles()

    # Update and return the Union SDR

    # Decrement activation of all union SDR cells
    self._poolingActivation[self._unionSDR] -= 1

    # Set activation of cells receiving input from active cells
    # TODO Debug and understand this calculation
    stuff = numpy.where(overlapActiveCells[activeColumns] > 0)[0]
    activeColsFromActiveCells = activeColumns[stuff]
    self._poolingActivation[
      activeColsFromActiveCells] = self._regularColumnActivationPeriod

    # Reset activation of cells receiving predicted input. This ordering assumes
    # the activation period due to predicted active cells will be
    # greater than that of standard input
    stuff = numpy.where(overlapPredActiveCells[activeColumns] > 0)[0]
    activeColsFromPredActiveCells = activeColumns[stuff]
    self._poolingActivation[
      activeColsFromPredActiveCells] = self._miniBurstColumnActivationPeriod

    self._unionSDR = self._poolingActivation.nonzero()[0]
    return self._unionSDR
