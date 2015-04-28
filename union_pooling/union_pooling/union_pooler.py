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

import numpy

from nupic.bindings.math import (SM32 as SparseMatrix,
                                 SM_01_32_32 as SparseBinaryMatrix)

from nupic.research.spatial_pooler import SpatialPooler



REAL_DTYPE = numpy.float32



class UnionPooler(SpatialPooler):
  """
  Experimental Union Pooler Python implementation. The Union Pooler builds a
  "union SDR" of the most recent sets of active columns. It is driven by
  active-cell input and, more strongly, by predictive-active cell input. The
  latter is more likely to produce active columns. Such winning columns will
  also tend to persist longer in the union SDR.
  """


  def __init__(self,
               inputDimensions=[32,32],
               columnDimensions=[64,64],
               potentialRadius=16,
               potentialPct=0.9,
               globalInhibition=True,
               localAreaDensity=-1.0,
               numActiveColumnsPerInhArea=20.0,
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
               activeOverlapWeight = 1.0,
               predictedActiveOverlapWeight = 1.0,
               activePoolingPeriod=100,
               predictedActivePoolingPeriod=1000):
    """
    Please see spatial_pooler.py in NuPIC for super class parameter
    descriptions.

    Class-specific parameters:
    -------------------------------------

    @param activeOverlapWeight: A multiplicative weight applied to
    the overlap between connected synapses and active-cell input

    @param predictedActiveOverlapWeight: A multiplicative weight applied to
    the overlap between connected synapses and predicted-active-cell input

    @param activePoolingPeriod: The maximum number of timesteps that a
      column activated by only (unpredicted) active-cell input will remain
      active in the Union SDR in the absence of any further active-cell input.

    @param predictedActivePoolingPeriod: The maximum number of timesteps that
      a column activated by predicted-active-cell input will remain
      active in the Union SDR in the absence of any further
      predicted-active-cell input.
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

    self._activeOverlapWeight = activeOverlapWeight
    self._predictedActiveOverlapWeight = predictedActiveOverlapWeight
    self._activePoolingPeriod = activePoolingPeriod
    self._predictedActivePoolingPeriod = predictedActivePoolingPeriod
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


  def compute(self, activeInput, predictedActiveInput, burstingColumns,
              learn):
    """
    Computes one cycle of the Union Pooler algorithm.
    """
    assert (numpy.size(activeInput) == self._numInputs)
    assert (numpy.size(predictedActiveInput) == self._numInputs)
    self._updateBookeepingVars(learn)

    activeOverlap = self._calculateOverlap(activeInput)
    predictedActiveOverlap = self._calculateOverlap(predictedActiveInput)
    totalOverlap = (activeOverlap * self._activeOverlapWeight  +
                    predictedActiveOverlap * self._predictedActiveOverlapWeight)

    # Apply boosting when learning is on
    if learn:
      boostedOverlaps = self._boostFactors * totalOverlap
    else:
      boostedOverlaps = totalOverlap

    # Apply inhibition to determine the winning columns
    activeColumns = self._inhibitColumns(boostedOverlaps)

    if learn:
      # TODO Follow TP' _adaptSynapses to guide this one
      # self._adaptSynapses(activeCellInput, activeColumns)
      self._updateDutyCycles(totalOverlap, activeColumns)
      self._bumpUpWeakColumns()
      self._updateBoostFactors()
      if self._isUpdateRound():
        self._updateInhibitionRadius()
        self._updateMinDutyCycles()

    # Update and return the Union SDR

    # Decrement activation of all union SDR cells
    self._poolingActivation[self._unionSDR] -= 1

    # Set activation of cells receiving input from active cells
    columnIndices = numpy.where(activeOverlap[activeColumns] > 0)[0]
    activeColsFromActiveCells = activeColumns[columnIndices]
    self._poolingActivation[
      activeColsFromActiveCells] = self._activePoolingPeriod

    # Reset activation of cells receiving predicted input. This ordering assumes
    # the activation period due to predicted active cells will be greater than
    # that of standard input
    columnIndices = numpy.where(predictedActiveOverlap[activeColumns] > 0)[0]
    activeColsFromPredActiveCells = activeColumns[columnIndices]
    self._poolingActivation[
      activeColsFromPredActiveCells] = self._predictedActivePoolingPeriod

    self._unionSDR = self._poolingActivation.nonzero()[0]
    return self._unionSDR
