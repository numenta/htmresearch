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

               # union_pooler.py parameters
               activeOverlapWeight=1.0,
               predictedActiveOverlapWeight=10.0,
               maxUnionActivity=0.20):
    """
    Please see spatial_pooler.py in NuPIC for super class parameter
    descriptions.

    Class-specific parameters:
    -------------------------------------

    @param activeOverlapWeight: A multiplicative weight applied to
    the overlap between connected synapses and active-cell input

    @param predictedActiveOverlapWeight: A multiplicative weight applied to
    the overlap between connected synapses and predicted-active-cell input

    @param maxUnionActivity: Maximum number of active cells allowed in
    union SDR simultaneously in terms of the ratio between the number of active
    cells and the number of total cells
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
    self._maxUnionActivity = maxUnionActivity

    self._maxUnionCells = int(self._numColumns * self._maxUnionActivity)
    self._poolingActivation = numpy.zeros(self._numColumns, dtype=REAL_DTYPE)
    self._unionSDR = []


  def reset(self):
    """
    Reset the state of the Union Pooler.
    """

    # Reset Union Pooler fields
    self._poolingActivation = numpy.zeros(self._numColumns, dtype=REAL_DTYPE)
    self._unionSDR = []

    # Reset Spatial Pooler fields
    self._overlapDutyCycles = numpy.zeros(self._numColumns, dtype=REAL_DTYPE)
    self._activeDutyCycles = numpy.zeros(self._numColumns, dtype=REAL_DTYPE)
    self._minOverlapDutyCycles = numpy.zeros(self._numColumns, dtype=REAL_DTYPE)
    self._minActiveDutyCycles = numpy.zeros(self._numColumns, dtype=REAL_DTYPE)
    self._boostFactors = numpy.ones(self._numColumns, dtype=REAL_DTYPE)


  def compute(self, activeInput, predictedActiveInput, learn):
    """
    Computes one cycle of the Union Pooler algorithm.
    """
    assert numpy.size(activeInput) == self._numInputs
    assert numpy.size(predictedActiveInput) == self._numInputs
    self._updateBookeepingVars(learn)

    # Compute proximal dendrite overlaps with active and active-predicted inputs
    overlapsActive = self._calculateOverlap(activeInput)
    overlapsPredictedActive = self._calculateOverlap(predictedActiveInput)
    totalOverlap = (overlapsActive * self._activeOverlapWeight  +
                    overlapsPredictedActive *
                    self._predictedActiveOverlapWeight)

    if learn:
      boostedOverlaps = self._boostFactors * totalOverlap
    else:
      boostedOverlaps = totalOverlap

    activeCells = self._inhibitColumns(boostedOverlaps)

    if learn:
      self._adaptSynapses(activeInput, activeCells)
      self._updateDutyCycles(totalOverlap, activeCells)
      self._bumpUpWeakColumns()
      self._updateBoostFactors()
      if self._isUpdateRound():
        self._updateInhibitionRadius()
        self._updateMinDutyCycles()

    # Decrement pooling activation of all cells
    self._decayPoolingActivation()

    # Add to the poolingActivation of active Union Pooler cells receiving input
    # from active input cells
    self._addToPoolingActivation(activeCells, overlapsActive)

    # Add to the poolingActivation of active Union Pooler cells receiving
    # active-predicted input cells.
    self._addToPoolingActivation(activeCells, overlapsPredictedActive)

    return self._getMostActiveCells()


  def _decayPoolingActivation(self):
    """
    Decrements pooling activation of all cells
    """
    self._poolingActivation -= 1
    self._poolingActivation[self._poolingActivation < 0] = 0
    return self._poolingActivation


  def _addToPoolingActivation(self, activeCells, overlaps):
    """
    Adds overlaps from specified active cells to cells' pooling
    activation.
    :param activeCells: Indices of those cells winning the inhibition step
    :param overlaps: A current set of overlap values for each cell
    """
    cellIndices = numpy.where(overlaps[activeCells] > 0)[0]
    activeCellsSubset = activeCells[cellIndices]
    self._poolingActivation[activeCellsSubset] += overlaps[activeCellsSubset]
    return self._poolingActivation


  def _getMostActiveCells(self):
    """
    Gets the most active cells in the Union SDR having at least non-zero
    activation.
    :return: a list of cell indices
    """
    potentialUnionSDR = numpy.argsort(
      self._poolingActivation)[::-1][:len(self._poolingActivation)]

    topCells = potentialUnionSDR[0: self._maxUnionCells]
    nonZeroTopCells = self._poolingActivation[topCells] > 0
    self._unionSDR = topCells[nonZeroTopCells]
    return self._unionSDR
