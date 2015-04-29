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

    @param maxUnionActivity: Maximum percentage of cells allowed to be in
    union SDR

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

    activeOverlaps = self._calculateOverlap(activeInput)
    predictedActiveOverlaps = self._calculateOverlap(predictedActiveInput)
    totalOverlap = (activeOverlaps * self._activeOverlapWeight  +
                    predictedActiveOverlaps *
                    self._predictedActiveOverlapWeight)

    # Apply boosting when learning is on
    if learn:
      boostedOverlaps = self._boostFactors * totalOverlap
    else:
      boostedOverlaps = totalOverlap

    # Apply inhibition to determine the winning cells
    activeCells = self._inhibitColumns(boostedOverlaps)

    if learn:
      self._adaptSynapses(activeInput, predictedActiveInput, activeCells)
      self._updateDutyCycles(totalOverlap, activeCells)
      self._bumpUpWeakColumns()
      self._updateBoostFactors()
      if self._isUpdateRound():
        self._updateInhibitionRadius()
        self._updateMinDutyCycles()

    # Update and return the Union SDR

    # Decrement pooling activation of all cells
    self._poolingActivation -= 1
    self._poolingActivation[numpy.where(self._poolingActivation < 0)] = 0

    # Set activation of cells receiving input from active cells
    columnIndices = numpy.where(activeOverlaps[activeCells] > 0)[0]
    activeCellsFromActiveInputs = activeCells[columnIndices]
    self._poolingActivation[activeCellsFromActiveInputs] += (
      activeOverlaps[activeCellsFromActiveInputs])

    # Reset activation of cells receiving predicted input. This ordering assumes
    # that the pooling period due to predicted active input will be greater than
    # the period due to active input
    columnIndices = numpy.where(predictedActiveOverlaps[activeCells] > 0)[0]
    activeCellsFromPredActiveInputs = activeCells[columnIndices]
    self._poolingActivation[activeCellsFromPredActiveInputs] += (
      predictedActiveOverlaps[activeCellsFromPredActiveInputs])

    potentialUnionSDR = self._poolingActivation.nonzero()[0].sort()
    self._unionSDR = potentialUnionSDR[0 : self._numColumns *
                                           self._maxUnionActivity]
    return self._unionSDR


  def _adaptSynapses(self, activeInput, predictedActiveInput, activeCells):
    """
    This is the synaptic learning method for the Union Pooler. It updates
    the permanence of synapses based on the active and predicted-active input to
    the Union Pooler. For each active Union Pooler cell, its synapses'
    permanences are updated as follows:

    1. if pre-synaptic input is ON due to a correctly predicted cell,
       increase permanence by _synPredictedInc
    2. else if input is ON due to an active cell, increase permanence by
       _synPermActiveInc
    3. else input is OFF, decrease permanence by _synPermInactiveDec

    Parameters:
    ----------------------------
    activeInput:    a numpy array whose ON bits represent the active cells from
                    Temporal Memory

    predictedActiveInput: a numpy array with numInputs elements. A 1 indicates
                          that this input was a correctly predicted cell in
                          Temporal Memory

    activeCells:    an array containing the indices of the cells that
                    survived the Union Pooler inhibition step
    """
    activeInputIndices = numpy.where(activeInput > 0)[0]
    predictedActiveInputIndices = numpy.where(predictedActiveInput > 0)[0]
    permanenceChanges = numpy.zeros(self._numInputs)

    # Decrement connections from inactive TM cell -> active Union Pooler cell
    permanenceChanges.fill(-1 * self._synPermInactiveDec)

    # Increment connections from active TM cell -> active Union Pooler cell
    permanenceChanges[activeInputIndices] = self._synPermActiveInc

    # Increment connections from correctly predicted TM cell -> active Union
    # Pooler cell
    permanenceChanges[predictedActiveInputIndices] = self._synPredictedInc

    if self._spVerbosity > 4:
      print "\n============== _adaptSynapses ======"
      print "Active input indices:", activeInputIndices
      print "Predicted-active input indices:", predictedActiveInputIndices
      print "\n============== _adaptSynapses ======\n"

    for i in activeCells:
      # Get the permanences of the synapses of Union Pooler cell i
      permanence = self._permanences.getRow(i)

      # Only consider connections in column's potential pool (receptive field)
      maskPotential = numpy.where(self._potentialPools.getRow(i) > 0)[0]
      permanence[maskPotential] += permanenceChanges[maskPotential]
      self._updatePermanencesForColumn(permanence, i, raisePerm=False)
