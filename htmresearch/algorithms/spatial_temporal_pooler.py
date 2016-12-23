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



class SpatialTemporalPooler(object):


  def __init__(self,
               inputDimensions=[32,32],
               columnDimensions=[64,64],
               potentialRadius=16,
               potentialPct=0.9,
               globalInhibition=True,
               localAreaDensity=-1.0,
               numActiveColumnsPerInhArea=10.0,
               stimulusThreshold=2,
               synPermInactiveDec=0.01,
               synPermActiveInc=0.03,
               synPredictedInc=0.5,
               synPermConnected=0.3,
               minPctOverlapDutyCycle=0.001,
               dutyCyclePeriod=1000,
              _boostStrength=0.0,
               useBurstingRule = False,
               usePoolingRule = True,
               poolingLife = 1000,
               poolingThreshUnpredicted = 0.0,
               initConnectedPct = 0.2,
               seed=-1,
               spVerbosity=0,
               wrapAround=True):
    self.inputDimensions = inputDimensions
    self.columnDimensions = columnDimensions
    self.synPermInactiveDec = synPermInactiveDec
    self.synPermActiveInc = synPermActiveInc
    self.synPredictedInc = synPredictedInc

    self._permanences = self._initPermanences()
    self._connectedCounts = self._computeConnectedCounts()

    self.reset()


  def compute(self,
              inputVector,
              learn,
              activeArray,
              burstingColumns,
              predictedCells):
    overlaps = self._computeOverlaps(inputVector, predictedCells)
    activeColumns = self._inhibitColumns(overlaps)

    if learn:
      self._adaptPermanences(activeColumns, inputVector, predictedCells)

    return activeColumns


  def reset(self):
    self._overlaps = numpy.zeros(self.getNumColumns())


  def getNumInputs(self):
    return numpy.product(self.inputDimensions)


  def getNumColumns(self):
    return numpy.product(self.columnDimensions)


  def getPermanence(self, column, permanence):
    permanence[:] = self._permanences[column]


  def _initPermanences(self):
    size = [self.getNumColumns(), self.getNumInputs()]
    permanences = numpy.random.normal(0.5, 0.5, size)
    permanences[permanences < 0] = 0
    permanences[permanences > 1] = 1

    return permanences


  def _connectedPermanences(self):
    return self._permanences > 0.5


  def _computeConnectedCounts(self):
    return numpy.sum(self._permanences, axis=1)


  def _computeOverlaps(self, inputVector, predictedCells):
    scores = numpy.array(inputVector)
    scores[predictedCells == 1] += 10
    overlaps = numpy.dot(self._permanences, numpy.transpose(scores))

    overlaps = self._overlaps * .32 + overlaps
    self._overlaps = overlaps

    return overlaps


  def _inhibitColumns(self, overlaps):
    numActive = int(0.02 * self.getNumColumns())
    return numpy.argpartition(overlaps, -numActive)[-numActive:]


  def _getSubsetArray(self, array, percent=0.3):
    subset = numpy.array(array)
    mask = numpy.random.choice(self.getNumInputs(),
                               int(self.getNumInputs() * (1 - percent)),
                               replace=False)
    subset[mask] = 0
    return subset


  def _adaptPermanences(self, activeColumns, inputVector, predictedCells):
    for column in activeColumns:
      delta = numpy.zeros(self.getNumInputs())
      delta[predictedCells == 1] = self.synPredictedInc
      permanences = self._permanences[column]
      total = permanences.sum()
      permanences += delta
      permanences /= (permanences.sum() / total)
      self._permanences[column] = permanences

    self._connectedCounts = self._computeConnectedCounts()
