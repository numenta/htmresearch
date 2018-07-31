# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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

from itertools import izip, count
import string
import numpy as np
from nupic.encoders import ScalarEncoder
from htmresearch.frameworks.specific_timing.apical_dependent_sequence_timing_memory import ApicalDependentSequenceTimingMemory as TM


class TimingADTM(object):

  """
  Uses the ApicalDependentSequenceTimingMemory for learning sequences with specific times.
  Rescales apical input values (times) to recognize sequences sped up/slowed down by a factor of two

  """

  def __init__(self, numColumns, numActiveCells, numTimeColumns,
               numActiveTimeCells, numTimeSteps):

    """
    :param numColumns: Number of minicolumns
    :param numActiveCells: Number of ON bits in SDR encoding of a given letter
    :param numTimeColumns: Number of bits in the time encoder output
    :param numActiveTimeCells: Number of ON bits in SDR encoding of a given timestamp
    :param numTimeSteps: Number of time intervals that can be distinctly represented (clock resolution)
    """

    self.adtm = TM(columnCount=numColumns,
                   apicalInputSize=numTimeColumns,
                   cellsPerColumn=32,
                   activationThreshold=13,
                   reducedBasalThreshold=10,
                   initialPermanence=0.50,
                   connectedPermanence=0.50,
                   minThreshold=10,
                   sampleSize=20,
                   permanenceIncrement=0.1,
                   permanenceDecrement=0.1,
                   basalPredictedSegmentDecrement=0.1,
                   apicalPredictedSegmentDecrement=0.05,
                   maxSynapsesPerSegment=-1,
                   seed=42)

    self.numColumns = numColumns
    self.numActiveCells = numActiveCells
    self.numTimeColumns = numTimeColumns
    self.numActiveTimeCells = numActiveTimeCells
    self.numTimeSteps = numTimeSteps

    self.letters = list(string.ascii_uppercase)
    self.letterIndices = self.encodeLetters()
    self.letterIndexArray = np.array(self.letterIndices)
    self.timeIndices = self.encodeTime()

    self.results = dict()
    self.results['active_cells'] = []
    self.results['predicted_cells'] = []
    self.results['basal_predicted_cells'] = []
    self.results['apical_predicted_cells'] = []

    self.apicalIntersect = []


  def resetResults(self):
    self.results = dict()
    self.results['active_cells'] = []
    self.results['predicted_cells'] = []
    self.results['basal_predicted_cells'] = []
    self.results['apical_predicted_cells'] = []


  def learn(self, trainSeq, numIter):
    """

    :param trainSeq: list of (feature, timestamp) tuples i.e. (('A', 5), ('B', 8), ('C', 12), ('D', 16))
    :param numIter: Number of iterations (in a row) over which a given sequence should be learned
    """
    for _ in range(numIter):

      for item in trainSeq:

        activeColumns = self.letterIndices[self.letters.index(item[0])]

        timeStamp = int(item[1] % 21)
        apicalInput = self.timeIndices[timeStamp]

        self.adtm.compute(activeColumns,
                          apicalInput=apicalInput,
                          apicalGrowthCandidates=None,
                          learn=True)

    self.adtm.reset()
    print '{:<30s}{:<10s}'.format('Train Sequence:', trainSeq)


  def infer(self, testSeq):
    """

    :param testSeq: list of (feature, timestamp) tuples (same format as trainSeq, above)
    """

    self.resetResults()
    tempoFactor = 1

    for item in testSeq:

      activeColumns = self.letterIndices[self.letters.index(item[0])]

      timeStamp = item[1]
      apicalTimestamp = int((timeStamp * tempoFactor) % 21)

      apicalInput = self.timeIndices[apicalTimestamp]

      self.adtm.compute(activeColumns,
                        apicalInput=apicalInput,
                        apicalGrowthCandidates=None,
                        learn=False)

      self.apicalIntersect = np.empty(0)
      # for ii in range(int(round(item[1][1]))):
      for ii in range(apicalTimestamp):
        self.apicalIntersect = np.union1d(self.apicalIntersect, self.adtm.apicalCheck(self.timeIndices[ii]))

      self.results['active_cells'].append(self.adtm.getActiveCells())
      self.results['basal_predicted_cells'].append(self.adtm.getNextBasalPredictedCells())
      self.results['apical_predicted_cells'].append(self.adtm.getNextApicalPredictedCells())
      self.results['predicted_cells'].append(self.adtm.getNextPredictedCells())

      #if (not self.adtm.getNextApicalPredictedCells().any()) & (self.adtm.getNextBasalPredictedCells().any()):
      if (not self.adtm.getNextPredictedCells().any()) & (self.adtm.getNextBasalPredictedCells().any()):

        if self.apicalIntersect.any():
          tempoFactor = tempoFactor * 0.5

        else:
          tempoFactor = tempoFactor * 2

    print '{:<30s}{:<10s}'.format('Test Sequence:', testSeq)
    print '{:<30s}{:<10s}'.format('--------------', '--------------')
    self.displayResults()

    print '{:<30s}{:<10s}'.format('~~~~~~~~~~~~~~', '~~~~~~~~~~~~~~')

    self.adtm.reset()


  def displayResults(self):
    resultLengths = {k: [len(i) for i in self.results[k]] for k in self.results}
    resultLetters = {k: self.letterConverter(self.results[k]) for k in self.results}

    sortOrder = ['active_cells', 'basal_predicted_cells', 'apical_predicted_cells', 'predicted_cells']

    for k in sortOrder:
      print '{:<30s}{:<10s}'.format(k, map(lambda x, y: (x, y), resultLengths[k], resultLetters[k]))


  def letterConverter(self, resultsKey):
    """

    :param resultsKey: results dictionary self.results, indexed by key
    :return: letter(s) corresponding to predicted cell numbers
    """
    convertedLetters = []

    for c in enumerate(resultsKey):
      columnIdx = [int(i / self.adtm.cellsPerColumn) for i in c[1]]

      if not columnIdx:
        convertedLetters.append(['-'])

      else:
        cl = []
        for cell in np.unique(columnIdx):
          cl.append([d2 for d2 in np.where(self.letterIndexArray == cell)[0]])

        convertedLetters.append([self.letters[i] for i in np.unique(cl)])

    return convertedLetters


  def encodeTime(self):

    timeEncoder = ScalarEncoder(n=self.numTimeColumns,
                                w=self.numActiveTimeCells,
                                minval=0,
                                maxval=self.numTimeSteps,
                                forced=True)

    timeArray = np.zeros((self.numTimeSteps, self.numTimeColumns))
    timeIndices = []
    for k in range(self.numTimeSteps):
      timeArray[k, :] = timeEncoder.encode(k)
      idxTimes = [i for i, j in izip(count(), timeArray[k]) if j == 1]
      timeIndices.append(idxTimes)

    return timeIndices

  def encodeLetters(self):
    letterEncoder = ScalarEncoder(n=self.numColumns, w=self.numActiveCells, minval=0, maxval=25)

    numLetters = np.shape(self.letters)[0]
    letterArray = np.zeros((numLetters, self.numColumns))
    letterIndices = []
    for k in range(numLetters):
      letterArray[k, :] = letterEncoder.encode(k)
      idxLetters = [i for i, j in izip(count(), letterArray[k]) if j == 1]
      letterIndices.append(idxLetters)

    return letterIndices

  def debugResults(self):
    return self.results

  def debugLetters(self):
    return self.letterIndexArray
