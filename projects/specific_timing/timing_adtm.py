from itertools import izip, count
import string
import numpy as np
from nupic.encoders import ScalarEncoder
from projects.specific_timing.apical_dependent_sequence_timing_memory import ApicalDependentSequenceTimingMemory as TM


class TimingADTM(object):

  """
  Use the (vanilla) apical dependent sequence memory for
  learning sequences in time
  """

  def __init__(self, numColumns, numActiveCells, numTimeColumns,
               numActiveTimeCells, numTimeSteps):

    """
    :param numColumns:
    :param numActiveCells:
    :param numTimeColumns:
    :param numActiveTimeCells:
    :param numTimeSteps:
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
    for _ in range(numIter):

      for item in enumerate(trainSeq):

        activeColumns = self.letterIndices[self.letters.index(item[1][0])]

        timeStamp = int(item[1][1] % 21)
        apicalInput = self.timeIndices[timeStamp]

        self.adtm.compute(activeColumns,
                          apicalInput=apicalInput,
                          apicalGrowthCandidates=None,
                          learn=True)

    self.adtm.reset()
    print '{:<30s}{:<10s}'.format('Train Sequence:', trainSeq)


  def infer(self, testSeq):

    self.resetResults()
    tempoFactor = 1

    for item in enumerate(testSeq):

      activeColumns = self.letterIndices[self.letters.index(item[1][0])]

      timeStamp = item[1][1]
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
