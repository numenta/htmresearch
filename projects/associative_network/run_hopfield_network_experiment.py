# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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
"""
Experiment with associative network that uses SDRs

Two experiments are included in this script
1. Capacity experiment: How many unique items can a network store such that
each item can be reliably retrieved?

2. Simultaneously retrieve multiple items by relaxing the sparsity
"""
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
plt.ion()
plt.close('all')

class hyperColumnNetwork(object):
  def __init__(self,
               numHyperColumn,
               numNeuronPerHyperColumn,
               numActiveNeuronPerHyperColumn,
               numInputs,
               minThreshold=0,
               matchThreshold=10):
    self.numHyperColumn = numHyperColumn
    self.numNeuronPerHyperColumn = numNeuronPerHyperColumn
    self.numActiveNeuronPerHyperColumn = numActiveNeuronPerHyperColumn
    self.numInputs = numInputs
    self.minThreshold = minThreshold
    self.matchThreshold = matchThreshold
    self.numNeuronTotal = numHyperColumn * numNeuronPerHyperColumn

    # initialize weight matrix
    self.weightFF = np.eye(self.numNeuronTotal, numInputs)
    self.weightRecurrent = np.zeros((self.numNeuronTotal, self.numNeuronTotal))


  def initializeObjectSDRs(self, numObjects, seed=1):
    # initialize object SDRs in HC
    np.random.seed(seed)
    objectSDRActiveBits = []
    for i in range(numObjects):
      objectSDRActiveBits.append([])
      for j in range(self.numHyperColumn):
        randomCells = np.random.permutation(range(self.numNeuronPerHyperColumn))
        objectSDRActiveBits[i].append(
          randomCells[:self.numActiveNeuronPerHyperColumn])

    return objectSDRActiveBits


  def memorizeObjectSDRs(self, objectSDRActiveBits):
    numObjects = len(objectSDRActiveBits)
    # initialize recurrent connections
    self.weightRecurrent = np.zeros((self.numNeuronTotal, self.numNeuronTotal))
    for i in range(numObjects):
      offset = 0
      objectSDR = np.zeros((self.numNeuronTotal, 1))
      for j in range(self.numHyperColumn):
        objectSDR[offset+objectSDRActiveBits[i][j], 0] = 1
        offset += self.numNeuronPerHyperColumn
      self.weightRecurrent += np.dot(objectSDR, np.transpose(objectSDR))

    for i in range(self.numNeuronTotal):
      self.weightRecurrent[i, i] = 0


  def run(self, initialState, feedforwardInputs):
    """
    Run network for multiple steps
    :param initialState:
    :param feedforwardInputs: list of feedforward inputs
    :return: list of active cell indices over time
    """
    currentState = initialState
    activeStateHistory = [np.where(initialState > 0)[0]]
    numStep = len(feedforwardInputs)
    for i in range(numStep):
      currentState = self.runSingleStep(currentState,
                                        feedforwardInputs[i])
      activeStateHistory.append([np.where(currentState > 0)[0]])
    return activeStateHistory


  def runSingleStep(self,
                    previousState,
                    feedforwardInputs):
    """
    Run network for one step
    :param previousState: a (Ncell, 1) numpy array of network states
    :param maxNumberOfActiveCellsPerColumn: maximum number of active cells per
          column
    :return: newState
    """
    print "previous activeCells ", np.sort(np.where(previousState>0)[0])
    feedforwardInputOverlap = np.dot(self.weightFF, feedforwardInputs)
    lateralInputOverlap = np.dot(self.weightRecurrent, previousState)
    totalInput = feedforwardInputOverlap + lateralInputOverlap
    print "feedforwardInputOverlap: ", np.sort(np.where(feedforwardInputOverlap>0)[0])
    # cells with active feedforward zone
    feedforwardActive = feedforwardInputOverlap > self.minThreshold

    # cells with active distal zone (that receives lateral connections)
    lateralActive = lateralInputOverlap > self.minThreshold

    # cells with both active feedforward zone and lateral zone
    strongActive = np.logical_and(feedforwardActive, lateralActive)

    newState = np.zeros((self.numNeuronTotal, 1))
    offset = 0

    for i in range(self.numHyperColumn):
      numberOfStrongActiveCellsInColumn = np.sum(
        strongActive[offset:offset+self.numNeuronPerHyperColumn])
      print "numberOfStrongActiveCellsInColumn: ", numberOfStrongActiveCellsInColumn
      if numberOfStrongActiveCellsInColumn > self.matchThreshold:
        self.numActiveNeuronPerHyperColumn = self.numActiveNeuronPerHyperColumn/2

      w = self.numActiveNeuronPerHyperColumn

      cellIdx = np.argsort(totalInput[offset:offset+self.numNeuronPerHyperColumn, 0])
      activeCells = cellIdx[-w:]

      activeCells = activeCells[np.where(
        totalInput[activeCells] > self.minThreshold)[0]]
      newState[offset + activeCells] = 1
      print "activeCells ", np.sort(activeCells)
      offset += self.numNeuronPerHyperColumn

    return newState



def convertActiveCellsToSDRs(activeStateHistory, numCells):
  """
  Convert list of active cell indices to a list of SDRs
  :param activeStateHistory: list of active cell indices per step
  :param numCells: total number of cells
  :return: sdrHistory numpy array of (numStep, numCells)
  """
  numStep = len(activeStateHistory)
  sdrHistory = np.zeros((numStep, numCells))
  for i in range(numStep):
    sdrHistory[i, activeStateHistory[i]] = 1
  return sdrHistory



def stripSDRHistoryForDisplay(sdrHistory, removePortion=0.5):
  """
  Strip SDR History (remove unused bits) for display purpose
  :param sdrHistory:
  :return: displayBitIndex
  """
  sdrHistorySum = np.sum(sdrHistory, axis=0)
  unusedBitIndices = np.where(sdrHistorySum == 0)[0]
  usedBitIndices = np.where(sdrHistorySum > 1)[0]

  numUnusedBitKeep = int(len(unusedBitIndices) * (1-removePortion))
  unusedBitIndices = np.random.permutation(unusedBitIndices)
  unusedBitIndices = unusedBitIndices[:numUnusedBitKeep]

  displayBitIndex = np.concatenate((usedBitIndices, unusedBitIndices))
  displayBitIndex = np.sort(displayBitIndex)
  return displayBitIndex



def generateSDRforDisplay(numNeuron, activeBits, displayBitIndex):
  sdrForDisplay = np.zeros((1, numNeuron))
  sdrForDisplay[0, activeBits] = 1
  sdrForDisplay = np.matlib.repmat(sdrForDisplay[:, displayBitIndex], 10, 1)
  return sdrForDisplay



def runSingleExperiment(numObjects, numBitNoise, seed=10):
  np.random.seed(seed)
  hcNet = hyperColumnNetwork(numHyperColumn=1,
                             numNeuronPerHyperColumn=1024,
                             numActiveNeuronPerHyperColumn=20,
                             numInputs=1024)

  objectSDRActiveBits = hcNet.initializeObjectSDRs(numObjects=numObjects,
                                                   seed=seed)
  hcNet.memorizeObjectSDRs(objectSDRActiveBits)

  objectIDTest = np.random.choice(numObjects, 100)
  finalOverlapList = []

  for objectID in objectIDTest:
    initialState = np.zeros((hcNet.numNeuronTotal, 1))
    randomCells = np.random.permutation(range(hcNet.numNeuronTotal))
    initialState[objectSDRActiveBits[objectID][0][:(20-numBitNoise)]] = 1
    initialState[randomCells[:numBitNoise]] = 1
    feedforwardInputs = [np.zeros((hcNet.numNeuronTotal, 1))] * 5
    activeStateHistory = hcNet.run(initialState, feedforwardInputs)

    sdrHistory = convertActiveCellsToSDRs(activeStateHistory,
                                          hcNet.numNeuronTotal)

    initialActiveCells = np.where(sdrHistory[0, :] > 0)[0]
    finalActiveCells = np.where(sdrHistory[-1, :] > 0)[0]
    finalOverlap = len(
      set(objectSDRActiveBits[objectID][0]).intersection(finalActiveCells))
    initialOverlap = len(
      set(objectSDRActiveBits[objectID][0]).intersection(initialActiveCells))

    finalOverlapList.append(finalOverlap)

  return finalOverlapList



def capacityExperiment():
  numObjectList = np.linspace(start=100, stop=2000, num=10).astype('int')

  numBitNoiseList = [2, 4, 8, 10, 15]
  numRpts = 3
  avgFinalOverlap = np.zeros(
    (numRpts, len(numBitNoiseList), len(numObjectList)))
  for i in range(len(numBitNoiseList)):
    for j in range(len(numObjectList)):
      for rpt in range(3):
        print "run experiment with object # {} noise # {} rpt {}".format(
          numObjectList[j], numBitNoiseList[i], rpt
        )
        finalOverlap = runSingleExperiment(numObjectList[j],
                                           numBitNoiseList[i], seed=rpt)
        avgFinalOverlap[rpt, i, j] = (np.mean(finalOverlap))

  plt.figure()
  finalOverlaps = np.mean(avgFinalOverlap, 0)
  legendList = []
  for i in range(len(numBitNoiseList)):
    plt.plot(numObjectList, finalOverlaps[i, :])
    legendList.append("noise = {}".format(numBitNoiseList[i]))
  plt.legend(legendList)
  plt.plot([140, 140], [0, 20], 'k--')
  plt.xlabel('Number of Object')
  plt.ylabel('Overlap(retrieved sdr, original sdr)')
  plt.savefig('capacity_experiment_result.pdf')



def retrieveMultipleItems():
  hcNet = hyperColumnNetwork(numHyperColumn=1,
                             numNeuronPerHyperColumn=1024,
                             numActiveNeuronPerHyperColumn=20,
                             numInputs=1024,
                             minThreshold=0)
  numObjects = 100
  objectSDRActiveBits = hcNet.initializeObjectSDRs(numObjects=numObjects,
                                                   seed=42)
  hcNet.memorizeObjectSDRs(objectSDRActiveBits)
  hcNet.numActiveNeuronPerHyperColumn = 40

  objectID1 = 1
  objectID2 = 2
  ambiguousInput = np.zeros((hcNet.numNeuronTotal, 1))
  ambiguousInput[objectSDRActiveBits[objectID1][0][:10]] = 10
  ambiguousInput[objectSDRActiveBits[objectID2][0][:10]] = 10

  nStep = 20
  feedforwardInputs = [ambiguousInput]
  for i in range(1, nStep):
    feedforwardInputs.append(np.zeros((hcNet.numNeuronTotal, 1)))
  feedforwardInputs[10][objectSDRActiveBits[objectID1][0]] = 1

  initialState = np.zeros((hcNet.numNeuronTotal, 1))
  # initialState = ambiguousInput

  activeStateHistory = hcNet.run(initialState, feedforwardInputs)

  sdrHistory = convertActiveCellsToSDRs(activeStateHistory,
                                        hcNet.numNeuronTotal)
  displayBitIndex = stripSDRHistoryForDisplay(sdrHistory, removePortion=0.9)

  initialActiveCells = np.where(sdrHistory[0, :] > 0)[0]
  print initialActiveCells
  finalActiveCells = np.where(sdrHistory[-1, :] > 0)[0]

  initialOverlap1 = len(
    set(objectSDRActiveBits[objectID1][0]).intersection(initialActiveCells))

  initialOverlap2 = len(
    set(objectSDRActiveBits[objectID2][0]).intersection(initialActiveCells))

  finalOverlap1 = len(
    set(objectSDRActiveBits[objectID1][0]).intersection(finalActiveCells))

  finalOverlap2 = len(
    set(objectSDRActiveBits[objectID2][0]).intersection(finalActiveCells))

  print "Initial overlap with object SDR 1: {}".format(initialOverlap1)
  print "Initial overlap with object SDR 2: {}".format(initialOverlap2)

  print "Final overlap with object SDR 1: {}".format(finalOverlap1)
  print "Final overlap with object SDR 2: {}".format(finalOverlap2)

  fig, ax = plt.subplots(nrows=4, ncols=1)
  object1SDR = generateSDRforDisplay(hcNet.numNeuronTotal,
                                     objectSDRActiveBits[objectID1],
                                     displayBitIndex)
  object2SDR = generateSDRforDisplay(hcNet.numNeuronTotal,
                                     objectSDRActiveBits[objectID2],
                                     displayBitIndex)
  querySDR = np.matlib.repmat(np.transpose(ambiguousInput[displayBitIndex]), 10, 1)
  ax[0].imshow(object1SDR, cmap='gray')
  ax[0].set_title('SDR for Object A')
  ax[1].imshow(object2SDR, cmap='gray')
  ax[1].set_title('SDR for Object B')
  ax[2].imshow(querySDR, cmap='gray')
  ax[2].set_title('query SDR')
  ax[3].imshow(sdrHistory[:, displayBitIndex], cmap='gray')
  ax[3].set_title('Network states over time')
  plt.savefig('figures/retrieveMultipleItems.pdf')



if __name__ == "__main__":
  retrieveMultipleItems()
  #
  # hcNet = hyperColumnNetwork(numHyperColumn=3,
  #                            numNeuronPerHyperColumn=1024,
  #                            numActiveNeuronPerHyperColumn=20,
  #                            numInputs=1024,
  #                            minThreshold=5)
  # numObjects = 10
  # objectSDRActiveBits = hcNet.initializeObjectSDRs(numObjects=numObjects,
  #                                                  seed=42)
  # hcNet.memorizeObjectSDRs(objectSDRActiveBits)
  #
  # initialState = np.zeros((hcNet.numNeuronTotal, 1))
  #
  # objectID1 = 0
  # # objectID2 = 1
  # offset = 0
  # for i in range(hcNet.numHyperColumn):
  #   initialState[offset + objectSDRActiveBits[objectID1][i][:10]] = 1
  #   # initialState[offset + objectSDRActiveBits[objectID2][i][:10]] = 1
  #   offset += hcNet.numNeuronPerHyperColumn
  #
  # activeStateHistory = hcNet.run(initialState, 10, numActiveBit=40)
  # sdrHistory = convertActiveCellsToSDRs(activeStateHistory,
  #                                       hcNet.numNeuronTotal)
  #
  # activationColumn1 = sdrHistory[:, :1024]
  # c = 0
  # initialOverlap1 = len(
  #   set(objectSDRActiveBits[objectID1][c]).intersection(set(np.where(activationColumn1[0, :]>0)[0])))
  #
  # finalOverlap1 = len(
  #   set(objectSDRActiveBits[objectID1][c]).intersection(set(np.where(activationColumn1[-1, :]>0)[0])))
  # set(np.where(initialState > 0)[0])
