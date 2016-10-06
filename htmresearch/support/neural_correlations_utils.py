# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
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
Common functions used in the neural_correlations project
"""

import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import itertools
try:
    import capnp
except ImportError:
    capnp = None
if capnp:
    from nupic.proto import TemporalMemoryProto_capnp



def randomizeSequence(sequence, symbolsPerSequence, numColumns, sparsity):
  """
  Takes a sequence as input and randomizes a percentage p of it by choosing
  SDRs at random while preserving the remaining invariant.
  
  @param sequence (array) sequence to be randomized
  @return randomizedSequence (array) sequence that contains p percentage of new SDRs
  """
  randomizedSequence = []
  sparseCols = int(numColumns * sparsity)
  p = 0.25 #percentage of symbols to be replaced
  numSymbolsToChange = int(symbolsPerSequence * p)
  symIndices = np.random.permutation(np.arange(symbolsPerSequence))  
  for symbol in range(symbolsPerSequence):
    randomizedSequence.append(sequence[symbol])
  i = 0
  while numSymbolsToChange > 0:
    randomizedSequence[symIndices[i]] = generateRandomSymbol(numColumns, sparseCols)
    i += 1
    numSymbolsToChange -= 1
  return randomizedSequence


def percentOverlap(x1, x2, numColumns):
  """
  Calculates the percentage of overlap between two SDRs
  
  @param x1 (array) SDR
  @param x2 (array) SDR
  @return percentageOverlap (float) percentage overlap between x1 and x2
  """  
  nonZeroX1 = np.count_nonzero(x1)
  nonZeroX2 = np.count_nonzero(x2)
  sparseCols = min(nonZeroX1, nonZeroX2)  
  # transform input vector specifying columns into binary vector
  binX1 = np.zeros(numColumns, dtype="uint32")
  binX2 = np.zeros(numColumns, dtype="uint32")
  for i in range(sparseCols):
    binX1[x1[i]] = 1
    binX2[x2[i]] = 1
  return float(np.dot(binX1, binX2))/float(sparseCols)


def generateRandomSymbol(numColumns, sparseCols):
  """
  Generates a random SDR with sparseCols number of active columns
  
  @param numColumns (int) number of columns in the temporal memory
  @param sparseCols (int) number of sparse columns for desired SDR
  @return symbol (list) SDR
  """
  symbol = list()
  remainingCols = sparseCols
  while remainingCols > 0:
    col = random.randrange(numColumns)
    if col not in symbol:
      symbol.append(col)
      remainingCols -= 1
  return symbol


def generateRandomSequence(numSymbols, numColumns, sparsity):
  """
  Generate a random sequence comprising numSymbols SDRs
  
  @param numSymbols (int) number of SDRs in random sequence
  @param numColumns (int) number of columns in the temporal memory
  @param sparsity (float) percentage of sparsity (real number between 0 and 1)
  @return sequence (array) random sequence generated
  """
  sequence = []
  sparseCols = int(numColumns * sparsity)
  for _ in range(numSymbols):
    sequence.append(generateRandomSymbol(numColumns, sparseCols))
  return sequence


def computePWCorrelations(spikeTrains, removeAutoCorr):
  """
  Computes pairwise correlations from spikeTrains
  
  @param spikeTrains (array) spike trains obtained from the activation of cells in the TM
         the array dimensions are: numCells x timeSteps
  @param removeAutoCorr (boolean) if true, auto-correlations are removed by substracting
         the diagonal of the correlation matrix
  @return corrMatrix (array) numCells x numCells matrix containing the Pearson correlation
          coefficient of spike trains of cell i and cell j
  @return numNegPCC (int) number of negative pairwise correlations (PCC(i,j) < 0)
  """
  numCells = np.shape(spikeTrains)[0]
  corrMatrix = np.zeros((numCells, numCells))
  numNegPCC = 0
  for i in range(numCells):
    for j in range(numCells):
      if i == j and removeAutoCorr == True:
        continue
      if not all(spikeTrains[i,:] == 0) and not all(spikeTrains[j,:] == 0):
        corrMatrix[i,j] = np.corrcoef(spikeTrains[i,:], spikeTrains[j,:])[0,1]    
        if corrMatrix[i,j] < 0:
          numNegPCC += 1
  return (corrMatrix, numNegPCC)

  
def accuracy(current, predicted):
  """
  Computes the accuracy of the TM at time-step t based on the prediction
  at time-step t-1 and the current active columns at time-step t.
  
  @param current (array) binary vector containing current active columns
  @param predicted (array) binary vector containing predicted active columns  
  @return acc (float) prediction accuracy of the TM at time-step t
  """  
  acc = 0
  if np.count_nonzero(predicted) > 0:
    acc = float(np.dot(current, predicted))/float(np.count_nonzero(predicted))
  return acc   


def sampleCellsRandom(numCellPairs, cellsPerColumn, numColumns, seed=42):
  """
  Generate indices of cell pairs randomly
  @return cellPairs (list) list of cell pairs
  """
  np.random.seed(seed)
  cellPairs = []
  for i in range(numCellPairs):
    randCols = np.random.choice(np.arange(numColumns), (2, ), replace=True)
    randCells = np.random.choice(np.arange(cellsPerColumn), (2, ), replace=True)

    cellsPair = np.zeros((2, ))
    for j in range(2):
      cellsPair[j] = randCols[j] * cellsPerColumn + randCells[j]
    cellPairs.append(cellsPair.astype('int32'))
  return cellPairs


def sampleCellsWithinColumns(numCellPairs, cellsPerColumn, numColumns, seed=42):
  """
  Generate indices of cell pairs, each pair of cells are from the same column
  @return cellPairs (list) list of cell pairs
  """
  np.random.seed(seed)
  cellPairs = []
  for i in range(numCellPairs):
    randCol = np.random.randint(numColumns)
    randCells = np.random.choice(np.arange(cellsPerColumn), (2, ), replace=False)

    cellsPair = randCol * cellsPerColumn + randCells
    cellPairs.append(cellsPair)
  return cellPairs



def sampleCellsAcrossColumns(numCellPairs, cellsPerColumn, numColumns, seed=42):
  """
  Generate indices of cell pairs, each pair of cells are from different column
  @return cellPairs (list) list of cell pairs
  """
  np.random.seed(seed)
  cellPairs = []
  for i in range(numCellPairs):
    randCols = np.random.choice(np.arange(numColumns), (2, ), replace=False)
    randCells = np.random.choice(np.arange(cellsPerColumn), (2, ), replace=False)

    cellsPair = np.zeros((2, ))
    for j in range(2):
      cellsPair[j] = randCols[j] * cellsPerColumn + randCells[j]
    cellPairs.append(cellsPair.astype('int32'))
  return cellPairs



def subSample(spikeTrains, numCells, totalCells, currentTS):
  """
  Obtains a random sample of cells from the whole spike train matrix consisting of numCells cells
  from the start of simulation time up to currentTS
  
  @param spikeTrains
  @param numCells (int) number of cells to be sampled from the matrix of spike trains
  @param currentTS (int) time-step upper bound of sample (sample will go from time-step 0 up to currentTS)
  @return subSpikeTrains (array) spike train matrix sampled from the total spike train matrix
  """
  indices = np.random.permutation(np.arange(totalCells))
  if currentTS > 0:
    subSpikeTrains = np.zeros((numCells, currentTS), dtype = "uint32")
    for i in range(numCells):
      for t in range(currentTS):
        subSpikeTrains[i,t] = spikeTrains[indices[i],t]
  else:
    timeSteps = 1000
    totalTS = np.shape(spikeTrains)[1]
    subSpikeTrains = np.zeros((numCells, timeSteps), dtype = "uint32")
    rnd = random.randrange(totalTS - timeSteps)
    print rnd
    for i in range(numCells):
      for t in range(timeSteps):
        subSpikeTrains[i,t] = spikeTrains[indices[i],rnd + t]
    
  return subSpikeTrains


def subSampleWholeColumn(spikeTrains, colIndices, cellsPerColumn, currentTS):
  """
  Obtains subsample from matrix of spike trains by considering the cells in columns specified
  by colIndices. Thus, it returns a matrix of spike trains of cells within the same column.
    
  @param spikeTrains
  @param colIndices
  @param cellsPerColumn
  @param currentTs (int) time-step upper bound of sample (sample will go from time-step 0 up to currentTS)
  @return subSpikeTrains (array) spike train matrix sampled from the total spike train matrix
  """
  numColumns = np.shape(colIndices)[0]
  numCells = numColumns * cellsPerColumn

  if currentTS > 0:
    subSpikeTrains = np.zeros((numCells, currentTS), dtype = "uint32")
    for i in range(numColumns):
      currentCol = colIndices[i]
      initialCell = cellsPerColumn * currentCol
      for j in range(cellsPerColumn):
        for t in range(currentTS):
          subSpikeTrains[(cellsPerColumn*i) + j,t] = spikeTrains[initialCell + j,t]
  else:
    timeSteps = 1000
    subSpikeTrains = np.zeros((numCells, timeSteps), dtype = "uint32")
    rnd = random.randrange(totalTS - timeSteps)
    print rnd
    for i in range(numColumns):
      currentCol = colIndices[i]
      initialCell = cellsPerColumn * currentCol
      for j in range(cellsPerColumn):
        for t in range(timeSteps):
          subSpikeTrains[(cellsPerColumn*i) + j,t] = spikeTrains[initialCell + j,rnd + t]
    
  return subSpikeTrains


def computeEntropy(spikeTrains):
  """
  Estimates entropy in spike trains.
  
  @param spikeTrains (array) matrix of spike trains
  @return entropy (float) entropy
  """
  MIN_ACTIVATION_PROB = 0.000001
  activationProb = np.mean(spikeTrains, 1)
  activationProb[activationProb < MIN_ACTIVATION_PROB] = MIN_ACTIVATION_PROB
  activationProb = activationProb / np.sum(activationProb)
  entropy = -np.dot(activationProb, np.log2(activationProb))
  return entropy


def computeISI(spikeTrains):
  """
  Estimates the inter-spike interval from a spike train matrix.
  
  @param spikeTrains (array) matrix of spike trains
  @return isi (array) matrix with the inter-spike interval obtained from the spike train.
          Each entry in this matrix represents the number of time-steps in-between 2 spikes
          as the algorithm scans the spike train matrix.
  """
  zeroCount = 0
  isi = []
  cells = 0
  for i in range(np.shape(spikeTrains)[0]):
    if cells > 0 and cells % 250 == 0:
      print str(cells) + " cells processed"
    for j in range(np.shape(spikeTrains)[1]):
      if spikeTrains[i][j] == 0:
        zeroCount += 1
      elif zeroCount > 0:
        isi.append(zeroCount)
        zeroCount = 0
    zeroCount = 0
    cells += 1
  print "**All cells processed**"
  return isi


def poissonSpikeGenerator(firingRate, nBins, nTrials):
  """
  Generates a Poisson spike train.
  
  @param firingRate (int) 
  @param nBins (int)
  @param nTrials (int)
  @return poissonSpikeTrain (array)
  """
  dt = 0.001 # we are simulating a ms as a single bin in a vector, ie 1sec = 1000bins
  poissonSpikeTrain = np.zeros((nTrials, nBins), dtype = "uint32")
  for i in range(nTrials):
    for j in range(int(nBins)):
      if random.random() < firingRate*dt:
        poissonSpikeTrain[i,j] = 1
  return poissonSpikeTrain


def raster(event_times_list, color='k'):
  """
  Creates a raster from spike trains.
  
  @param event_times_list (array) matrix containing times in which a cell fired
  @param color (string) color of spike in raster
  
  @return ax (int) position of plot axes
  """
  ax = plt.gca()
  for ith, trial in enumerate(event_times_list):
    plt.vlines(trial, ith + .5, ith + 1.5, color=color)
  plt.ylim(.5, len(event_times_list) + .5)
  return ax


def rasterPlot(spikeTrain, model):
  """
  Plots raster and saves figure in working directory
  
  @param spikeTrain (array) matrix of spike trains
  @param model (string) string specifying the name of the origin of the spike trains
         for the purpose of concatenating it to the filename (either TM or Poisson)
  """
  nTrials = np.shape(spikeTrain)[0]
  spikes = []
  for i in range(nTrials):
    spikes.append(spikeTrain[i].nonzero()[0].tolist())
  plt.figure()
  ax = raster(spikes)
  plt.xlabel('Time')
  plt.ylabel('Neuron')
  # plt.show()
  plt.savefig("raster" + str(model))
  plt.close()   


def countInSample(binaryWord, spikeTrain):
  """
  Counts the times in which a binary pattern (e.g. [1 0 0 1 1 0 1]) occurs in a spike train
  
  @param binaryWord (array) binary vector to be search within the spike train matrix.
  @param spikeTrain (array) matrix of spike trains. Dimensions of binary word are 1xN, whereas
         dimensions of spikeTrain must be NxtimeSteps
  @return count (int) number of occurrences of binaryWord in spikeTrains
  """
  count = 0
  timeSteps = np.shape(spikeTrain)[1]
  for i in range(timeSteps):
    if np.dot(binaryWord, spikeTrain[:,i]) == np.count_nonzero(binaryWord):
      count += 1
      #print i
  return count
  
def simpleAccuracyTest(model, tm, allSequences):
  """
  Computes a simple accuracy measure in-between two time-steps in the simulation.
  At time t accuracy is estimated by computing the number of columns correctly
  predicted by the temporal memory at time t-1
  
  @param model (string) string specifying whether the experiment is based on random
  						or periodic data
  @param tm (TemporalMemory) temporal memory used during the experiment
  @param allSequences (array) sequences used during the experiment
  """
  totalItems = np.shape(allSequences)[0]
  
  # has the TM learned anything at all? Let's test its predictive power
  currentColumns = np.zeros(tm.numberOfColumns(), dtype="uint32")
  predictedColumns = np.zeros(tm.numberOfColumns(), dtype="uint32")
  
  if model == "periodic":
    items = 10
    offset = random.randrange(totalItems - items)
    print "Starting from record no.: " + str(offset)
    for i in range(items):
      tm.compute(allSequences[offset + i][0].tolist(), learn=False)
      activeColumnsIndices = [tm.columnForCell(i) for i in tm.getActiveCells()]
      predictedColumnIndices = [tm.columnForCell(i) for i in tm.getPredictiveCells()]
      currentColumns = [1 if i in activeColumnsIndices else 0 for i in range(tm.numberOfColumns())]
      acc = accuracy(currentColumns, predictedColumns)
      print "Accuracy: " + str(acc)
      predictedColumns = [1 if i in predictedColumnIndices else 0 for i in range(tm.numberOfColumns())]
      print("Active cols: " + str(np.nonzero(currentColumns)[0]))
      print("Predicted cols: " + str(np.nonzero(predictedColumns)[0]))
      print ""
  elif model == "random":
    symbolsPerSequence = np.shape(allSequences)[1]
    # choose a sequences at random
    rnd = random.randrange(totalItems)
    print "Sequence no. " + str(rnd) + "\n"
    for symbol in range(symbolsPerSequence):
      tm.compute(allSequences[rnd][symbol], learn=False)
      activeColumnsIndices = [tm.columnForCell(i) for i in tm.getActiveCells()]
      predictedColumnIndices = [tm.columnForCell(i) for i in tm.getPredictiveCells()]
      currentColumns = [1 if i in activeColumnsIndices else 0 for i in range(tm.numberOfColumns())]
      acc = accuracy(currentColumns, predictedColumns)
      print "Accuracy: " + str(acc)
      predictedColumns = [1 if i in predictedColumnIndices else 0 for i in range(tm.numberOfColumns())]
      print("Active cols: " + str(np.nonzero(currentColumns)[0]))
      print("Predicted cols: " + str(np.nonzero(predictedColumns)[0]))
      print ""
    
def saveTM(tm):
  """
  Saves the temporal memory and the sequences generated for its training.
  
  @param tm (TemporalMemory) temporal memory used during the experiment
  """
  # Save the TM to a file for future use
  proto1 = TemporalMemoryProto_capnp.TemporalMemoryProto.new_message()
  tm.write(proto1)
  # Write the proto to a file and read it back into a new proto
  with open('tm.nta', 'wb') as f:
    proto1.write(f)
    
def inputAnalysis(allSequences, model, numColumns):
  """
  Calculates the overlap score of each SDR used as input to the temporal memory. Generates
  an overlap matrix with entries (i,j) = overlapScore(i, j)
  
  @param allSequences (array) sequences using during the experiment
  @param model (string) string specifying whether the experiment used random or
  						periodic data
  @param numColumns (int) number of columns in the temporal memory
   
  @return overlapMatrix (array) matrix whose entries (i, j) contain the overlap score
  								between SDRs i and j
  """
  records = np.shape(allSequences)[0]
  symbols = np.shape(allSequences)[1]
  totalItems = records * symbols
  
  overlapMatrix = np.zeros((totalItems, totalItems))
  
  for i in range(totalItems):
    if i % 500 == 0:
      print str(i) + " rows processed"
    for j in range(totalItems):
      if model == "random":
        overlapMatrix[i, j] = percentOverlap(allSequences[int(i/symbols)][i%symbols], 
                                             allSequences[int(j/symbols)][j%symbols], 
                                             numColumns)
      elif model == "periodic":
        overlapMatrix[i, j] = percentOverlap(allSequences[i][0], allSequences[j][0], 
                                             numColumns)
  print "***All rows processed!***"    
  # substract diagonal from correlation matrix
  overlapMatrix = np.subtract(overlapMatrix, np.identity(totalItems))
  
  return overlapMatrix  
  
  