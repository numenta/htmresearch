#!/usr/bin/env python
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

import numpy as np
import random
import matplotlib.pyplot as plt

from nupic.encoders import ScalarEncoder
from nupic.bindings.algorithms import TemporalMemory as TM
from nupic.bindings.algorithms import SpatialPooler as SP
from htmresearch.support.neural_correlations_utils import *

from htmresearch.support.generate_sdr_dataset import getMovingBar

plt.ion()

random.seed(1)


def showBarMovie(bars, totalRpts=1):
  plt.figure(1)
  i = 0
  numRpts = 0
  while numRpts < totalRpts:
    plt.imshow(np.transpose(bars[i]), cmap='gray')
    plt.pause(.05)
    i += 1
    if i >= len(bars):
      numRpts += 1
      i = 0


def generateMovingBarDataset(Nx, Ny):
  barMovies = []
  barHalfLength = 2

  # horizongtal bars
  stratNy = 1
  for startNx in range(barHalfLength, Nx-barHalfLength+1):
    barMovie = getMovingBar(startLocation=(startNx, stratNy),
                        direction=(0, 1),
                        imageSize=(Nx, Ny),
                        barHalfLength=barHalfLength,
                        steps=Ny-stratNy)
    barMovies.append(barMovie)

  # vertical bars
  # stratNx = 1
  # for startNy in range(barHalfLength, Ny-barHalfLength+1, 2):
  #   barMovie = getMovingBar(startLocation=(startNx, stratNy),
  #                       direction=(1, 0),
  #                       imageSize=(Nx, Ny),
  #                       barHalfLength=barHalfLength,
  #                       steps=Nx-stratNx)
  #   barMovies.append(barMovie)
  return barMovies


def createSpatialPooler():
  sparsity = 0.02
  numColumns = 2048
  sparseCols = int(numColumns * sparsity)

  sp = SP(inputDimensions=(inputSize,),
          columnDimensions=(numColumns,),
          potentialRadius = int(0.5*inputSize),
          numActiveColumnsPerInhArea = sparseCols,
          globalInhibition = True,
          synPermActiveInc = 0.0001,
          synPermInactiveDec = 0.0005,
          synPermConnected = 0.5,
          maxBoost = 1.0,
          spVerbosity = 1
         )
  return sp


def createTemporalMemory():
  tm = TM(columnDimensions = (2048,),
          cellsPerColumn=8, # We changed here the number of cells per col, initially they were 32
          initialPermanence=0.21,
          connectedPermanence=0.3,
          minThreshold=15,
          maxNewSynapseCount=40,
          permanenceIncrement=0.1,
          permanenceDecrement=0.1,
          activationThreshold=15,
          predictedSegmentDecrement=0.01
         )
  return  tm



def calculateCorrelation(spikeTrains, pairs):
  numPairs = len(pairs)
  corr = np.zeros((numPairs, ))
  for pairI in range(numPairs):
    if (np.sum(spikeTrains[pairs[pairI][0], :]) == 0 or
      np.sum(spikeTrains[pairs[pairI][1], :]) == 0):
      corr[pairI] = np.nan
      continue

    (corrMatrix, numNegPCC) = computePWCorrelations(
      spikeTrains[pairs[pairI], :], removeAutoCorr=True)
    corr[pairI] = corrMatrix[0, 1]
  return corr



if __name__ == "__main__":
  Nx = 20
  Ny = 20

  inputSize = Nx * Ny
  barMovies = generateMovingBarDataset(Nx, Ny)
  # showBarMovie(barMovies[0])

  sp = createSpatialPooler()
  tm = createTemporalMemory()

  numEpochs = 20

  stepsPerEpoch = 0
  for barMoive in barMovies:
    stepsPerEpoch += len(barMoive)

  totalTS = stepsPerEpoch * numEpochs


  columnUsage = np.zeros(tm.numberOfColumns(), dtype="uint32")

  entropyX = []
  entropyY = []

  negPCCX_cells = []
  negPCCY_cells = []

  negPCCX_cols = []
  negPCCY_cols = []

  # Randomly generate the indices of the columns to keep track during simulation time
  colIndices = np.random.permutation(tm.numberOfColumns())[
               0:4]  # keep track of 4 columns

  corrWithinColumnVsEpoch = []
  corrAcrossColumnVsEpoch = []
  corrRandomVsEpoch = []
  predictedActiveColVsEpoch = []
  for epoch in range(numEpochs):
    print " {} epochs processed".format(epoch)

    predCellNum = []
    activeCellNum = []
    predictedActiveColumnsNum = []

    spikeTrains = np.zeros((tm.numberOfCells(), stepsPerEpoch), dtype="uint32")
    t = 0

    for i in range(len(barMovies)):
      barMoive = barMovies[i]

      tm.reset()
      for image in barMoive:
        prePredictiveCells = tm.getPredictiveCells()
        prePredictiveColumn = np.array(
          list(prePredictiveCells)) / tm.getCellsPerColumn()

        outputColumns = np.zeros(sp.getNumColumns(), dtype="uint32")
        sp.compute(image, False, outputColumns)

        tm.compute(outputColumns.nonzero()[0], learn=True)

        for cell in tm.getActiveCells():
          spikeTrains[cell, t] = 1

        # Obtain active columns:
        activeColumnsIndices = [tm.columnForCell(i) for i in
                                tm.getActiveCells()]
        currentColumns = [1 if i in activeColumnsIndices else 0 for i in
                          range(tm.numberOfColumns())]
        for col in np.nonzero(currentColumns)[0]:
          columnUsage[col] += 1

        t += 1

        predictiveCells = tm.getPredictiveCells()
        predCellNum.append(len(predictiveCells))
        predColumn = np.array(list(predictiveCells)) / tm.getCellsPerColumn()

        activeCellNum.append(len(activeColumnsIndices))

        predictedActiveColumns = np.intersect1d(prePredictiveColumn,
                                                activeColumnsIndices)
        predictedActiveColumnsNum.append(len(predictedActiveColumns))

    print
    print " Predicted Active Column {}".format(np.mean(predictedActiveColumnsNum[1:]))

    numPairs = 1000
    tmNumCols = np.prod(tm.getColumnDimensions())
    cellsPerColumn = tm.getCellsPerColumn()

    # within column correlation
    withinColPairs = sampleCellsWithinColumns(numPairs, cellsPerColumn, tmNumCols)
    corrWithinColumn = calculateCorrelation(spikeTrains, withinColPairs)

    # across column correlaiton
    corrAcrossColumn = np.zeros((numPairs, ))
    acrossColPairs = sampleCellsAcrossColumns(numPairs, cellsPerColumn, tmNumCols)
    corrAcrossColumn = calculateCorrelation(spikeTrains, acrossColPairs)

    # sample random pairs
    randomPairs = sampleCellsRandom(numPairs, cellsPerColumn, tmNumCols)
    corrRandomPairs = calculateCorrelation(spikeTrains, randomPairs)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].hist(corrWithinColumn, range=[-.2, 1], bins=50)
    ax[0, 0].set_title('within column')
    ax[0, 1].hist(corrAcrossColumn, range=[-.1, .1], bins=50)
    ax[0, 1].set_title('across column')
    ax[1, 0].hist(corrRandomPairs, range=[-.1, .1], bins=50)
    ax[1, 0].set_title('random pairs')
    plt.savefig('plots/corrHist/epoch_{}.pdf'.format(epoch))
    plt.close(fig)
    print "Within column correlation {}".format(np.nanmean(corrWithinColumn))
    print "Across column correlation {}".format(np.nanmean(corrAcrossColumn))
    print "Random Cell Pair correlation {}".format(np.nanmean(corrRandomPairs))

    predictedActiveColVsEpoch.append(np.mean(predictedActiveColumnsNum[1:]))
    corrWithinColumnVsEpoch.append(corrWithinColumn)
    corrAcrossColumnVsEpoch.append(corrAcrossColumn)
    corrRandomVsEpoch.append(corrRandomPairs)


  fig, ax = plt.subplots(4, 1)
  ax[0].plot(predictedActiveColVsEpoch)
  ax[0].set_title('Correctly Predicted Cols')
  ax[0].set_xticks([])
  ax[1].plot(np.nanmean(corrWithinColumnVsEpoch, 1))
  ax[1].set_title('corr within column')
  ax[1].set_xticks([])
  ax[2].plot(np.nanmean(corrAcrossColumnVsEpoch, 1))
  ax[2].set_title('corr across column')
  ax[2].set_xticks([])
  ax[3].plot(np.nanmean(corrRandomVsEpoch, 1))
  ax[3].set_title('corr random pairs')
  ax[3].set_xlabel(' epochs ')
  ax[3].set_xticks([])
  plt.savefig('CorrelationVsTraining.pdf')





