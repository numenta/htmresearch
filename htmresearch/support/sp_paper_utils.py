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

import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

from htmresearch.frameworks.sp_paper.sp_metrics import (
calculateInputOverlapMat, percentOverlap
)
from nupic.bindings.math import GetNTAReal

realDType = GetNTAReal()
uintType = "uint32"


def plotPermInfo(permInfo):
  fig, ax = plt.subplots(5, 1, sharex=True)
  ax[0].plot(permInfo['numConnectedSyn'])
  ax[0].set_title('connected syn #')
  ax[1].plot(permInfo['numNonConnectedSyn'])
  ax[0].set_title('non-connected syn #')
  ax[2].plot(permInfo['avgPermConnectedSyn'])
  ax[2].set_title('perm connected')
  ax[3].plot(permInfo['avgPermNonConnectedSyn'])
  ax[3].set_title('perm unconnected')
  # plt.figure()
  # plt.subplot(3, 1, 1)
  # plt.plot(perm - initialPermanence[columnIndex, :])
  # plt.subplot(3, 1, 2)
  # plt.plot(truePermanence - initialPermanence[columnIndex, :], 'r')
  # plt.subplot(3, 1, 3)
  # plt.plot(truePermanence - perm, 'r')



def plotAccuracyVsNoise(noiseLevelList, predictionAccuracy):
  plt.figure()
  plt.plot(noiseLevelList, predictionAccuracy, '-o')
  plt.ylim([0, 1.05])
  plt.xlabel('Noise level')
  plt.ylabel('Prediction Accuracy')



def plotSPstatsOverTime(numNewlyConnectedSynapsesTrace,
                        numEliminatedSynapsesTrace,
                        noiseRobustnessTrace,
                        stabilityTrace,
                        entropyTrace,
                        fileName=None):
  fig, axs = plt.subplots(nrows=5, ncols=1, sharex=True)

  axs[0].plot(stabilityTrace)
  axs[0].set_ylabel('Stability')

  axs[1].plot(entropyTrace)
  axs[1].set_ylabel('Entropy (bits)')


  if len(noiseRobustnessTrace) > 0:
    axs[2].plot(noiseRobustnessTrace)
    axs[2].set_ylabel('Noise Robustness')

  axs[3].plot(numNewlyConnectedSynapsesTrace)
  axs[3].set_ylabel('Synapses Formation')

  axs[4].plot(numEliminatedSynapsesTrace)
  axs[4].set_ylabel('Synapse Removal')
  axs[4].set_xlim([0, len(numEliminatedSynapsesTrace)])
  axs[4].set_xlabel('epochs')
  if fileName is not None:
    plt.savefig(fileName)
  return axs



def plotReceptiveFields2D(sp, Nx, Ny):
  inputSize = Nx * Ny
  numColumns = np.product(sp.getColumnDimensions())

  nrows = 4
  ncols = 4
  fig, ax = plt.subplots(nrows, ncols)
  for r in range(nrows):
    for c in range(ncols):
      colID = np.random.randint(numColumns)
      connectedSynapses = np.zeros((inputSize,), dtype=uintType)
      sp.getConnectedSynapses(colID, connectedSynapses)
      receptiveField = np.reshape(connectedSynapses, (Nx, Ny))
      ax[r, c].imshow(1-receptiveField, interpolation="nearest", cmap='gray')
      # ax[r, c].set_title('col {}'.format(colID))
      ax[r, c].set_xticks([])
      ax[r, c].set_yticks([])



def plotReceptiveFields(sp, nDim1=8, nDim2=8):
  """
  Plot 2D receptive fields for 16 randomly selected columns
  :param sp:
  :return:
  """
  columnNumber = np.product(sp.getColumnDimensions())
  fig, ax = plt.subplots(nrows=4, ncols=4)
  for rowI in range(4):
    for colI in range(4):
      col = np.random.randint(columnNumber)
      connectedSynapses = np.zeros((nDim1*nDim2,), dtype=uintType)
      sp.getConnectedSynapses(col, connectedSynapses)
      receptiveField = connectedSynapses.reshape((nDim1, nDim2))
      ax[rowI, colI].imshow(receptiveField, cmap='gray')
      ax[rowI, colI].set_title("col: {}".format(col))



def plotReceptiveFieldCenter(RFcenters, connectedCounts, inputDims,
                             minConnection=None, maxConnection=None):
  nX, nY = inputDims
  import matplotlib.cm as cm
  cmap = cm.get_cmap('jet')

  if minConnection is None:
    minConnection = np.min(connectedCounts)

  if maxConnection is None:
    maxConnection = np.max(connectedCounts)
  fig = plt.figure()
  sc = plt.scatter(RFcenters[:, 0], RFcenters[:, 1],
                   vmin=minConnection, vmax=maxConnection,
                   c=connectedCounts, cmap=cmap)
  plt.colorbar(sc)
  plt.axis('equal')
  plt.xlim([-1, nX + 1])
  plt.ylim([-1, nY + 1])

  return fig



def plotBoostTrace(sp, inputVectors, columnIndex):
  """
  Plot boostfactor for a selected column

  Note that learning is ON for SP here

  :param sp: sp instance
  :param inputVectors: input data
  :param columnIndex: index for the column of interest
  """
  numInputVector, inputSize = inputVectors.shape
  columnNumber = np.prod(sp.getColumnDimensions())
  boostFactorsTrace = np.zeros((columnNumber, numInputVector))
  activeDutyCycleTrace = np.zeros((columnNumber, numInputVector))
  minActiveDutyCycleTrace = np.zeros((columnNumber, numInputVector))
  for i in range(numInputVector):
    outputColumns = np.zeros(sp.getColumnDimensions(), dtype=uintType)
    inputVector = copy.deepcopy(inputVectors[i][:])
    sp.compute(inputVector, True, outputColumns)

    boostFactors = np.zeros((columnNumber, ), dtype=realDType)
    sp.getBoostFactors(boostFactors)
    boostFactorsTrace[:, i] = boostFactors

    activeDutyCycle = np.zeros((columnNumber, ), dtype=realDType)
    sp.getActiveDutyCycles(activeDutyCycle)
    activeDutyCycleTrace[:, i] = activeDutyCycle

    minActiveDutyCycle = np.zeros((columnNumber, ), dtype=realDType)
    sp.getMinActiveDutyCycles(minActiveDutyCycle)
    minActiveDutyCycleTrace[:, i] = minActiveDutyCycle

  plt.figure()
  plt.subplot(2, 1, 1)
  plt.plot(boostFactorsTrace[columnIndex, :])
  plt.ylabel('Boost Factor')
  plt.subplot(2, 1, 2)
  plt.plot(activeDutyCycleTrace[columnIndex, :])
  plt.plot(minActiveDutyCycleTrace[columnIndex, :])
  plt.xlabel(' Time ')
  plt.ylabel('Active Duty Cycle')



def analyzeReceptiveFieldSparseInputs(inputVectors, sp):
  numColumns = np.product(sp.getColumnDimensions())
  overlapMat = calculateInputOverlapMat(inputVectors, sp)

  sortedOverlapMat = np.zeros(overlapMat.shape)
  for c in range(numColumns):
    sortedOverlapMat[c, :] = np.sort(overlapMat[c, :])

  avgSortedOverlaps = np.flipud(np.mean(sortedOverlapMat, 0))
  plt.figure()
  plt.plot(avgSortedOverlaps, '-o')
  plt.xlabel('sorted input vector #')
  plt.ylabel('percent overlap')

  plt.figure()
  plt.imshow(overlapMat[:100, :], interpolation="nearest", cmap="magma")
  plt.xlabel("Input Vector #")
  plt.ylabel("SP Column #")
  plt.colorbar()
  plt.title('percent overlap')



def analyzeReceptiveFieldCorrelatedInputs(
        inputVectors, sp, params, inputVectors1, inputVectors2):

  columnNumber = np.prod(sp.getColumnDimensions())
  numInputVector, inputSize = inputVectors.shape
  numInputVector1 = params['numInputVectorPerSensor']
  numInputVector2 = params['numInputVectorPerSensor']
  w = params['numActiveInputBits']
  inputSize1 = int(params['inputSize']/2)
  inputSize2 = int(params['inputSize']/2)

  connectedCounts = np.zeros((columnNumber,), dtype=uintType)
  sp.getConnectedCounts(connectedCounts)

  numColumns = np.product(sp.getColumnDimensions())
  overlapMat1 = np.zeros((numColumns, inputVectors1.shape[0]))
  overlapMat2 = np.zeros((numColumns, inputVectors2.shape[0]))
  numColumns = np.product(sp.getColumnDimensions())
  numInputVector, inputSize = inputVectors.shape

  for c in range(numColumns):
    connectedSynapses = np.zeros((inputSize,), dtype=uintType)
    sp.getConnectedSynapses(c, connectedSynapses)
    for i in range(inputVectors1.shape[0]):
      overlapMat1[c, i] = percentOverlap(connectedSynapses[:inputSize1],
                                         inputVectors1[i, :inputSize1])
    for i in range(inputVectors2.shape[0]):
      overlapMat2[c, i] = percentOverlap(connectedSynapses[inputSize1:],
                                         inputVectors2[i, :inputSize2])

  sortedOverlapMat1 = np.zeros(overlapMat1.shape)
  sortedOverlapMat2 = np.zeros(overlapMat2.shape)
  for c in range(numColumns):
    sortedOverlapMat1[c, :] = np.sort(overlapMat1[c, :])
    sortedOverlapMat2[c, :] = np.sort(overlapMat2[c, :])
  fig, ax = plt.subplots(nrows=2, ncols=2)
  ax[0, 0].plot(np.mean(sortedOverlapMat1, 0), '-o')
  ax[0, 1].plot(np.mean(sortedOverlapMat2, 0), '-o')

  fig, ax = plt.subplots(nrows=1, ncols=2)
  ax[0].imshow(overlapMat1[:100, :], interpolation="nearest", cmap="magma")
  ax[0].set_xlabel('# Input 1')
  ax[0].set_ylabel('SP Column #')
  ax[1].imshow(overlapMat2[:100, :], interpolation="nearest", cmap="magma")
  ax[1].set_xlabel('# Input 2')
  ax[1].set_ylabel('SP Column #')



def runSPOnBatch(sp, inputVectors, learn):
  numInputVector, inputSize = inputVectors.shape
  numColumns = np.prod(sp.getColumnDimensions())

  outputColumns = np.zeros((numInputVector, numColumns), dtype=uintType)
  for i in range(numInputVector):
    sp.compute(inputVectors[i][:], learn, outputColumns[i][:])

  return outputColumns