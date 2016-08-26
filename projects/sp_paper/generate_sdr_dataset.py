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

import random
import numpy as np

uintType = "uint32"



def getCross(nX, nY, barHalfLength):
  cross = np.zeros((nX, nY), dtype=uintType)
  xLoc = np.random.randint(barHalfLength, nX - barHalfLength)
  yLoc = np.random.randint(barHalfLength, nY - barHalfLength)
  cross[(xLoc - barHalfLength):(xLoc + barHalfLength+1), yLoc] = 1
  cross[xLoc, (yLoc - barHalfLength):(yLoc + barHalfLength+1)] = 1
  return cross



def generateRandomSDR(numSDR, numDims, numActiveInputBits, seed=42):
  """
  Generate a set of random SDR's
  @param numSDR:
  @param nDim:
  @param numActiveInputBits:
  """
  randomSDRs = np.zeros((numSDR, numDims), dtype=uintType)
  indices = np.array(range(numDims))
  np.random.seed(seed)
  for i in range(numSDR):
    randomIndices = np.random.permutation(indices)
    activeBits = randomIndices[:numActiveInputBits]
    randomSDRs[i, activeBits] = 1

  return randomSDRs



def getBar(nX, nY, barHalfLength, orientation=0):
  bar = np.zeros((nX, nY), dtype=uintType)
  if orientation == 0:
    xLoc = np.random.randint(barHalfLength, nX - barHalfLength)
    yLoc = np.random.randint(0, nY)
    bar[(xLoc - barHalfLength):(xLoc + barHalfLength + 1), yLoc] = 1
  else:
    xLoc = np.random.randint(0, nX)
    yLoc = np.random.randint(barHalfLength, nY - barHalfLength)
    bar[xLoc, (yLoc - barHalfLength):(yLoc + barHalfLength + 1)] = 1
  return bar



def generateCorrelatedSDRPairs(numInputVectors,
                               inputSize,
                               numInputVectorPerSensor,
                               numActiveInputBits,
                               corrStrength=0.1,
                               seed=42):

  inputVectors1 = generateRandomSDR(
    numInputVectorPerSensor, int(inputSize/2), numActiveInputBits, seed)
  inputVectors2 = generateRandomSDR(
    numInputVectorPerSensor, int(inputSize/2), numActiveInputBits, seed+1)


  # for each input on sensor 1, how many inputs on the 2nd sensor are
  # strongly correlated with it?
  numCorrPairs = 2
  numInputVector1 = numInputVectorPerSensor
  numInputVector2 = numInputVectorPerSensor
  corrPairs = np.zeros((numInputVector1, numInputVector2))
  for i in range(numInputVector1):
    idx = np.random.choice(np.arange(numInputVector2),
                           size=(numCorrPairs, ), replace=False)
    corrPairs[i, idx] = 1.0/numCorrPairs

  uniformDist = np.ones((numInputVector1, numInputVector2))/numInputVector2
  sampleProb = corrPairs * corrStrength + uniformDist * (1-corrStrength)
  inputVectors = np.zeros((numInputVectors, inputSize))
  for i in range(numInputVectors):
    vec1 = np.random.randint(numInputVector1)
    vec2 = np.random.choice(np.arange(numInputVector2), p=sampleProb[vec1, :])

    inputVectors[i][:] = np.concatenate((inputVectors1[vec1],
                                         inputVectors2[vec2]))

  return inputVectors, inputVectors1, inputVectors2, corrPairs



def generateDenseVectors(numVectors, inputSize, seed):
  np.random.seed(seed)
  inputVectors = np.zeros((numVectors, inputSize), dtype=uintType)
  for i in range(numVectors):
    for j in range(inputSize):
      inputVectors[i][j] = random.randrange(2)
  return inputVectors



class SDRDataSet(object):
  """
  Generate, store, and manipulate SDR dataset
  """
  def __init__(self,
               params):

    self._params = params
    self._inputVectors = []
    self._dataType = params['dataType']
    self._additionalInfo = {}
    self.initialize(params)


  def initialize(self, params):

    if params['dataType'] == 'randomSDR':
      self._inputVectors = generateRandomSDR(
        params['numInputVectors'],
        params['inputSize'],
        params['numActiveInputBits'],
        params['seed'])

    if params['dataType'] == 'denseVectors':
      self._inputVectors = generateDenseVectors(
        params['numInputVectors'],
        params['inputSize'],
        params['seed'])

    if params['dataType'] == 'randomBarPairs':
      inputSize = params['nX'] * params['nY']
      numInputVectors = params['numInputVectors']
      self._inputVectors = np.zeros((numInputVectors, inputSize), dtype=uintType)
      for i in range(numInputVectors):
        bar1 = getBar(params['nX'], params['nY'], params['barHalfLength'], 0)
        bar2 = getBar(params['nX'], params['nY'], params['barHalfLength'], 1)
        data = bar1 + bar2
        self._inputVectors[i, :] = np.reshape(data, newshape=(1, inputSize))

    if params['dataType'] == 'randomCross':
      inputSize = params['nX'] * params['nY']
      numInputVectors = params['numInputVectors']
      self._inputVectors = np.zeros((numInputVectors, inputSize), dtype=uintType)
      for i in range(numInputVectors):
        data = getCross(params['nX'], params['nY'], params['barHalfLength'])
        self._inputVectors[i, :] = np.reshape(data, newshape=(1, inputSize))


    if params['dataType'] == 'correlatedSDRPairs':
      (inputVectors, inputVectors1, inputVectors2, corrPairs) = \
        generateCorrelatedSDRPairs(
          params['numInputVectors'],
          params['inputSize'],
          params['numInputVectorPerSensor'],
          params['numActiveInputBits'],
          params['corrStrength'],
          params['seed'])
      self._inputVectors = inputVectors
      self._additionalInfo = {"inputVectors1": inputVectors1,
                              "inputVectors2": inputVectors2,
                              "corrPairs": corrPairs}

  def getInputVectors(self):
    return self._inputVectors

  def getAdditionalInfo(self):
    return self._additionalInfo