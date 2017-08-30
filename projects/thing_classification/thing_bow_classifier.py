# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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
This file is used to run Thing experiments using simulated sensations with
a simple logistic encoder, with/without location signal


Example usage

Train network with (feature, location) as input and a spatial pooler
The spatial pooler speeds up convergence
python thing_feedforward_network.py --spatial_pooler 1 --location 1

Train network with only feature input
python thing_feedforward_network.py --spatial_pooler 0 --location 0
"""

from optparse import OptionParser
import numpy as np
from nupic.bindings.algorithms import SpatialPooler as CPPSpatialPooler
import  matplotlib.pyplot as plt
from thing_convergence import loadThingObjects


def createSpatialPooler(inputWidth):
  spParam = {
    "inputDimensions": (inputWidth, ),
    "columnDimensions": (inputWidth, ),
    "potentialRadius": inputWidth,
    "potentialPct": 0.5,
    "globalInhibition": True,
    "localAreaDensity": .02,
    "numActiveColumnsPerInhArea": -1,
    "stimulusThreshold": 1,
    "synPermInactiveDec": 0.004,
    "synPermActiveInc": 0.02,
    "synPermConnected": 0.5,
    "minPctOverlapDutyCycle": 0.0,
    "dutyCyclePeriod": 1000,
    "boostStrength": 0.0,
    "seed": 1936
  }
  print "use spatial pooler to encode feature location pairs "
  print " initializing spatial pooler ... "
  sp = CPPSpatialPooler(**spParam)
  return sp



def _getArgs():
  parser = OptionParser(usage="Train BoW classifier on Thing data")

  parser.add_option("-l",
                    "--location",
                    type=int,
                    default=0,
                    dest="useLocation",
                    help="Whether to use location signal")

  parser.add_option("--spatial_pooler",
                    type=int,
                    default=0,
                    dest="useSP",
                    help="Whether to use spatial pooler")

  (options, remainder) = parser.parse_args()
  print options
  return options, remainder


def findWordInVocabulary(input, wordList):
  findWord = None
  for j in range(wordList.shape[0]):
    numBitDiff = np.sum(np.abs(input - wordList[j, :]))
    if numBitDiff == 0:
      findWord = j
  return findWord



def bowClassifierPredict(input, bowVectors, distance="L1"):
  numClasses = bowVectors.shape[0]
  output = np.zeros((numClasses,))

  # normalize input
  if distance == "L1":
    for i in range(numClasses):
      output[i] = np.sum(np.abs(input - bowVectors[i, :]))

  elif distance == "dotProduct":
    # normalize input
    input = input / np.linalg.norm(input)
    for i in range(numClasses):
      bowVectors[i, :] = bowVectors[i, :]/np.linalg.norm(bowVectors[i, :])
      output[i] = np.dot(input, bowVectors[i, :])
    output = 1 - output
  return output


if __name__ == "__main__":
  (expConfig, _args) = _getArgs()

  objects, OnBitsList = loadThingObjects(1, './data')
  objects = objects.provideObjectsToLearn()
  objectNames = objects.keys()
  numObjs = len(objectNames)
  featureWidth = 256
  locationWidth = 1024
  useLocation = expConfig.useLocation
  useSpatialPooler = expConfig.useSP

  numInputVectors = 0
  for i in range(numObjs):
    numInputVectors += len(objects[objectNames[i]])
  if useLocation:
    inputWidth = featureWidth + locationWidth
  else:
    inputWidth = featureWidth
  data = np.zeros((numInputVectors, inputWidth))
  label = np.zeros((numInputVectors, numObjs))


  if useSpatialPooler:
    sp = createSpatialPooler(inputWidth)
  else:
    sp = None

  k = 0
  for i in range(numObjs):
    print "converting object {} ...".format(i)
    numSenses = len( objects[objectNames[i]])
    for j in range(numSenses):
      activeBitsLocation = np.array(list(objects[objectNames[i]][j][0][0]))
      activeBitsFeature = np.array(list(objects[objectNames[i]][j][0][1]))

      data[k, activeBitsFeature] = 1
      if useLocation:
        data[k, featureWidth+activeBitsLocation] = 1
      label[k, i] = 1

      if useSpatialPooler:
        inputVector = data[k, :]
        outputColumns = np.zeros((inputWidth, ))
        sp.compute(inputVector, False, outputColumns)
        activeBits = np.where(outputColumns)[0]
        data[k, activeBits] = 1
      k += 1

  # enumerate number of distinct "words"
  wordList = np.zeros((0, inputWidth), dtype='int32')
  featureList = np.zeros((data.shape[0], ))
  for i in range(data.shape[0]):
    findWord = False
    for j in range(wordList.shape[0]):
      index = findWordInVocabulary(data[i, :], wordList)
      if index is not None:
        featureList[i] = index
        findWord = True
        break

    if findWord is False:
      newWord = np.zeros((1, inputWidth), dtype='int32')
      newWord[0, :] = data[i, :]
      wordList = np.concatenate((wordList, newWord))
      featureList[i] = wordList.shape[0]-1

  numWords = wordList.shape[0]
  print "number of distinct words {}".format(numWords)

  # convert objects to BOW representations
  bowVectors = np.zeros((numObjs, numWords))
  k = 0
  for i in range(numObjs):
    numSenses = len(objects[objectNames[i]])
    for j in range(numSenses):
      index = findWordInVocabulary(data[k, :], wordList)
      bowVectors[i, index] += 1
      k += 1

  plt.figure()
  plt.imshow(np.transpose(bowVectors))
  plt.xlabel('Object #')
  plt.ylabel('Word #')
  plt.title("BoW representations")


  numCorrect = 0
  for i in range(numObjs):
    output = bowClassifierPredict(bowVectors[i, :], bowVectors)
    predictLabel = np.argmin(output)
    numCorrect += predictLabel == i
    print " true label {} predicted label {}".format(i, predictLabel)
  print "BOW classifier accuracy {}".format(float(numCorrect)/numObjs)

  # plot accuracy as a function of number of sensations
  accuracyList = []
  for maxSenses in range(30):
    bowVectorsTest = np.zeros((numObjs, numWords))
    offset = 0
    for i in range(numObjs):
      numSenses = len(objects[objectNames[i]])
      numSenses = min(numSenses, maxSenses)
      for j in range(numSenses):
        index = findWordInVocabulary(data[offset+j, :], wordList)
        bowVectorsTest[i, index] += 1
      offset += len(objects[objectNames[i]])

    numCorrect = 0
    for i in range(numObjs):
      output = bowClassifierPredict(bowVectorsTest[i, :], bowVectors)
      predictLabel = np.argmin(output)
      numCorrect += predictLabel == i
    accuracy = float(numCorrect)/numObjs
    accuracyList.append(accuracy)
    print "maxSenses {} accuracy {}".format(maxSenses, accuracy)
  plt.figure()
  plt.plot(range(30), accuracyList)
  plt.xlabel('number of sensations')
  plt.ylabel('accuracy ')


