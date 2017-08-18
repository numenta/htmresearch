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

Train a bag of words classifier with/without location
python bag_of_words_classifier.py --location 1

python bag_of_words_classifier.py  --location 0
"""

import cPickle
from optparse import OptionParser
import numpy as np
import  matplotlib.pyplot as plt

from htmresearch.frameworks.layers.object_machine_factory import (
  createObjectMachine
)


plt.ion()
plt.close('all')
def _getArgs():
  parser = OptionParser(usage="Train BoW classifier on Thing data")

  parser.add_option("-l",
                    "--location",
                    type=int,
                    default=0,
                    dest="useLocation",
                    help="Whether to use location signal")


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



def bowClassifierPredict(input, bowVectors, distance="dotProduct"):
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

  # Create the objects
  numFeatures = 10
  pointRange = 1
  numObjects = 50
  numLocations = 10
  numPoints = 10
  numTrials = 10

  accuracyList = np.zeros((numTrials, numFeatures))

  for trial in range(numTrials):
    objects = createObjectMachine(
      machineType="simple",
      numInputBits=20,
      sensorInputSize=150,
      externalInputSize=2400,
      numCorticalColumns=1,
      numFeatures=numFeatures,
      seed=trial
    )

    for p in range(pointRange):
      objects.createRandomObjects(numObjects, numPoints=numPoints+p,
                                        numLocations=numLocations,
                                        numFeatures=numFeatures)

    objects = objects.provideObjectsToLearn()
    objectNames = objects.keys()
    numObjs = len(objectNames)
    featureWidth = 150
    locationWidth = 2400
    useLocation = expConfig.useLocation


    # compute the number of sensations across all objects
    numInputVectors = 0
    for i in range(numObjs):
      numInputVectors += len(objects[objectNames[i]])

    if useLocation:
      inputWidth = featureWidth + locationWidth
    else:
      inputWidth = featureWidth

    # create "training" dataset
    data = np.zeros((numInputVectors, inputWidth))
    label = np.zeros((numInputVectors, numObjs))
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
    for maxSenses in range(10):
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
      accuracyList[trial, maxSenses] = accuracy
      print "maxSenses {} accuracy {}".format(maxSenses, accuracy)

  result = {"numTouches": range(10),
            "accuracy": accuracyList}

  # Pickle results for later use
  resultsName = "bag_of_words_useLocation_{}.pkl".format(useLocation)
  with open(resultsName,"wb") as f:
    cPickle.dump(result,f)

  plt.figure()
  plt.plot(range(10), np.mean(accuracyList, 0))
  plt.xlabel('number of sensations')
  plt.ylabel('accuracy ')


