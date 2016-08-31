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
This file test the capacity of a single L4-L2 column

In this test, we consider a set of objects without any shared (feature, locaiton)
pairs and without any noise. A single L4-L2 column is trained on all objects.

In the test phase, we randomly pick a (feature, location) SDR and feed it to
the network, and asked whether the correct object can be retrieved.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from htmresearch.frameworks.layers.l2_l4_inference import L4L2Experiment


def createRandomObjects(numObjects,
                        numPointsPerObject,
                        numLocations,
                        numFeatures):
  """
  Create numObjects with non-shared (feature, location) pairs
  :param numObjects: number of objects
  :param numPointsPerObject: number of (feature, location) pairs per object
  :param numLocations: number of unique locations
  :param numFeatures: number of unique features
  :return:   (list(list(tuple))  List of lists of feature / location pairs.
  """
  requiredFeatureLocPairs = numObjects * numPointsPerObject
  uniqueFeatureLocPairs = numLocations * numFeatures
  if requiredFeatureLocPairs > uniqueFeatureLocPairs:
    raise RuntimeError("Not Enough Feature Location Pairs")

  randomPairIdx = np.random.choice(
    np.arange(uniqueFeatureLocPairs),
    numObjects * numPointsPerObject,
    replace=False)

  randomFeatureLocPairs = []
  for idx in randomPairIdx:
    randomFeatureLocPairs.append(divmod(idx, numFeatures))

  objects = []
  for i in range(numObjects):
    object = []
    for j in range(numPointsPerObject):
      object.append(randomFeatureLocPairs.pop())
    objects.append(object)
  return objects



def testNetworkWithOneObject(objects, exp, testObject, numTestPoints):
  numObjects = len(objects)
  numPointsPerObject = len(objects[0])

  testPts = np.random.choice(np.arange(numPointsPerObject),
                             (numTestPoints,),
                             replace=False)
  testPairs = [objects[testObject][i] for i in testPts]

  exp._unsetLearningMode()
  exp.sendResetSignal()

  overlap = np.zeros((numTestPoints, numObjects))
  for step in xrange(numTestPoints):
    exp._addPointToQueue([testPairs[step]], noise=None)
    exp.network.run(1)
    for obj in range(numObjects):
      overlap[step, obj] = (len(exp.objectL2Representations[obj][0]
                          & exp.getL2Representations()[0]))
  return overlap



def testOnSingleRandomSDR(objects, exp, numRepeats=100):
  numObjects = len(objects)
  numPointsPerObject = len(objects[0])
  overlapTrueObj = np.zeros((numRepeats, ))
  confusion = np.zeros((numRepeats,))
  outcome = np.zeros((numRepeats,))

  for i in range(numRepeats):
    targetObject = np.random.choice(np.arange(numObjects))

    nonTargetObjs = range(numObjects)
    nonTargetObjs.remove(targetObject)
    nonTargetObjs = np.array(nonTargetObjs)

    overlap = testNetworkWithOneObject(objects, exp, targetObject, 1)
    outcome[i] = 1 if np.argmax(overlap[0, :]) == targetObject else 0
    confusion[i] = np.max(overlap[0, nonTargetObjs])
    overlapTrueObj[i] = overlap[0, targetObject]

  l2Overlap = []
  for i in range(numObjects):
    for j in range(i+1, numObjects):
      l2Overlap.append(len(exp.objectL2Representations[0][0] &
                           exp.objectL2Representations[1][0]))
  return {"numObjects": numObjects,
          "numPointsPerObject": numPointsPerObject,
          "confusion": np.mean(confusion),
          "accuracy": np.mean(outcome),
          "overlapTrueObj": np.mean(overlapTrueObj),
          "l2OverlapMean": np.mean(l2Overlap),
          "l2OverlapMax": np.max(l2Overlap)}



def plotResults(result, xaxis="numPointsPerObject", filename=None):
  if xaxis == "numPointsPerObject":
    x = result.numPointsPerObject
    xlabel = "# Pts / Obj"
  elif xaxis == "numObjects":
    x = result.numObjects
    xlabel = "Object #"
  fig, ax = plt.subplots(2, 2)
  ax[0, 0].plot(x, result.accuracy, '-o')
  ax[0, 0].set_ylabel("Accuracy")
  ax[0, 0].set_xlabel(xlabel)

  ax[0, 1].plot(x, result.l2OverlapMean, '-o')
  ax[0, 1].set_ylabel("L2 Overlap")
  ax[0, 1].set_xlabel("# Pts / Obj")
  ax[0, 1].set_ylim([0, 41])
  ax[0, 1].set_xlabel(xlabel)

  ax[1, 0].plot(x, result.overlapTrueObj, '-o')
  ax[1, 0].set_ylabel("OverlapTrueObject")
  ax[1, 0].set_ylim([0, 41])
  ax[1, 0].set_xlabel(xlabel)

  ax[1, 1].plot(x, result.confusion, '-o')
  ax[1, 1].set_ylabel("OverlapFalseObject")
  ax[1, 1].set_ylim([0, 41])
  ax[1, 1].set_xlabel(xlabel)

  plt.tight_layout()
  if filename is not None:
    plt.savefig(filename)



def runCapacityTestTwoObjectsVaryingObjectSize(numObjects = 2):
  """
  Runs experiment with two objects, varying number of points per object
  """
  result = None
  for numPointsPerObject in [5, 10, 20, 30, 40, 50, 60, 80, 100]:

    exp = L4L2Experiment("capacity_two_objects",
                         numLearningPoints=1)

    numLocations = len(exp.locations[0])
    numFeatures = len(exp.features[0])

    pairs = createRandomObjects(
      numObjects, numPointsPerObject, numLocations, numFeatures)

    objects = {}
    for object in pairs:
      objects = exp.addObject(object, objects=objects)

    exp.learnObjects(objects)

    testResult = testOnSingleRandomSDR(objects, exp)
    print testResult

    if result is None:
      result = pd.DataFrame.from_dict([testResult])
    else:
      result = pd.concat([result, pd.DataFrame.from_dict([testResult])])

  plotResults(result, xaxis="numPointsPerObject",
              filename='plots/single_column_capacity_varying_object_size.pdf')



def runCapacityTestVaryingObjectNum(numPointsPerObject = 10):
  """
  Run experiment with fixed number of pts per object, varying number of objects

  """
  result = None
  for numObjects in [2, 3, 5, 10, 15, 20, 30, 40]:

    exp = L4L2Experiment("capacity_two_objects",
                         numLearningPoints=1)

    numLocations = len(exp.locations[0])
    numFeatures = len(exp.features[0])

    pairs = createRandomObjects(
      numObjects, numPointsPerObject, numLocations, numFeatures)

    objects = {}
    for object in pairs:
      objects = exp.addObject(object, objects=objects)

    exp.learnObjects(objects)

    testResult = testOnSingleRandomSDR(objects, exp)
    print testResult

    if result is None:
      result = pd.DataFrame.from_dict([testResult])
    else:
      result = pd.concat([result, pd.DataFrame.from_dict([testResult])])

  plotResults(result, xaxis="numObjects",
              filename='plots/single_column_capacity_varying_object_num.pdf')


if __name__ == "__main__":
  # Varying number of pts per objects, two objects
  runCapacityTestTwoObjectsVaryingObjectSize()

  # 10 pts per object, varying number of objects
  runCapacityTestVaryingObjectNum()