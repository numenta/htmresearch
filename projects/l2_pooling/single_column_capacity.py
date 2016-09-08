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

from htmresearch.frameworks.layers.object_machine_factory import (
  createObjectMachine
)
from htmresearch.frameworks.layers.l2_l4_inference import L4L2Experiment


NUM_LOCATIONS = 500
NUM_FEATURES = 500

def getL4Params():
  """
  Returns a good default set of parameters to use in the L4 region.
  """
  return {
    "columnCount": 2048,
    "cellsPerColumn": 8,
    "formInternalConnections": 0,
    "formInternalBasalConnections": 0,  # inconsistency between CPP and PY
    "learningMode": 1,
    "inferenceMode": 1,
    "learnOnOneCell": 0,
    "initialPermanence": 0.51,
    "connectedPermanence": 0.6,
    "permanenceIncrement": 0.1,
    "permanenceDecrement": 0.02,
    "minThreshold": 10,
    "predictedSegmentDecrement": 0.002,
    "activationThreshold": 13,
    "maxNewSynapseCount": 20,
    "monitor": 0,
    "implementation": "cpp",
  }



def getL2Params():
  """
  Returns a good default set of parameters to use in the L4 region.
  """
  return {
    "columnCount": 1024,
    "inputWidth": 2048 * 8,
    "learningMode": 1,
    "inferenceMode": 1,
    "initialPermanence": 0.41,
    "connectedPermanence": 0.5,
    "permanenceIncrement": 0.1,
    "permanenceDecrement": 0.02,
    "numActiveColumnsPerInhArea": 40,
    "synPermProximalInc": 0.1,
    "synPermProximalDec": 0.001,
    "initialProximalPermanence": 0.6,
    "minThreshold": 3,
    "predictedSegmentDecrement": 0.002,
    "activationThreshold": 3,
    "maxNewSynapseCount": 5,
  }



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
  """
  Check whether a trained L4-L2 network can successfully retrieve an object
  based on a sequence of (feature, location) pairs on this object

  :param objects: list of lists of (feature, location) pairs for all objects
  :param exp: L4L2Experiment instance with a trained network
  :param testObject: the index for the object being tested
  :param numTestPoints: number of test points on the test object
  :return:
  """
  numObjects = len(objects.getObjects())
  numPointsPerObject = len(objects.getObjects()[0])

  testPts = np.random.choice(np.arange(numPointsPerObject),
                             (numTestPoints,),
                             replace=False)
  testPairs = [objects[testObject][i] for i in testPts]

  exp._unsetLearningMode()
  exp.sendReset()

  overlap = np.zeros((numTestPoints, numObjects))

  for step in xrange(numTestPoints):

    locationIdx, featureIdx = testPairs[step]
    feature = objects.features[0][featureIdx]
    location = objects.locations[0][locationIdx]
    exp.sensorInputs[0].addDataToQueue(list(feature), 0, 0)
    exp.externalInputs[0].addDataToQueue(list(location), 0, 0)
    exp.network.run(1)
    for obj in range(numObjects):
      overlap[step, obj] = (len(exp.objectL2Representations[obj][0]
                                & exp.getL2Representations()[0]))

  inferConfig = {
    "numSteps": 5,
    "pairs": {
      0: objects[testObject]
    }
  }
  sensationList = objects.provideObjectToInfer(inferConfig)
  overlap = np.zeros((len(sensationList), numObjects))
  step = 0
  exp._unsetLearningMode()
  exp.sendReset()

  for sensations in sensationList:
    # feed all columns with sensations
    for col in xrange(exp.numColumns):
      location, feature = sensations[col]
      exp.sensorInputs[col].addDataToQueue(list(feature), 0, 0)
      exp.externalInputs[col].addDataToQueue(list(location), 0, 0)
    exp.network.run(1)
    L2Representation = exp.getL2Representations()
    for obj in range(numObjects):
      overlap[step, obj] = (len(exp.objectL2Representations[obj][0]
                                & L2Representation[0]))
    step += 1


  inferConfig = {
    "numSteps": 5,
    "pairs": {
      0: objects[testObject]
    }
  }
  exp.infer(objects.provideObjectToInfer(inferConfig), objectName=0)
  return overlap



def testOnSingleRandomSDR(objects, exp, numRepeats=100):
  """
  Test a trained L4-L2 network on (feature, location) pairs multiple times
  Compute object retrieval accuracy, overlap with the correct and incorrect
  objects

  :param objects: list of lists of (feature, location) pairs for all objects
  :param exp: L4L2Experiment instance with a trained network
  :param numRepeats: number of repeats
  :return: a set of metrics for retrieval accuracy
  """

  numObjects = len(objects.getObjects())
  numPointsPerObject = len(objects.getObjects()[0])
  overlapTrueObj = np.zeros((numRepeats, ))
  confusion = np.zeros((numRepeats,))
  outcome = np.zeros((numRepeats,))

  for i in range(numRepeats):
    targetObject = np.random.choice(np.arange(numObjects))

    nonTargetObjs = range(numObjects)
    nonTargetObjs.remove(targetObject)
    nonTargetObjs = np.array(nonTargetObjs)

    overlap = testNetworkWithOneObject(objects, exp, targetObject, 3)
    outcome[i] = 1 if np.argmax(overlap[-1, :]) == targetObject else 0
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



def plotResults(result, ax=None, xaxis="numPointsPerObject",
                filename=None, marker='-bo'):
  if xaxis == "numPointsPerObject":
    x = result.numPointsPerObject
    xlabel = "# Pts / Obj"
  elif xaxis == "numObjects":
    x = result.numObjects
    xlabel = "Object #"

  if ax is None:
    fig, ax = plt.subplots(2, 2)
  ax[0, 0].plot(x, result.accuracy, marker)
  ax[0, 0].set_ylabel("Accuracy")
  ax[0, 0].set_xlabel(xlabel)

  ax[0, 1].plot(x, result.l2OverlapMean, marker)
  ax[0, 1].set_ylabel("L2 Overlap")
  ax[0, 1].set_xlabel("# Pts / Obj")
  ax[0, 1].set_ylim([0, 41])
  ax[0, 1].set_xlabel(xlabel)

  ax[1, 0].plot(x, result.overlapTrueObj, marker)
  ax[1, 0].set_ylabel("OverlapTrueObject")
  ax[1, 0].set_ylim([0, 41])
  ax[1, 0].set_xlabel(xlabel)

  ax[1, 1].plot(x, result.confusion, marker)
  ax[1, 1].set_ylabel("OverlapFalseObject")
  ax[1, 1].set_ylim([0, 41])
  ax[1, 1].set_xlabel(xlabel)

  plt.tight_layout()
  if filename is not None:
    plt.savefig(filename)



def runCapacityTest(numObjects,
                    numPointsPerObject,
                    maxNewSynapseCount,
                    activationThreshold):
  """
  Generate [numObjects] objects with [numPointsPerObject] points per object
  Train L4-l2 network all the objects with single pass learning
  Test on (feature, location) pairs and compute

  :param numObjects:
  :param numPointsPerObject:
  :param maxNewSynapseCount:
  :param activationThreshold:
  :return:
  """
  l4Params = getL4Params()
  l2Params = getL2Params()
  l2Params['maxNewSynapseCount'] = maxNewSynapseCount
  l2Params['activationThreshold'] = activationThreshold
  l2Params['minThreshold'] = activationThreshold

  objects = createObjectMachine(
    machineType="simple",
    numInputBits=20,
    sensorInputSize=1024,
    externalInputSize=1024,
    numCorticalColumns=1,
    numLocations=NUM_LOCATIONS,
    numFeatures=NUM_FEATURES
  )

  exp = L4L2Experiment("capacity_two_objects",
                       numInputBits=int(l4Params["columnCount"]*0.02),
                       L4Overrides=l4Params,
                       L2Overrides=l2Params,
                       numLearningPoints=10)

  exp = L4L2Experiment("capacity_two_objects",
                       numInputBits=int(l4Params["columnCount"]*0.02),
                       numLearningPoints=10)


  pairs = createRandomObjects(
    numObjects, numPointsPerObject, NUM_LOCATIONS, NUM_FEATURES)
  for object in pairs:
    objects.addObject(object)

  exp.learnObjects(objects.provideObjectsToLearn())

  inferConfig = {
    "numSteps": 5,
    "pairs": {
      0: objects[1]
    }
  }
  exp.infer(objects.provideObjectToInfer(inferConfig), objectName=0)
  exp.getInferenceStats()

  testResult = testOnSingleRandomSDR(objects, exp)
  return testResult



# def runSimulatedCapacityTest(numObjects,
#                     numPointsPerObject,
#                     maxNewSynapseCount,
#                     activationThreshold):
#   """
#   Generate [numObjects] objects with [numPointsPerObject] points per object
#   Create a set of simulated L2 neurons that randomly sample from L4 SDRs
#
#   Test on (feature, location) pairs and compute
#
#   :param numObjects:
#   :param numPointsPerObject:
#   :param maxNewSynapseCount:
#   :param activationThreshold:
#   :return:
#   """
#   l4Params = getL4Params()
#   l2Params = getL2Params()
#   l2Params['maxNewSynapseCount'] = maxNewSynapseCount
#   l2Params['activationThreshold'] = activationThreshold
#   l2Params['minThreshold'] = activationThreshold
#
#   exp = L4L2Experiment("capacity_two_objects",
#                        numInputBits=int(l4Params["columnCount"]*0.02),
#                        L4Overrides=l4Params,
#                        L2Overrides=l2Params,
#                        numLearningPoints=1)
#
#   numLocations = len(exp.locations[0])
#   numFeatures = len(exp.features[0])
#
#   pairs = createRandomObjects(
#     numObjects, numPointsPerObject, numLocations, numFeatures)
#
#   objects = {}
#   for object in pairs:
#     objects = exp.addObject(object, objects=objects)
#
#   for object, pairs in objects.iteritems():
#     for col in xrange(exp.numColumns):
#       locationID, featureID = pairs[col]
#       feature = exp.features[col][featureID]
#
#       # generate random location if requested
#       if locationID == -1:
#         location = list(exp.generatePattern(exp.numInputBits,
#                                             exp.config["sensorInputSize"]))
#       # generate union of locations if requested
#       elif isinstance(locationID, tuple):
#         location = set()
#         for idx in list(locationID):
#           location = location | exp.locations[col][idx]
#         location = list(location)
#       else:
#         location = exp.locations[col][locationID]



def runCapacityTestVaryingObjectSize(numObjects=2,
                                               maxNewSynapseCount=5,
                                               activationThreshold=3):
  """
  Runs experiment with two objects, varying number of points per object
  """

  result = None
  for numPointsPerObject in [5, 10, 20, 30, 40, 50, 60, 80]:
    testResult = runCapacityTest(
      numObjects, numPointsPerObject, maxNewSynapseCount, activationThreshold)
    print testResult
    if result is None:
      result = pd.DataFrame.from_dict([testResult])
    else:
      result = pd.concat([result, pd.DataFrame.from_dict([testResult])])

  resultFileName = 'plots/single_column_capacity_varying_object_size_' \
                   'synapses_{}_thresh_{}'.format(
    maxNewSynapseCount, activationThreshold)

  pd.DataFrame.to_csv(result, resultFileName + '.csv')
  plotResults(result, None,
              xaxis="numPointsPerObject", filename=resultFileName+'.pdf')



def runCapacityTestVaryingObjectNum(numPointsPerObject=10,
                                    maxNewSynapseCount=5,
                                    activationThreshold=3):
  """
  Run experiment with fixed number of pts per object, varying number of objects

  """
  result = None

  for numObjects in [2, 3, 5, 10, 15, 20, 30, 40, 50, 60]:

    testResult = runCapacityTest(
      numObjects, numPointsPerObject, maxNewSynapseCount, activationThreshold)

    print testResult

    if result is None:
      result = pd.DataFrame.from_dict([testResult])
    else:
      result = pd.concat([result, pd.DataFrame.from_dict([testResult])])

  resultFileName = 'plots/single_column_capacity_varying_object_num_' \
                   'synapses_{}_thresh_{}'.format(
    maxNewSynapseCount, activationThreshold)
  pd.DataFrame.to_csv(result, resultFileName + '.csv')

  plotResults(result, None, xaxis="numObjects",
              filename=resultFileName + '.pdf')



def runExperiment1():
  """
  Varying number of pts per objects, two objects
  Try different sampling and activation threshold
  """
  numObjects = 2
  for maxNewSynapseCount in [20]: #[5, 10, 20, 40]:
    activationThreshold=int(.6*maxNewSynapseCount)

    print "maxNewSynapseCount: {} \nactivationThreshold: {} \n".format(
      maxNewSynapseCount, activationThreshold)

    runCapacityTestVaryingObjectSize(numObjects,
                                     maxNewSynapseCount,
                                     activationThreshold)

  markers = ['-bo', '-ro', '-co', '-go']
  ploti = 0
  fig, ax = plt.subplots(2, 2)
  legendEntries = []
  for maxNewSynapseCount in [5, 10, 20, 40]:
    activationThreshold = int(.6 * maxNewSynapseCount)
    resultFileName = 'plots/single_column_capacity_varying_object_size_' \
                     'synapses_{}_thresh_{}'.format(
      maxNewSynapseCount, activationThreshold)
    result = pd.read_csv(resultFileName+ '.csv')

    plotResults(result, ax, "numPointsPerObject", None, markers[ploti])
    ploti += 1
    legendEntries.append("# syn {}".format(maxNewSynapseCount))
  plt.legend(legendEntries)
  plt.savefig('plots/single_column_capacity_varying_object_size_summary.pdf')



def runExperiment2():
  """
  runCapacityTestVaryingObjectNum()
  Try different sampling and activation threshold
  """
  numPointsPerObject = 10
  for maxNewSynapseCount in [5, 10, 20, 40]:
    activationThreshold=int(.6*maxNewSynapseCount)

    print "maxNewSynapseCount: {} \nactivationThreshold: {} \n".format(
      maxNewSynapseCount, activationThreshold)

    runCapacityTestVaryingObjectNum(numPointsPerObject,
                                    maxNewSynapseCount,
                                    activationThreshold)

  markers = ['-bo', '-ro', '-co', '-go']
  ploti = 0
  fig, ax = plt.subplots(2, 2)
  legendEntries = []
  for maxNewSynapseCount in [5, 10, 20, 40]:
    activationThreshold = int(.6 * maxNewSynapseCount)
    resultFileName = 'plots/single_column_capacity_varying_object_num_' \
                     'synapses_{}_thresh_{}'.format(
      maxNewSynapseCount, activationThreshold)
    result = pd.read_csv(resultFileName+ '.csv')

    plotResults(result, ax, "numObjects", None, markers[ploti])
    ploti += 1
    legendEntries.append("# syn {}".format(maxNewSynapseCount))
  plt.legend(legendEntries, loc=4)
  plt.savefig('plots/single_column_capacity_varying_object_num_summary.pdf')



if __name__ == "__main__":
  # Varying number of pts per objects, two objects
  runExperiment1()

  # 10 pts per object, varying number of objects
  # runExperiment2()