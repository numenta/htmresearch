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
This file tests the capacity of L4-L2 columns

In this test, we consider a set of objects without any shared (feature,
location) pairs and without any noise. One, or more, L4-L2 columns is trained
on all objects.

In the test phase, we randomly pick a (feature, location) SDR and feed it to
the network, and asked whether the correct object can be retrieved.
"""

import argparse
import multiprocessing
import os
import os.path
from pprint import pprint

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from htmresearch.frameworks.layers.object_machine_factory import (
  createObjectMachine
)
from htmresearch.frameworks.layers.l2_l4_inference import L4L2Experiment

import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42

DEFAULT_NUM_LOCATIONS = 5000
DEFAULT_NUM_FEATURES = 5000
DEFAULT_RESULT_DIR_NAME = "results"
DEFAULT_PLOT_DIR_NAME = "plots"
DEFAULT_NUM_CORTICAL_COLUMNS = 1
DEFAULT_COLORS = ("b", "r", "c", "g", 'm', 'y', 'w', 'k')



def _prepareResultsDir(resultBaseName, resultDirName=DEFAULT_RESULT_DIR_NAME):
  """
  Ensures that the requested resultDirName exists.  Attempt to create it if not.
  Returns the combined absolute path to result.
  """
  resultDirName = os.path.abspath(resultDirName)
  resultFileName = os.path.join(resultDirName, resultBaseName)

  try:
    if not os.path.isdir(resultDirName):
      # Directory does not exist, attempt to create recursively
      os.makedirs(resultDirName)
  except os.error:
    # Unlikely, but directory may have been created already.  Double check to
    # make sure it's safe to ignore error in creation
    if not os.path.isdir(resultDirName):
      raise Exception("Unable to create results directory at {}"
                      .format(resultDirName))

  return resultFileName



def getL4Params():
  """
  Returns a good default set of parameters to use in the L4 region.
  """
  return {
    "columnCount": 150,
    "cellsPerColumn": 16,
    "learn": True,
    "learnOnOneCell": False,
    "initialPermanence": 0.51,
    "connectedPermanence": 0.6,
    "permanenceIncrement": 0.1,
    "permanenceDecrement": 0.02,
    "minThreshold": 10,
    "basalPredictedSegmentDecrement": 0.0,
    "activationThreshold": 13,
    "sampleSize": 25,
    "implementation": "ApicalTiebreakCPP",
  }



def getL2Params():
  """
  Returns a good default set of parameters to use in the L4 region.
  """
  return {
    "inputWidth": 2048 * 8,
    "cellCount": 4096,
    "sdrSize": 40,
    "synPermProximalInc": 0.1,
    "synPermProximalDec": 0.001,
    "initialProximalPermanence": 0.6,
    "minThresholdProximal": 6,
    "sampleSizeProximal": 10,
    "connectedPermanenceProximal": 0.5,
    "synPermDistalInc": 0.1,
    "synPermDistalDec": 0.001,
    "initialDistalPermanence": 0.41,
    "activationThresholdDistal": 18,
    "sampleSizeDistal": 20,
    "connectedPermanenceDistal": 0.5,
    "learningMode": True,
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
    replace=False
  )

  randomFeatureLocPairs = (divmod(idx, numFeatures) for idx in randomPairIdx)

  # Return sequences of random feature-location pairs.  Each sequence will
  # contain a number of pairs defined by 'numPointsPerObject'
  return zip(*[iter(randomFeatureLocPairs)] * numPointsPerObject)



def createRandomObjectsSharedPairs(numObjects,
                                   numPointsPerObject,
                                   numLocations,
                                   numFeatures):
  """
  Create numObjects. (feature, location) pairs may be shared.
  :param numObjects: number of objects
  :param numPointsPerObject: number of (feature, location) pairs per object
  :param numLocations: number of unique locations
  :param numFeatures: number of unique features
  :return:   (list(list(tuple))  List of lists of feature / location pairs.
  """
  locations = np.arange(numLocations)
  features = np.arange(numFeatures)

  objects = []
  objectsSets = set()
  for _ in xrange(numObjects):
    objectLocations = np.random.choice(locations, numPointsPerObject,
                                       replace=False)
    objectFeatures = np.random.choice(features, numPointsPerObject,
                                      replace=True)

    o = zip(objectLocations, objectFeatures)

    # Make sure this is a unique object.
    objectAsSet = frozenset(o)
    assert objectAsSet not in objectsSets
    objectsSets.add(objectAsSet)

    objects.append(o)

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
  innerObjs = objects.getObjects()
  numObjects = len(innerObjs)
  numPointsPerObject = len(innerObjs[0])

  testPts = np.random.choice(np.arange(numPointsPerObject),
                             (numTestPoints,),
                             replace=False)

  testPairs = [objects[testObject][i] for i in testPts]

  exp._unsetLearningMode()
  exp.sendReset()

  overlap = np.zeros((numTestPoints, numObjects))
  numL2ActiveCells = np.zeros((numTestPoints))
  numL4ActiveCells = np.zeros((numTestPoints))

  for step, pair in enumerate(testPairs):
    (locationIdx, featureIdx) = pair
    for colIdx in xrange(exp.numColumns):
      feature = objects.features[colIdx][featureIdx]
      location = objects.locations[colIdx][locationIdx]

      exp.sensorInputs[colIdx].addDataToQueue(list(feature), 0, 0)
      exp.externalInputs[colIdx].addDataToQueue(list(location), 0, 0)

    exp.network.run(1)

    for colIdx in xrange(exp.numColumns):
      numL2ActiveCells[step] += float(len(exp.getL2Representations()[colIdx]))
      numL4ActiveCells[step] += float(len(exp.getL4Representations()[colIdx]))

    numL2ActiveCells[step] /= exp.numColumns
    numL4ActiveCells[step] /= exp.numColumns

    overlapByColumn = exp.getCurrentObjectOverlaps()
    overlap[step] = np.mean(overlapByColumn, axis=0)

    # columnPooler = exp.L2Columns[0]._pooler
    # tm = exp.L4Columns[0]._tm
    # print "step : {}".format(step)
    # print "{} L4 cells predicted : ".format(
    #   len(exp.getL4PredictedCells()[0])), exp.getL4PredictedeCells()
    # print "{} L4 cells active : ".format(len(exp.getL4Representations()[0])), exp.getL4Representations()
    # print "L2 activation: ", columnPooler.getActiveCells()
    # print "overlap : ", (overlap[step, :])

  return overlap, numL2ActiveCells, numL4ActiveCells



def testOnSingleRandomSDR(objects, exp, numRepeats=100, repeatID=0):
  """
  Test a trained L4-L2 network on (feature, location) pairs multiple times
  Compute object retrieval accuracy, overlap with the correct and incorrect
  objects

  :param objects: list of lists of (feature, location) pairs for all objects
  :param exp: L4L2Experiment instance with a trained network
  :param numRepeats: number of repeats
  :return: a set of metrics for retrieval accuracy
  """

  innerObjs = objects.getObjects()

  numObjects = len(innerObjs)
  numPointsPerObject = len(innerObjs[0])
  overlapTrueObj = np.zeros((numRepeats,))
  l2ActivationSize = np.zeros((numRepeats,))
  l4ActivationSize = np.zeros((numRepeats,))
  confusion = overlapTrueObj.copy()
  outcome = overlapTrueObj.copy()

  columnPooler = exp.L2Columns[0]._pooler
  numConnectedProximal = columnPooler.numberOfConnectedProximalSynapses()
  numConnectedDistal = columnPooler.numberOfConnectedDistalSynapses()

  result = None
  for i in xrange(numRepeats):
    targetObject = np.random.choice(np.arange(numObjects))

    nonTargetObjs = np.array(
      [obj for obj in xrange(numObjects) if obj != targetObject]
    )

    overlap, numActiveL2Cells, numL4ActiveCells = testNetworkWithOneObject(
      objects,
      exp,
      targetObject,
      10
    )

    lastOverlap = overlap[-1, :]

    maxOverlapIndices = (
      np.where(lastOverlap == lastOverlap[np.argmax(lastOverlap)])[0].tolist()
    )

    # Only set to 1 iff target object is the lone max overlap index.  Otherwise
    # the network failed to conclusively identify the target object.
    outcome[i] = 1 if maxOverlapIndices == [targetObject] else 0
    confusion[i] = np.max(lastOverlap[nonTargetObjs])
    overlapTrueObj[i] = lastOverlap[targetObject]
    l2ActivationSize[i] = numActiveL2Cells[-1]
    l4ActivationSize[i] = numL4ActiveCells[-1]

    testResult = {
      "repeatID": repeatID,
      "repeatI": i,
      "numberOfConnectedProximalSynapses": numConnectedProximal,
      "numberOfConnectedDistalSynapses": numConnectedDistal,
      "numObjects": numObjects,
      "numPointsPerObject": numPointsPerObject,
      "l2ActivationSize": l2ActivationSize[i],
      "l4ActivationSize": l4ActivationSize[i],
      "confusion": confusion[i],
      "accuracy":  outcome[i],
      "overlapTrueObj": overlapTrueObj[i]}
    result = (
      pd.concat([result, pd.DataFrame.from_dict([testResult])])
      if result is not None else
      pd.DataFrame.from_dict([testResult])
    )

  return result




def plotResults(result, ax=None, xaxis="numObjects",
                filename=None, marker='-bo', confuseThresh=30, showErrBar=1):

  if ax is None:
    fig, ax = plt.subplots(2, 2)

  numRpts = max(result['repeatID']) + 1
  resultsRpts = result.groupby(['repeatID'])

  if xaxis == "numPointsPerObject":
    x = np.array(resultsRpts.get_group(0).numPointsPerObject)
    xlabel = "# Pts / Obj"
    x = np.unique(x)
    d = resultsRpts.get_group(0)
    d = d.groupby(['numPointsPerObject'])
  elif xaxis == "numObjects":
    x = np.array(resultsRpts.get_group(0).numObjects)
    xlabel = "Object #"
    x = np.unique(x)
    d = resultsRpts.get_group(0)
    d = d.groupby(['numObjects'])
  accuracy = np.zeros((1, len(x),))
  numberOfConnectedProximalSynapses = np.zeros((1, len(x),))
  l2ActivationSize = np.zeros((1, len(x),))
  confusion = np.zeros((1, len(x),))
  for j in range(len(x)):
    accuracy[0,j] = np.sum(np.logical_and(d.get_group(x[j]).accuracy == 1,
                                        d.get_group(x[j]).confusion < confuseThresh)) / \
                  float(len(d.get_group(x[j]).accuracy))
    l2ActivationSize[0,j] = np.mean(d.get_group(x[j]).l2ActivationSize)
    confusion[0,j] = np.mean(d.get_group(x[j]).confusion)
    numberOfConnectedProximalSynapses[0,j] = np.mean(
      d.get_group(x[j]).numberOfConnectedProximalSynapses)

  if ax is None:
    fig, ax = plt.subplots(2, 2)

  for rpt in range(1, numRpts):
    d = resultsRpts.get_group(rpt)
    d = d.groupby(['numObjects'])
    accuracyRpt = np.zeros((1, len(x)))
    numberOfConnectedProximalSynapsesRpt = np.zeros((1, len(x)))
    l2ActivationSizeRpt = np.zeros((1, len(x)))
    confusionRpt = np.zeros((1, len(x)))

    for j in range(len(x)):
      accuracyRpt[0,j] = np.sum(np.logical_and(
        d.get_group(x[j]).accuracy == 1,
        d.get_group(x[j]).confusion < confuseThresh)) / \
                       float(len(d.get_group(x[j]).accuracy))

      l2ActivationSizeRpt[0,j] = np.mean(d.get_group(x[j]).l2ActivationSize)
      confusionRpt[0,j] = np.mean(d.get_group(x[j]).confusion)
      numberOfConnectedProximalSynapsesRpt[0,j] = np.mean(
        d.get_group(x[j]).numberOfConnectedProximalSynapses)

    accuracy = np.vstack((accuracy, accuracyRpt))
    confusion = np.vstack((confusion, confusionRpt))
    l2ActivationSize = np.vstack((l2ActivationSize, l2ActivationSizeRpt))
    numberOfConnectedProximalSynapses = np.vstack((
      numberOfConnectedProximalSynapses, numberOfConnectedProximalSynapsesRpt))

  if showErrBar==0:
    s = 0
  else:
    s = 1
  ax[0, 0].errorbar(x, np.mean(accuracy, 0), yerr=np.std(accuracy, 0)*s,
                    color=marker)
  ax[0, 0].set_ylabel("Accuracy")
  ax[0, 0].set_xlabel(xlabel)
  ax[0, 0].set_ylim([0.1, 1.05])
  ax[0, 0].set_xlim([0, 820])

  ax[0, 1].errorbar(x, np.mean(numberOfConnectedProximalSynapses, 0),
                    yerr=np.std(numberOfConnectedProximalSynapses, 0)*s, color=marker)
  ax[0, 1].set_ylabel("# connected proximal synapses")
  ax[0, 1].set_xlabel("# Pts / Obj")
  ax[0, 1].set_xlabel(xlabel)
  ax[0, 1].set_xlim([0, 820])

  ax[1, 0].errorbar(x, np.mean(l2ActivationSize, 0), yerr=np.std(l2ActivationSize, 0)*s, color=marker)
  ax[1, 0].set_ylabel("l2ActivationSize")
  # ax[1, 0].set_ylim([0, 41])
  ax[1, 0].set_xlabel(xlabel)
  ax[1, 0].set_xlim([0, 820])

  ax[1, 1].errorbar(x, np.mean(confusion, 0), yerr=np.std(confusion, 0)*s, color=marker)
  ax[1, 1].set_ylabel("OverlapFalseObject")
  ax[1, 1].set_ylim([0, 41])
  ax[1, 1].set_xlabel(xlabel)
  ax[1, 1].set_xlim([0, 820])

  plt.tight_layout()
  if filename is not None:
    plt.savefig(filename)



def runCapacityTest(numObjects,
                    numPointsPerObject,
                    numCorticalColumns,
                    l2Params,
                    l4Params,
                    objectParams,
                    networkType = "MultipleL4L2Columns",
                    repeat=0):
  """
  Generate [numObjects] objects with [numPointsPerObject] points per object
  Train L4-l2 network all the objects with single pass learning
  Test on (feature, location) pairs and compute

  :param numObjects:
  :param numPointsPerObject:
  :param sampleSize:
  :param activationThreshold:
  :param numCorticalColumns:
  :return:
  """
  l4ColumnCount = l4Params["columnCount"]

  numInputBits = objectParams['numInputBits']
  externalInputSize = objectParams['externalInputSize']

  if numInputBits is None:
    numInputBits = int(l4ColumnCount * 0.02)

  numLocations = objectParams["numLocations"]
  numFeatures = objectParams["numFeatures"]

  objects = createObjectMachine(
    machineType="simple",
    numInputBits=numInputBits,
    sensorInputSize=l4ColumnCount,
    externalInputSize=externalInputSize,
    numCorticalColumns=numCorticalColumns,
    numLocations=numLocations,
    numFeatures=numFeatures
  )

  exp = L4L2Experiment("capacity_two_objects",
                       numInputBits=numInputBits,
                       L2Overrides=l2Params,
                       L4Overrides=l4Params,
                       inputSize=l4ColumnCount,
                       networkType = networkType,
                       externalInputSize=externalInputSize,
                       numLearningPoints=3,
                       numCorticalColumns=numCorticalColumns,
                       objectNamesAreIndices=True)

  if objectParams["uniquePairs"]:
    pairs = createRandomObjects(
      numObjects,
      numPointsPerObject,
      numLocations,
      numFeatures
    )
  else:
    pairs = createRandomObjectsSharedPairs(
      numObjects,
      numPointsPerObject,
      numLocations,
      numFeatures
    )

  for object in pairs:
    objects.addObject(object)

  exp.learnObjects(objects.provideObjectsToLearn())

  testResult = testOnSingleRandomSDR(objects, exp, 100, repeat)
  return testResult




def runCapacityTestVaryingObjectSize(
    numObjects=2,
    numCorticalColumns=DEFAULT_NUM_CORTICAL_COLUMNS,
    resultDirName=DEFAULT_RESULT_DIR_NAME,
    expName=None,
    cpuCount=None,
    l2Params=None,
    l4Params=None,
    objectParams=None):
  """
  Runs experiment with two objects, varying number of points per object
  """

  result = None

  cpuCount = cpuCount or multiprocessing.cpu_count()
  pool = multiprocessing.Pool(cpuCount, maxtasksperchild=1)

  l4Params = l4Params or getL4Params()
  l2Params = l2Params or getL2Params()

  params = [(numObjects,
             numPointsPerObject,
             numCorticalColumns,
             l2Params,
             l4Params,
             objectParams,
             0)
            for numPointsPerObject in np.arange(10, 160, 20)]

  for testResult in pool.map(invokeRunCapacityTest, params):
    result = (
      pd.concat([result, testResult])
      if result is not None else testResult
    )

  resultFileName = _prepareResultsDir(
    "{}.csv".format(expName),
    resultDirName=resultDirName
  )

  pd.DataFrame.to_csv(result, resultFileName)



def invokeRunCapacityTest(params):
  """ Splits out params so that runCapacityTest may be invoked with
  multiprocessing.Pool.map() to support parallelism
  """
  return runCapacityTest(*params)



def runCapacityTestVaryingObjectNum(numPointsPerObject=10,
                                    numCorticalColumns=DEFAULT_NUM_CORTICAL_COLUMNS,
                                    resultDirName=DEFAULT_RESULT_DIR_NAME,
                                    expName=None,
                                    cpuCount=None,
                                    l2Params=None,
                                    l4Params=None,
                                    objectParams=None,
                                    networkType="MultipleL4L2Columns",
                                    numRpts=1):
  """
  Run experiment with fixed number of pts per object, varying number of objects
  """

  l4Params = l4Params or getL4Params()
  l2Params = l2Params or getL2Params()

  cpuCount = cpuCount or multiprocessing.cpu_count()
  pool = multiprocessing.Pool(cpuCount, maxtasksperchild=1)

  numObjectsList = np.arange(50, 1300, 100)
  params = []
  for rpt in range(numRpts):
    for numObjects in numObjectsList:
      params.append((numObjects,
                     numPointsPerObject,
                     numCorticalColumns,
                     l2Params,
                     l4Params,
                     objectParams,
                     networkType,
                     rpt))
  result = None
  for testResult in pool.map(invokeRunCapacityTest, params):
    result = (
      pd.concat([result, testResult])
      if result is not None else testResult
    )

  resultFileName = _prepareResultsDir("{}.csv".format(expName),
                                      resultDirName=resultDirName)
  pd.DataFrame.to_csv(result, resultFileName)


def invokeRunCapacityTest(params):
  """ Splits out params so that runCapacityTest may be invoked with
  multiprocessing.Pool.map() to support parallelism
  """
  return runCapacityTest(*params)



def runCapacityTestWrapperNonParallel(numPointsPerObject=10,
                                      numCorticalColumns=DEFAULT_NUM_CORTICAL_COLUMNS,
                                      resultDirName=DEFAULT_RESULT_DIR_NAME,
                                      numObjects = 100,
                                      expName=None,
                                      l2Params=None,
                                      l4Params=None,
                                      objectParams=None,
                                      networkType="MultipleL4L2Columns",
                                      rpt=0):
  """
  Run experiment with fixed number of pts per object, varying number of objects
  """

  l4Params = l4Params or getL4Params()
  l2Params = l2Params or getL2Params()

  testResult = runCapacityTest(numObjects,
                               numPointsPerObject,
                               numCorticalColumns,
                               l2Params,
                               l4Params,
                               objectParams,
                               networkType,
                               rpt)


  resultFileName = _prepareResultsDir("{}.csv".format(expName),
                                      resultDirName=resultDirName)

  if os.path.isfile(resultFileName):
    pd.DataFrame.to_csv(testResult, resultFileName, mode = "a", header = False)
  else:
    pd.DataFrame.to_csv(testResult, resultFileName, mode = "a", header = True)

def invokeRunCapacityTestWrapper(params):
  """ Splits out params so that runCapacityTest may be invoked with
  multiprocessing.Pool.map() to support parallelism
  """
  return runCapacityTestWrapperNonParallel(*params)

def runExperiment1(numObjects=2,
                   sampleSizeRange=(10,),
                   numCorticalColumns=DEFAULT_NUM_CORTICAL_COLUMNS,
                   resultDirName=DEFAULT_RESULT_DIR_NAME,
                   plotDirName=DEFAULT_PLOT_DIR_NAME,
                   cpuCount=None):
  """
  Varying number of pts per objects, two objects
  Try different sample sizes
  """
  objectParams = {'numInputBits': 20,
                  'externalInputSize': 2400,
                  'numFeatures': DEFAULT_NUM_FEATURES,
                  'numLocations': DEFAULT_NUM_LOCATIONS,
                  'uniquePairs': True,}
  l4Params = getL4Params()
  l2Params = getL2Params()

  numInputBits = objectParams['numInputBits']

  l4Params["activationThreshold"] = int(numInputBits * .6)
  l4Params["minThreshold"] = int(numInputBits * .6)
  l4Params["sampleSize"] = int(2 * l4Params["activationThreshold"])

  for sampleSize in sampleSizeRange:
    print "sampleSize: {}".format(sampleSize)
    l2Params['sampleSizeProximal'] = sampleSize
    expName = "capacity_varying_object_size_synapses_{}".format(sampleSize)
    runCapacityTestVaryingObjectSize(numObjects,
                                     numCorticalColumns,
                                     resultDirName,
                                     expName,
                                     cpuCount,
                                     l2Params,
                                     l4Params,
                                     objectParams=objectParams)

  ploti = 0
  fig, ax = plt.subplots(2, 2)
  st = fig.suptitle(
    "Varying points per object x 2 objects ({} cortical column{})"
      .format(numCorticalColumns, "s" if numCorticalColumns > 1 else ""
              ), fontsize="x-large")

  legendEntries = []

  for sampleSize in sampleSizeRange:
    expName = "capacity_varying_object_size_synapses_{}".format(sampleSize)

    resultFileName = _prepareResultsDir("{}.csv".format(expName),
                                        resultDirName=resultDirName
                                        )
    result = pd.read_csv(resultFileName)

    plotResults(result, ax, "numPointsPerObject", None, DEFAULT_COLORS[ploti])
    ploti += 1
    legendEntries.append("# sample size {}".format(sampleSize))

  plt.legend(legendEntries, loc=2)
  fig.tight_layout()

  # shift subplots down:
  st.set_y(0.95)
  fig.subplots_adjust(top=0.85)

  plt.savefig(
    os.path.join(
      plotDirName,
      "capacity_varying_object_size_summary.pdf"
    )
  )



def runExperiment2(numCorticalColumns=DEFAULT_NUM_CORTICAL_COLUMNS,
                   resultDirName=DEFAULT_RESULT_DIR_NAME,
                   plotDirName=DEFAULT_PLOT_DIR_NAME,
                   cpuCount=None):
  """
  runCapacityTestVaryingObjectNum()
  Try different sample sizes
  """
  sampleSizeRange = (10,)
  numPointsPerObject = 10
  l4Params = getL4Params()
  l2Params = getL2Params()
  objectParams = {'numInputBits': 20,
                  'externalInputSize': 2400,
                  'numFeatures': DEFAULT_NUM_FEATURES,
                  'numLocations': DEFAULT_NUM_LOCATIONS,
                  'uniquePairs': True,}

  for sampleSize in sampleSizeRange:
    print "sampleSize: {}".format(sampleSize)
    l2Params['sampleSizeProximal'] = sampleSize
    expName = "capacity_varying_object_num_synapses_{}".format(sampleSize)

    print "l4Params: "
    pprint(l4Params)
    print "l2Params: "
    pprint(l2Params)

    runCapacityTestVaryingObjectNum(numPointsPerObject,
                                    numCorticalColumns,
                                    resultDirName,
                                    expName,
                                    cpuCount,
                                    l2Params,
                                    l4Params,
                                    objectParams)

  # plot result
  ploti = 0
  fig, ax = plt.subplots(2, 2)
  st = fig.suptitle(
    "Varying number of objects ({} cortical column{})"
      .format(numCorticalColumns, "s" if numCorticalColumns > 1 else ""
              ), fontsize="x-large"
  )

  for axi in (0, 1):
    for axj in (0, 1):
      ax[axi][axj].xaxis.set_major_locator(ticker.MultipleLocator(50))

  legendEntries = []
  for sampleSize in sampleSizeRange:
    print "sampleSize: {}".format(sampleSize)
    l2Params['sampleSizeProximal'] = sampleSize
    expName = "capacity_varying_object_num_synapses_{}".format(sampleSize)

    resultFileName = _prepareResultsDir(
      "{}.csv".format(expName),
      resultDirName=resultDirName
    )

    result = pd.read_csv(resultFileName)

    plotResults(result, ax, "numObjects", None, DEFAULT_COLORS[ploti])
    ploti += 1
    legendEntries.append("# sample size {}".format(sampleSize))
  ax[0, 0].legend(legendEntries, loc=4, fontsize=8)
  fig.tight_layout()

  # shift subplots down:
  st.set_y(0.95)
  fig.subplots_adjust(top=0.85)

  plt.savefig(
    os.path.join(
      plotDirName,
      "capacity_varying_object_num_summary.pdf"
    )
  )



def runExperiment3(numCorticalColumns=DEFAULT_NUM_CORTICAL_COLUMNS,
                   resultDirName=DEFAULT_RESULT_DIR_NAME,
                   plotDirName=DEFAULT_PLOT_DIR_NAME,
                   cpuCount=None):
  """
  runCapacityTestVaryingObjectNum()
  Try different L4 network size
  """

  numPointsPerObject = 10
  numRpts = 1
  l4Params = getL4Params()
  l2Params = getL2Params()

  l2Params['cellCount'] = 4096
  l2Params['sdrSize'] = 40

  expParams = []
  expParams.append(
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 10,
     'thresh': 5})
  expParams.append(
    {'l4Column': 200, 'externalInputSize': 2400, 'w': 20, 'sample': 10,
     'thresh': 5})
  expParams.append(
    {'l4Column': 250, 'externalInputSize': 2400, 'w': 20, 'sample': 10,
     'thresh': 5})

  for expParam in expParams:
    l4Params["columnCount"] = expParam['l4Column']
    numInputBits = expParam['w']
    l2Params['sampleSizeProximal'] = expParam['sample']
    l2Params['minThresholdProximal'] = expParam['thresh']

    l4Params["activationThreshold"] = int(numInputBits * .6)
    l4Params["minThreshold"] = int(numInputBits * .6)
    l4Params["sampleSize"] = int(2 * l4Params["activationThreshold"])

    objectParams = {'numInputBits': numInputBits,
                    'externalInputSize': expParam['externalInputSize'],
                    'numFeatures': DEFAULT_NUM_FEATURES,
                    'numLocations': DEFAULT_NUM_LOCATIONS,
                    'uniquePairs': True,}

    print "l4Params: "
    pprint(l4Params)
    print "l2Params: "
    pprint(l2Params)

    expname = "multiple_column_capacity_varying_object_num_synapses_{}_thresh_{}_l4column_{}".format(
      expParam['sample'], expParam['thresh'], expParam["l4Column"])
    runCapacityTestVaryingObjectNum(numPointsPerObject,
                                    numCorticalColumns,
                                    resultDirName,
                                    expname,
                                    cpuCount,
                                    l2Params,
                                    l4Params,
                                    objectParams,
                                    "MultipleL4L2Columns",
                                    numRpts)

  # plot result
  ploti = 0
  fig, ax = plt.subplots(2, 2)
  st = fig.suptitle(
    "Varying number of objects ({} cortical column{})"
      .format(numCorticalColumns, "s" if numCorticalColumns > 1 else ""
              ), fontsize="x-large"
  )

  for axi in (0, 1):
    for axj in (0, 1):
      ax[axi][axj].xaxis.set_major_locator(ticker.MultipleLocator(100))

  legendEntries = []
  for expParam in expParams:
    expname = "multiple_column_capacity_varying_object_num_synapses_{}_thresh_{}_l4column_{}".format(
      expParam['sample'], expParam['thresh'], expParam["l4Column"])

    resultFileName = _prepareResultsDir("{}.csv".format(expname),
                                        resultDirName=resultDirName
                                        )

    result = pd.read_csv(resultFileName)

    plotResults(result, ax, "numObjects", None, DEFAULT_COLORS[ploti])
    ploti += 1
    legendEntries.append("L4 mcs {} w {} s {} thresh {}".format(
      expParam["l4Column"], expParam['w'], expParam['sample'],
      expParam['thresh']))
  ax[0, 0].legend(legendEntries, loc=4, fontsize=8)
  fig.tight_layout()

  # shift subplots down:
  st.set_y(0.95)
  fig.subplots_adjust(top=0.85)

  plt.savefig(
    os.path.join(
      plotDirName,
      "capacity_varying_object_num_l4size_summary.pdf"
    )
  )



def runExperiment4(resultDirName=DEFAULT_RESULT_DIR_NAME,
                   plotDirName=DEFAULT_PLOT_DIR_NAME,
                   cpuCount=None):
  """
  runCapacityTestVaryingObjectNum()
  varying number of cortical columns
  """

  numPointsPerObject = 10
  numRpts = 1

  l4Params = getL4Params()
  l2Params = getL2Params()

  expParams = []
  expParams.append(
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 10,
     'thresh': 5, 'l2Column': 1})
  expParams.append(
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 10,
     'thresh': 5, 'l2Column': 2})
  expParams.append(
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 10,
     'thresh': 5, 'l2Column': 3})

  for expParam in expParams:
    l2Params['sampleSizeProximal'] = expParam['sample']
    l2Params['minThresholdProximal'] = expParam['thresh']

    l4Params["columnCount"] = expParam['l4Column']
    numInputBits = expParam['w']
    numCorticalColumns = expParam['l2Column']

    l4Params["activationThreshold"] = int(numInputBits * .6)
    l4Params["minThreshold"] = int(numInputBits * .6)
    l4Params["sampleSize"] = int(2 * l4Params["activationThreshold"])

    objectParams = {'numInputBits': numInputBits,
                    'externalInputSize': expParam['externalInputSize'],
                    'numFeatures': DEFAULT_NUM_FEATURES,
                    'numLocations': DEFAULT_NUM_LOCATIONS,
                    'uniquePairs': True,}

    print "l4Params: "
    pprint(l4Params)
    print "l2Params: "
    pprint(l2Params)

    expName = "multiple_column_capacity_varying_object_num_synapses_{}_thresh_{}_l4column_{}_l2column_{}".format(
      expParam['sample'], expParam['thresh'], expParam["l4Column"],
      expParam['l2Column'])

    runCapacityTestVaryingObjectNum(numPointsPerObject,
                                    numCorticalColumns,
                                    resultDirName,
                                    expName,
                                    cpuCount,
                                    l2Params,
                                    l4Params,
                                    objectParams,
                                    "MultipleL4L2Columns",
                                    numRpts)

  # plot result
  ploti = 0
  fig, ax = plt.subplots(2, 2)
  st = fig.suptitle("Varying number of objects", fontsize="x-large")

  for axi in (0, 1):
    for axj in (0, 1):
      ax[axi][axj].xaxis.set_major_locator(ticker.MultipleLocator(100))

  legendEntries = []
  for expParam in expParams:
    expName = "multiple_column_capacity_varying_object_num_synapses_{}_thresh_{}_l4column_{}_l2column_{}".format(
      expParam['sample'], expParam['thresh'], expParam["l4Column"],
      expParam['l2Column'])

    resultFileName = _prepareResultsDir("{}.csv".format(expName),
                                        resultDirName=resultDirName
                                        )

    result = pd.read_csv(resultFileName)

    plotResults(result, ax, "numObjects", None, DEFAULT_COLORS[ploti])
    ploti += 1
    legendEntries.append("L4 mcs {} #cc {} ".format(
      expParam['l4Column'], expParam['l2Column']))
  ax[0, 0].legend(legendEntries, loc=3, fontsize=8)
  fig.tight_layout()

  # shift subplots down:
  st.set_y(0.95)
  fig.subplots_adjust(top=0.85)

  plt.savefig(
    os.path.join(
      plotDirName,
      "multiple_column_capacity_varying_object_num_and_column_num_summary.pdf"
    )
  )



def runExperiment5(resultDirName=DEFAULT_RESULT_DIR_NAME,
                   plotDirName=DEFAULT_PLOT_DIR_NAME,
                   cpuCount=None):
  """
  varying size of L2
  calculate capacity by varying number of objects with fixed size
  """

  numPointsPerObject = 10
  numRpts = 1
  numInputBits = 20
  externalInputSize = 2400
  numL4MiniColumns = 150
  l4Params = getL4Params()
  l2Params = getL2Params()

  expParams = []
  expParams.append(
    {'L2cellCount': 2048, 'L2activeBits': 40, 'w': 20, 'sample': 10, 'thresh': 5,
     'l2Column': 1})
  expParams.append(
    {'L2cellCount': 4096, 'L2activeBits': 40, 'w': 20, 'sample': 10, 'thresh': 5,
     'l2Column': 1})
  expParams.append(
    {'L2cellCount': 6144, 'L2activeBits': 40, 'w': 20, 'sample': 10, 'thresh': 5,
     'l2Column': 1})

  for expParam in expParams:
    l2Params['sampleSizeProximal'] = expParam['sample']
    l2Params['minThresholdProximal'] = expParam['thresh']
    l2Params['cellCount'] = expParam['L2cellCount']
    l2Params['sdrSize'] = expParam['L2activeBits']

    numCorticalColumns = expParam['l2Column']

    l4Params["columnCount"] = numL4MiniColumns
    l4Params["activationThreshold"] = int(numInputBits * .6)
    l4Params["minThreshold"] = int(numInputBits * .6)
    l4Params["sampleSize"] = int(2 * l4Params["activationThreshold"])

    objectParams = {'numInputBits': numInputBits,
                    'externalInputSize': externalInputSize,
                    'numFeatures': DEFAULT_NUM_FEATURES,
                    'numLocations': DEFAULT_NUM_LOCATIONS,
                    'uniquePairs': True,}

    print "l4Params: "
    pprint(l4Params)
    print "l2Params: "
    pprint(l2Params)

    expName = "multiple_column_capacity_varying_object_num_synapses_{}_thresh_{}_l2Cells_{}_l2column_{}".format(
      expParam['sample'], expParam['thresh'], expParam["L2cellCount"],
      expParam['l2Column'])

    runCapacityTestVaryingObjectNum(numPointsPerObject,
                                    numCorticalColumns,
                                    resultDirName,
                                    expName,
                                    cpuCount,
                                    l2Params,
                                    l4Params,
                                    objectParams,
                                    "MultipleL4L2Columns",
                                    numRpts)

  # plot result
  ploti = 0
  fig, ax = plt.subplots(2, 2)
  st = fig.suptitle("Varying number of objects", fontsize="x-large")

  for axi in (0, 1):
    for axj in (0, 1):
      ax[axi][axj].xaxis.set_major_locator(ticker.MultipleLocator(100))

  legendEntries = []
  for expParam in expParams:
    expName = "multiple_column_capacity_varying_object_num_synapses_{}_thresh_{}_l2Cells_{}_l2column_{}".format(
      expParam['sample'], expParam['thresh'], expParam["L2cellCount"],
      expParam['l2Column'])

    resultFileName = _prepareResultsDir("{}.csv".format(expName),
                                        resultDirName=resultDirName
                                        )

    result = pd.read_csv(resultFileName)

    plotResults(result, ax, "numObjects", None, DEFAULT_COLORS[ploti])
    ploti += 1
    legendEntries.append("L2 cells {}/{} #cc {} ".format(
      expParam['L2activeBits'], expParam['L2cellCount'], expParam['l2Column']))
  ax[0, 0].legend(legendEntries, loc=3, fontsize=8)
  fig.tight_layout()

  # shift subplots down:
  st.set_y(0.95)
  fig.subplots_adjust(top=0.85)

  plt.savefig(
    os.path.join(
      plotDirName,
      "capacity_vs_L2size.pdf"
    )
  )



def runExperiment6(resultDirName=DEFAULT_RESULT_DIR_NAME,
                   plotDirName=DEFAULT_PLOT_DIR_NAME,
                   cpuCount=None):
  """
  varying size of L2
  calculate capacity by varying number of objects with fixed size
  """

  numPointsPerObject = 10
  numRpts = 5
  numInputBits = 10
  externalInputSize = 2400
  numL4MiniColumns = 150

  l4Params = getL4Params()
  l2Params = getL2Params()

  expParams = []
  expParams.append(
    {'L2cellCount': 4096, 'L2activeBits': 10, 'w': 10, 'sample': 6, 'thresh': 3,
     'l2Column': 1})
  expParams.append(
    {'L2cellCount': 4096, 'L2activeBits': 20, 'w': 10, 'sample': 6, 'thresh': 3,
     'l2Column': 1})
  expParams.append(
    {'L2cellCount': 4096, 'L2activeBits': 40, 'w': 10, 'sample': 6, 'thresh': 3,
     'l2Column': 1})
  expParams.append(
    {'L2cellCount': 4096, 'L2activeBits': 80, 'w': 10, 'sample': 6, 'thresh': 3,
     'l2Column': 1})

  for expParam in expParams:
    l2Params['sampleSizeProximal'] = expParam['sample']
    l2Params['minThresholdProximal'] = expParam['thresh']
    l2Params['cellCount'] = expParam['L2cellCount']
    l2Params['sdrSize'] = expParam['L2activeBits']
    l2Params['sampleSizeDistal'] = int(l2Params['sdrSize'] / 2)
    l2Params['activationThresholdDistal'] = int(l2Params['sdrSize'] / 2) - 1
    numCorticalColumns = expParam['l2Column']

    l4Params["columnCount"] = numL4MiniColumns
    l4Params["activationThreshold"] = int(numInputBits * .6)
    l4Params["minThreshold"] = int(numInputBits * .6)
    l4Params["sampleSize"] = int(2 * l4Params["activationThreshold"])

    objectParams = {'numInputBits': numInputBits,
                    'externalInputSize': externalInputSize,
                    'numFeatures': DEFAULT_NUM_FEATURES,
                    'numLocations': DEFAULT_NUM_LOCATIONS,
                    'uniquePairs': True,}

    print "l4Params: "
    pprint(l4Params)
    print "l2Params: "
    pprint(l2Params)

    expName = "multiple_column_capacity_varying_object_sdrSize_{}_l2Cells_{}_l2column_{}".format(
      expParam['L2activeBits'], expParam["L2cellCount"], expParam['l2Column'])

    runCapacityTestVaryingObjectNum(numPointsPerObject,
                                    numCorticalColumns,
                                    resultDirName,
                                    expName,
                                    cpuCount,
                                    l2Params,
                                    l4Params,
                                    objectParams,
                                    "MultipleL4L2Columns",
                                    numRpts)

  # plot result
  ploti = 0
  fig, ax = plt.subplots(2, 2)
  st = fig.suptitle("Varying number of objects", fontsize="x-large")

  for axi in (0, 1):
    for axj in (0, 1):
      ax[axi][axj].xaxis.set_major_locator(ticker.MultipleLocator(100))

  legendEntries = []
  for expParam in expParams:
    expName = "multiple_column_capacity_varying_object_sdrSize_{}_l2Cells_{}_l2column_{}".format(
      expParam['L2activeBits'], expParam["L2cellCount"], expParam['l2Column'])

    resultFileName = _prepareResultsDir("{}.csv".format(expName),
                                        resultDirName=resultDirName
                                        )

    result = pd.read_csv(resultFileName)

    plotResults(result, ax, "numObjects", None, DEFAULT_COLORS[ploti])
    ploti += 1
    legendEntries.append("L2 cells {}/{} #cc {} ".format(
      expParam['L2activeBits'], expParam['L2cellCount'], expParam['l2Column']))
  ax[0, 0].legend(legendEntries, loc=3, fontsize=8)
  fig.tight_layout()

  # shift subplots down:
  st.set_y(0.95)
  fig.subplots_adjust(top=0.85)

  plt.savefig(
    os.path.join(
      plotDirName,
      "capacity_vs_L2_sparsity.pdf"
    )
  )



def runExperiment7(numCorticalColumns=DEFAULT_NUM_CORTICAL_COLUMNS,
                   resultDirName=DEFAULT_RESULT_DIR_NAME,
                   plotDirName=DEFAULT_PLOT_DIR_NAME,
                   cpuCount=None):
  """
  runCapacityTestVaryingObjectNum()
  Try different numLocations
  """

  numPointsPerObject = 10
  numRpts = 1
  l4Params = getL4Params()
  l2Params = getL2Params()

  l2Params['cellCount'] = 4096
  l2Params['sdrSize'] = 40

  expParams = [
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'numFeatures': 500, 'numLocations': 16},
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'numFeatures': 500, 'numLocations': 128},
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'numFeatures': 500, 'numLocations': 1000},
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'numFeatures': 500, 'numLocations': 5000},
  ]

  for expParam in expParams:
    l4Params["columnCount"] = expParam['l4Column']
    numInputBits = expParam['w']
    l2Params['sampleSizeProximal'] = expParam['sample']
    l2Params['minThresholdProximal'] = expParam['thresh']

    l4Params["activationThreshold"] = int(numInputBits * .6)
    l4Params["minThreshold"] = int(numInputBits * .6)
    l4Params["sampleSize"] = int(2 * l4Params["activationThreshold"])

    objectParams = {
      'numInputBits': numInputBits,
      'externalInputSize': expParam['externalInputSize'],
      'numFeatures': expParam['numFeatures'],
      'numLocations': expParam['numLocations'],
      'uniquePairs': False
    }

    print "l4Params: "
    pprint(l4Params)
    print "l2Params: "
    pprint(l2Params)

    expname = "multiple_column_capacity_varying_object_num_locations_{}_num_features_{}_l4column_{}".format(
      expParam['numLocations'], expParam['numFeatures'], expParam["l4Column"])
    runCapacityTestVaryingObjectNum(numPointsPerObject,
                                    numCorticalColumns,
                                    resultDirName,
                                    expname,
                                    cpuCount,
                                    l2Params,
                                    l4Params,
                                    objectParams,
                                    "MultipleL4L2Columns",
                                    numRpts)

  # plot result
  ploti = 0
  fig, ax = plt.subplots(2, 2)
  st = fig.suptitle(
    "Varying number of objects ({} cortical column{})"
      .format(numCorticalColumns, "s" if numCorticalColumns > 1 else ""
              ), fontsize="x-large"
  )

  for axi in (0, 1):
    for axj in (0, 1):
      ax[axi][axj].xaxis.set_major_locator(ticker.MultipleLocator(100))

  legendEntries = []
  for expParam in expParams:
    expname = "multiple_column_capacity_varying_object_num_locations_{}_num_features_{}_l4column_{}".format(
      expParam['numLocations'], expParam['numFeatures'], expParam["l4Column"])

    resultFileName = _prepareResultsDir("{}.csv".format(expname),
                                        resultDirName=resultDirName
                                        )

    result = pd.read_csv(resultFileName)

    plotResults(result, ax, "numObjects", None, DEFAULT_COLORS[ploti])
    ploti += 1
    legendEntries.append("L4 mcs {} locs {} feats {}".format(
      expParam["l4Column"], expParam['numLocations'], expParam['numFeatures']))
  ax[0, 0].legend(legendEntries, loc=3, fontsize=8)
  fig.tight_layout()

  # shift subplots down:
  st.set_y(0.95)
  fig.subplots_adjust(top=0.85)

  plt.savefig(
    os.path.join(
      plotDirName,
      "capacity_varying_object_num_locations_num_summary.pdf"
    )
  )



def runExperiment8(numCorticalColumns=DEFAULT_NUM_CORTICAL_COLUMNS,
                   resultDirName=DEFAULT_RESULT_DIR_NAME,
                   plotDirName=DEFAULT_PLOT_DIR_NAME,
                   cpuCount=None):
  """
  runCapacityTestVaryingObjectNum()
  Try different numFeatures
  """

  numPointsPerObject = 10
  numRpts = 1
  l4Params = getL4Params()
  l2Params = getL2Params()

  l2Params['cellCount'] = 4096
  l2Params['sdrSize'] = 40

  expParams = [
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'numFeatures': 15, 'numLocations': 128},
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'numFeatures': 150, 'numLocations': 128},
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'numFeatures': 500, 'numLocations': 128},
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'numFeatures': 5000, 'numLocations': 128},
  ]

  for expParam in expParams:
    l4Params["columnCount"] = expParam['l4Column']
    numInputBits = expParam['w']
    l2Params['sampleSizeProximal'] = expParam['sample']
    l2Params['minThresholdProximal'] = expParam['thresh']

    l4Params["activationThreshold"] = int(numInputBits * .6)
    l4Params["minThreshold"] = int(numInputBits * .6)
    l4Params["sampleSize"] = int(2 * l4Params["activationThreshold"])

    objectParams = {
      'numInputBits': numInputBits,
      'externalInputSize': expParam['externalInputSize'],
      'numFeatures': expParam['numFeatures'],
      'numLocations': expParam['numLocations'],
      'uniquePairs': False
    }

    print "l4Params: "
    pprint(l4Params)
    print "l2Params: "
    pprint(l2Params)

    expname = "multiple_column_capacity_varying_object_num_locations_{}_num_features_{}_l4column_{}".format(
      expParam['numLocations'], expParam['numFeatures'], expParam["l4Column"])
    runCapacityTestVaryingObjectNum(numPointsPerObject,
                                    numCorticalColumns,
                                    resultDirName,
                                    expname,
                                    cpuCount,
                                    l2Params,
                                    l4Params,
                                    objectParams,
                                    "MultipleL4L2Columns",
                                    numRpts)

  # plot result
  ploti = 0
  fig, ax = plt.subplots(2, 2)
  st = fig.suptitle(
    "Varying number of objects ({} cortical column{})"
      .format(numCorticalColumns, "s" if numCorticalColumns > 1 else ""
              ), fontsize="x-large"
  )

  for axi in (0, 1):
    for axj in (0, 1):
      ax[axi][axj].xaxis.set_major_locator(ticker.MultipleLocator(100))

  legendEntries = []
  for expParam in expParams:
    expname = "multiple_column_capacity_varying_object_num_locations_{}_num_features_{}_l4column_{}".format(
      expParam['numLocations'], expParam['numFeatures'], expParam["l4Column"])

    resultFileName = _prepareResultsDir("{}.csv".format(expname),
                                        resultDirName=resultDirName
                                        )

    result = pd.read_csv(resultFileName)

    plotResults(result, ax, "numObjects", None, DEFAULT_COLORS[ploti])
    ploti += 1
    legendEntries.append("L4 mcs {} locs {} feats {}".format(
      expParam["l4Column"], expParam['numLocations'], expParam['numFeatures']))
  ax[0, 0].legend(legendEntries, loc=3, fontsize=8)
  fig.tight_layout()

  # shift subplots down:
  st.set_y(0.95)
  fig.subplots_adjust(top=0.85)

  plt.savefig(
    os.path.join(
      plotDirName,
      "capacity_varying_object_num_features_num_summary.pdf"
    )
  )



def runExperiment9(resultDirName=DEFAULT_RESULT_DIR_NAME,
                   plotDirName=DEFAULT_PLOT_DIR_NAME,
                   cpuCount=None):
  """
  runCapacityTestVaryingObjectNum()
  varying number of cortical columns, 2d topology.
  """

  numPointsPerObject = 10
  numRpts = 3
  objectNumRange = range(10, 1000, 50)

  l4Params = getL4Params()
  l2Params = getL2Params()

  expParams = []
  expParams.append(
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'l2Column': 4, 'networkType': "MultipleL4L2Columns"})
  expParams.append(
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'l2Column': 9, 'networkType': "MultipleL4L2Columns"})
  expParams.append(
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'l2Column': 16, 'networkType': "MultipleL4L2Columns"})
  expParams.append(
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'l2Column': 4, 'networkType': "MultipleL4L2ColumnsWithTopology"})
  expParams.append(
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'l2Column': 9, 'networkType': "MultipleL4L2ColumnsWithTopology"})
  expParams.append(
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 20, 'sample': 6,
     'thresh': 3, 'l2Column': 16, 'networkType': "MultipleL4L2ColumnsWithTopology"})

  run_params = []
  for object_num in reversed(objectNumRange):
    for expParam in expParams:
      for rpt in range(numRpts):
        l2Params['sampleSizeProximal'] = expParam['sample']
        l2Params['minThresholdProximal'] = expParam['thresh']

        l4Params["columnCount"] = expParam['l4Column']
        numInputBits = expParam['w']
        numCorticalColumns = expParam['l2Column']
        networkType = expParam['networkType']

        l4Params["activationThreshold"] = int(numInputBits * .6)
        l4Params["minThreshold"] = int(numInputBits * .6)
        l4Params["sampleSize"] = int(2 * l4Params["activationThreshold"])

        objectParams = {'numInputBits': numInputBits,
                        'externalInputSize': expParam['externalInputSize'],
                        'numFeatures': DEFAULT_NUM_FEATURES,
                        'numLocations': DEFAULT_NUM_LOCATIONS,
                        'uniquePairs': True,}

        print "Experiment Params: "
        pprint(expParam)

        expName = "multiple_column_capacity_varying_object_num_synapses_{}_thresh_{}_l4column_{}_l2column_{}_{}".format(
            expParam['sample'], expParam['thresh'], expParam["l4Column"],
            expParam['l2Column'], expParam["networkType"])

        try:
          os.remove(_prepareResultsDir("{}.csv".format(expName),
            resultDirName=resultDirName))
        except OSError:
          pass

        run_params.append((numPointsPerObject,
                           numCorticalColumns,
                           resultDirName,
                           object_num,
                           expName,
                           l2Params,
                           l4Params,
                           objectParams,
                           networkType,
                           rpt))

  pool = multiprocessing.Pool(cpuCount or multiprocessing.cpu_count(), maxtasksperchild=1)
  pool.map(invokeRunCapacityTestWrapper, run_params, chunksize = 1)
  # plot result
  ploti = 0
  fig, ax = plt.subplots(2, 2)
  st = fig.suptitle("Varying number of objects", fontsize="x-large")

  for axi in (0, 1):
    for axj in (0, 1):
      ax[axi][axj].xaxis.set_major_locator(ticker.MultipleLocator(100))

  colormap = plt.get_cmap("jet")
  colors = [colormap(x) for x in np.linspace(0., 1., len(expParam))]

  legendEntries = []
  for expParam in expParams:
    expName = "multiple_column_capacity_varying_object_num_synapses_{}_thresh_{}_l4column_{}_l2column_{}_{}".format(
        expParam['sample'], expParam['thresh'], expParam["l4Column"],
        expParam['l2Column'], expParam["networkType"])

    resultFileName = _prepareResultsDir("{}.csv".format(expName),
                                        resultDirName=resultDirName
                                        )

    result = pd.read_csv(resultFileName)

    plotResults(result, ax, "numObjects", None, colors[ploti])
    ploti += 1
    if "Topology" in expParam["networkType"]:
      legendEntries.append("L4 mcs {} #cc {} w/ topology".format(
        expParam['l4Column'], expParam['l2Column']))
    else:
      legendEntries.append("L4 mcs {} #cc {}".format(
        expParam['l4Column'], expParam['l2Column']))
  ax[0, 0].legend(legendEntries, loc=3, fontsize=8)
  fig.tight_layout()

  # shift subplots down:
  st.set_y(0.95)
  fig.subplots_adjust(top=0.85)

  plt.savefig(
    os.path.join(
      plotDirName,
      "multiple_column_capacity_varying_object_num_column_num_connection_type_summary.pdf"
    )
  )



def runExperiment10(numCorticalColumns=DEFAULT_NUM_CORTICAL_COLUMNS,
                   resultDirName=DEFAULT_RESULT_DIR_NAME,
                   plotDirName=DEFAULT_PLOT_DIR_NAME,
                   cpuCount=1):
  """
  Try different L4 network size
  """

  numPointsPerObject = 10
  numRpts = 1
  l4Params = getL4Params()
  l2Params = getL2Params()

  l2Params['cellCount'] = 4096
  l2Params['sdrSize'] = 40

  expParams = []
  expParams.append(
    {'l4Column': 100, 'externalInputSize': 2400, 'w': 10, 'sample': 5,
     'L2cellCount': 2000, 'L2activeBits': 20, 'thresh': 4})
  expParams.append(
    {'l4Column': 150, 'externalInputSize': 2400, 'w': 15, 'sample': 8,
     'L2cellCount': 3000, 'L2activeBits': 30, 'thresh': 6})
  expParams.append(
    {'l4Column': 200, 'externalInputSize': 2400, 'w': 20, 'sample': 10,
     'L2cellCount': 4000, 'L2activeBits': 40, 'thresh': 8})
  expParams.append(
    {'l4Column': 250, 'externalInputSize': 2400, 'w': 25, 'sample': 13,
     'L2cellCount': 5000, 'L2activeBits': 50, 'thresh': 10})

  for expParam in expParams:
    l4Params["columnCount"] = expParam['l4Column']
    numInputBits = expParam['w']

    l4Params["activationThreshold"] = int(numInputBits * .6)
    l4Params["minThreshold"] = int(numInputBits * .6)
    l4Params["sampleSize"] = int(2 * l4Params["activationThreshold"])

    l2Params['sampleSizeProximal'] = expParam['sample']
    l2Params['minThresholdProximal'] = expParam['thresh']
    l2Params['cellCount'] = expParam['L2cellCount']
    l2Params['sdrSize'] = expParam['L2activeBits']
    l2Params['sampleSizeDistal'] = int(expParam['L2cellCount']*.5)
    l2Params['activationThresholdDistal'] = int(expParam['L2cellCount'] * .5)-1

    objectParams = {'numInputBits': numInputBits,
                    'externalInputSize': expParam['externalInputSize'],
                    'numFeatures': DEFAULT_NUM_FEATURES,
                    'numLocations': DEFAULT_NUM_LOCATIONS,
                    'uniquePairs': True,}

    print "l4Params: "
    pprint(l4Params)
    print "l2Params: "
    pprint(l2Params)

    expname = "multiple_column_capacity_varying_object_num_synapses_{}_thresh_{}_l4column_{}_l2cell_{}".format(
      expParam['sample'], expParam['thresh'], expParam["l4Column"], expParam['L2cellCount'])
    runCapacityTestVaryingObjectNum(numPointsPerObject,
                                    numCorticalColumns,
                                    resultDirName,
                                    expname,
                                    cpuCount,
                                    l2Params,
                                    l4Params,
                                    objectParams,
                                    "MultipleL4L2Columns",
                                    numRpts)

  # plot result
  ploti = 0
  fig, ax = plt.subplots(2, 2)
  st = fig.suptitle(
    "Varying number of objects ({} cortical column{})"
      .format(numCorticalColumns, "s" if numCorticalColumns > 1 else ""
              ), fontsize="x-large"
  )

  for axi in (0, 1):
    for axj in (0, 1):
      ax[axi][axj].xaxis.set_major_locator(ticker.MultipleLocator(100))

  legendEntries = []
  for expParam in expParams:
    expname = "multiple_column_capacity_varying_object_num_synapses_{}_thresh_{}_l4column_{}_l2cell_{}".format(
      expParam['sample'], expParam['thresh'], expParam["l4Column"], expParam['L2cellCount'])

    resultFileName = _prepareResultsDir("{}.csv".format(expname),
                                        resultDirName=resultDirName
                                        )

    result = pd.read_csv(resultFileName)

    plotResults(result, ax, "numObjects", None, DEFAULT_COLORS[ploti])
    ploti += 1
    legendEntries.append("L4 mcs {} w {} s {} thresh {}".format(
      expParam["l4Column"], expParam['w'], expParam['sample'],
      expParam['thresh']))
  ax[0, 0].legend(legendEntries, loc=4, fontsize=8)
  fig.tight_layout()

  # shift subplots down:
  st.set_y(0.95)
  fig.subplots_adjust(top=0.85)

  plt.savefig(
    os.path.join(
      plotDirName,
      "capacity_varying_object_num_l4l2size_summary.pdf"
    )
  )


def runExperiments(resultDirName, plotDirName, cpuCount):
#  # Varying number of pts per objects, two objects
#  runExperiment1(numCorticalColumns=1,
#                 resultDirName=resultDirName,
#                 plotDirName=plotDirName,
#                 cpuCount=cpuCount)
#
#  # 10 pts per object, varying number of objects
#  runExperiment2(numCorticalColumns=1,
#                 resultDirName=resultDirName,
#                 plotDirName=plotDirName,
#                 cpuCount=cpuCount)
#
 # 10 pts per object, varying number of objects, varying L4 size
 # runExperiment3(numCorticalColumns=1,
 #                resultDirName=resultDirName,
 #                plotDirName=plotDirName,
 #                cpuCount=cpuCount)
#
 # 10 pts per object, varying number of objects and number of columns
 runExperiment4(resultDirName=resultDirName,
                plotDirName=plotDirName,
                cpuCount=cpuCount)

#  # 10 pts per object, varying number of L2 cells
#  runExperiment5(resultDirName=resultDirName,
#                 plotDirName=plotDirName,
#                 cpuCount=cpuCount)
#
#  # 10 pts per object, varying sparsity of L2
#  runExperiment6(resultDirName=resultDirName,
#                 plotDirName=plotDirName,
#                 cpuCount=cpuCount)
#
#  # 10 pts per object, varying number of location SDRs
#  runExperiment7(numCorticalColumns=1,
#                 resultDirName=resultDirName,
#                 plotDirName=plotDirName,
#                 cpuCount=cpuCount)
#
#  # 10 pts per object, varying number of feature SDRs
#  runExperiment8(numCorticalColumns=1,
#                 resultDirName=resultDirName,
#                 plotDirName=plotDirName,
#                 cpuCount=cpuCount)
#   # #10 pts per object, varying number of objects and number of columns
#   runExperiment9(resultDirName=resultDirName,
#                  plotDirName=plotDirName,
#                  cpuCount=cpuCount)


  # 10 pts per object, varying number of objects, varying L4/L2 size
  # runExperiment10(numCorticalColumns=1,
  #                resultDirName=resultDirName,
  #                plotDirName=plotDirName,
  #                cpuCount=cpuCount)



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--resultDirName",
    default=DEFAULT_RESULT_DIR_NAME,
    type=str,
    metavar="DIRECTORY"
  )
  parser.add_argument(
    "--plotDirName",
    default=DEFAULT_PLOT_DIR_NAME,
    type=str,
    metavar="DIRECTORY"
  )
  parser.add_argument(
    "--cpuCount",
    default=None,
    type=int,
    metavar="NUM",
    help="Limit number of cpu cores.  Defaults to `multiprocessing.cpu_count()`"
  )

  opts = parser.parse_args()

  # runExperiments(resultDirName=opts.resultDirName,
  #                plotDirName=opts.plotDirName,
  #                cpuCount=opts.cpuCount)



  numObjects = 50
  numPointsPerObject=10
  numCorticalColumns=1
  l4Params = getL4Params()
  l2Params = getL2Params()
  objectParams = {'numInputBits': 20,
                  'externalInputSize': 2400,
                  'numFeatures': DEFAULT_NUM_FEATURES,
                  'numLocations': DEFAULT_NUM_LOCATIONS,
                  'uniquePairs': True}
  runCapacityTest(numObjects,
                      numPointsPerObject,
                      numCorticalColumns,
                      l2Params,
                      l4Params,
                      objectParams,
                      networkType = "MultipleL4L2Columns",
                      repeat=0)