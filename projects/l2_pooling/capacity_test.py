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

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from htmresearch.frameworks.layers.object_machine_factory import (
  createObjectMachine
)
from htmresearch.frameworks.layers.l2_l4_inference import L4L2Experiment



NUM_LOCATIONS = 5000
NUM_FEATURES = 5000
DEFAULT_RESULT_DIR_NAME = "results"
DEFAULT_PLOT_DIR_NAME = "plots"
DEFAULT_NUM_CORTICAL_COLUMNS = 1



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
    "columnCount": 2048,
    "cellsPerColumn": 8,
    "formInternalBasalConnections": True,
    "learningMode": True,
    "inferenceMode": True,
    "learnOnOneCell": False,
    "initialPermanence": 0.51,
    "connectedPermanence": 0.6,
    "permanenceIncrement": 0.1,
    "permanenceDecrement": 0.02,
    "minThreshold": 10,
    "predictedSegmentDecrement": 0.002,
    "activationThreshold": 13,
    "maxNewSynapseCount": 20,
    "implementation": "etm_cpp",
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
    "minThresholdProximal": 1,
    "sampleSizeProximal": 5,
    "connectedPermanenceProximal": 0.5,
    "synPermDistalInc": 0.1,
    "synPermDistalDec": 0.001,
    "initialDistalPermanence": 0.41,
    "activationThresholdDistal": 13,
    "sampleSizeDistal": 20,
    "connectedPermanenceDistal": 0.5,
    "distalSegmentInhibitionFactor": 1.5,
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

  for step, pair in enumerate(testPairs):
    (locationIdx, featureIdx) = pair
    for colIdx in xrange(exp.numColumns):
      feature = objects.features[colIdx][featureIdx]
      location = objects.locations[colIdx][locationIdx]

      exp.sensorInputs[colIdx].addDataToQueue(list(feature), 0, 0)
      exp.externalInputs[colIdx].addDataToQueue(list(location), 0, 0)

    exp.network.run(1)

    # columnPooler = exp.L2Columns[0]._pooler
    # tm = exp.L4Columns[0]._tm
    # print "step : {}".format(step)
    # print "predicted active cells: ", tm.getPredictedActiveCells()
    # print "L2 activation: ", columnPooler.getActiveCells()

    for obj in xrange(numObjects):
      overlap[step, obj] = np.mean([
        len(exp.objectL2Representations[obj][colIdx] &
            exp.getL2Representations()[colIdx])
        for colIdx in xrange(exp.numColumns)]
      )

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

  innerObjs = objects.getObjects()

  numObjects = len(innerObjs)
  numPointsPerObject = len(innerObjs[0])
  overlapTrueObj = np.zeros((numRepeats, ))
  confusion = overlapTrueObj.copy()
  outcome = overlapTrueObj.copy()

  for i in xrange(numRepeats):
    targetObject = np.random.choice(np.arange(numObjects))

    nonTargetObjs = np.array(
      [obj for obj in xrange(numObjects) if obj != targetObject]
    )

    overlap = testNetworkWithOneObject(
      objects,
      exp,
      targetObject,
      3
    )

    lastOverlap = overlap[-1, :]

    maxOverlapIndices = (
      np.where(lastOverlap == lastOverlap[np.argmax(lastOverlap)])[0].tolist()
    )

    # Only set to 1 iff target object is the lone max overlap index.  Otherwise
    # the network failed to conclusively identify the target object.
    outcome[i] = 1 if maxOverlapIndices == [targetObject] else 0

    confusion[i] = np.max(overlap[0, nonTargetObjs])
    overlapTrueObj[i] = overlap[0, targetObject]

  columnPooler = exp.L2Columns[0]._pooler
  numConnectedProximal = columnPooler.numberOfConnectedProximalSynapses()
  numConnectedDistal = columnPooler.numberOfConnectedDistalSynapses()

  return {"numberOfConnectedProximalSynapses": numConnectedProximal,
          "numberOfConnectedDistalSynapses": numConnectedDistal,
          "numObjects": numObjects,
          "numPointsPerObject": numPointsPerObject,
          "confusion": np.mean(confusion),
          "accuracy": np.mean(outcome),
          "overlapTrueObj": np.mean(overlapTrueObj)}



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

  ax[0, 1].plot(x, result.numberOfConnectedProximalSynapses, marker)
  ax[0, 1].set_ylabel("# connected proximal synapses")
  ax[0, 1].set_xlabel("# Pts / Obj")
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
                    sampleSize,
                    activationThreshold,
                    numCorticalColumns):
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
  l4Params = getL4Params()
  l2Params = getL2Params()
  l2Params["sampleSizeProximal"] = sampleSize # TODO
  l2Params["minThresholdProximal"] = activationThreshold

  l4ColumnCount = l4Params["columnCount"]
  numInputBits = int(l4Params["columnCount"]*0.02)

  objects = createObjectMachine(
    machineType="simple",
    numInputBits=numInputBits,
    sensorInputSize=l4ColumnCount,
    externalInputSize=l4ColumnCount,
    numCorticalColumns=numCorticalColumns,
    numLocations=NUM_LOCATIONS,
    numFeatures=NUM_FEATURES
  )

  exp = L4L2Experiment("capacity_two_objects",
                       numInputBits=numInputBits,
                       L2Overrides=l2Params,
                       L4Overrides=l4Params,
                       inputSize=l4ColumnCount,
                       externalInputSize=l4ColumnCount,
                       numLearningPoints=4,
                       numCorticalColumns=numCorticalColumns)

  pairs = createRandomObjects(
    numObjects,
    numPointsPerObject,
    NUM_LOCATIONS,
    NUM_FEATURES
  )

  for object in pairs:
    objects.addObject(object)

  exp.learnObjects(objects.provideObjectsToLearn())

  testResult = testOnSingleRandomSDR(objects, exp)

  return testResult



def runCapacityTestVaryingObjectSize(
    numObjects=2,
    sampleSize=5,
    activationThreshold=3,
    numCorticalColumns=DEFAULT_NUM_CORTICAL_COLUMNS,
    resultDirName=DEFAULT_RESULT_DIR_NAME,
    cpuCount=None):
  """
  Runs experiment with two objects, varying number of points per object
  """

  result = None

  cpuCount = cpuCount or multiprocessing.cpu_count()
  pool = multiprocessing.Pool(cpuCount, maxtasksperchild=1)

  params = [(numObjects,
             numPointsPerObject,
             sampleSize,
             activationThreshold,
             numCorticalColumns)
            for numPointsPerObject in np.arange(10, 270, 20)]

  for testResult in pool.map(invokeRunCapacityTest, params):
    print testResult

    result = (
      pd.concat([result, pd.DataFrame.from_dict([testResult])])
      if result is not None else
      pd.DataFrame.from_dict([testResult])
    )

  resultFileName = _prepareResultsDir(
    "multiple_column_capacity_varying_object_size_synapses_{}_thresh_{}.csv"
    .format(sampleSize, activationThreshold),
    resultDirName=resultDirName
  )

  pd.DataFrame.to_csv(result, resultFileName)



def invokeRunCapacityTest(params):
  """ Splits out params so that runCapacityTest may be invoked with
  multiprocessing.Pool.map() to support parallelism
  """
  return runCapacityTest(*params)



def runCapacityTestVaryingObjectNum(numPointsPerObject=10,
                                    sampleSize=5,
                                    activationThreshold=3,
                                    numCorticalColumns=DEFAULT_NUM_CORTICAL_COLUMNS,
                                    resultDirName=DEFAULT_RESULT_DIR_NAME,
                                    cpuCount=None):
  """
  Run experiment with fixed number of pts per object, varying number of objects
  """
  result = None

  cpuCount = cpuCount or multiprocessing.cpu_count()
  pool = multiprocessing.Pool(cpuCount, maxtasksperchild=1)

  params = [(numObjects,
             numPointsPerObject,
             sampleSize,
             activationThreshold,
             numCorticalColumns)
            for numObjects in np.arange(20, 1371, 150)]

  for testResult in pool.map(invokeRunCapacityTest, params):
    print testResult

    result = (
      pd.concat([result, pd.DataFrame.from_dict([testResult])])
      if result is not None else
      pd.DataFrame.from_dict([testResult])
    )

  resultFileName = _prepareResultsDir(
    "multiple_column_capacity_varying_object_num_synapses_{}_thresh_{}.csv"
      .format(sampleSize, activationThreshold),
    resultDirName=resultDirName
  )

  pd.DataFrame.to_csv(result, resultFileName)



def runExperiment1(numObjects=2,
                   sampleSizeRange=(10,),
                   numCorticalColumns=DEFAULT_NUM_CORTICAL_COLUMNS,
                   resultDirName=DEFAULT_RESULT_DIR_NAME,
                   plotDirName=DEFAULT_PLOT_DIR_NAME,
                   cpuCount=None):
  """
  Varying number of pts per objects, two objects
  Try different sampling and activation threshold
  """
  for sampleSize in sampleSizeRange:
    activationThreshold = int(sampleSize) - 1

    print "sampleSize: {}".format(sampleSize)
    print "activationThreshold: {}".format(activationThreshold)

    runCapacityTestVaryingObjectSize(numObjects,
                                     sampleSize,
                                     activationThreshold,
                                     numCorticalColumns,
                                     resultDirName,
                                     cpuCount)

  markers = ("-bo", "-ro", "-co", "-go")
  ploti = 0
  fig, ax = plt.subplots(2, 2)
  st = fig.suptitle(
    "Varying points per object x 2 objects ({} cortical column{})"
    .format(numCorticalColumns, "s" if numCorticalColumns > 1 else ""
  ), fontsize="x-large")

  legendEntries = []

  for sampleSize in sampleSizeRange:
    activationThreshold = int(sampleSize) - 1

    resultFileName = _prepareResultsDir(
      "multiple_column_capacity_varying_object_size_synapses_{}_thresh_{}.csv"
      .format(sampleSize, activationThreshold),
      resultDirName=resultDirName
    )

    result = pd.read_csv(resultFileName)

    plotResults(result, ax, "numPointsPerObject", None, markers[ploti])
    ploti += 1
    legendEntries.append("# syn {}".format(sampleSize))

  plt.legend(legendEntries, loc=2)
  fig.tight_layout()

  # shift subplots down:
  st.set_y(0.95)
  fig.subplots_adjust(top=0.85)

  plt.savefig(
    os.path.join(
      plotDirName,
      "multiple_column_capacity_varying_object_size_summary.pdf"
    )
  )



def runExperiment2(numCorticalColumns=DEFAULT_NUM_CORTICAL_COLUMNS,
                   resultDirName=DEFAULT_RESULT_DIR_NAME,
                   plotDirName=DEFAULT_PLOT_DIR_NAME,
                   cpuCount=None):
  """
  runCapacityTestVaryingObjectNum()
  Try different sampling and activation threshold
  """

  sampleSizeRange = (10, ) # Expand this tuple to include other ranges
                                   # Settled at 10 since we did not observe
                                   # much variance between 5, 10, 15, and 20
  numPointsPerObject = 10

  for sampleSize in sampleSizeRange:
    runCapacityTestVaryingObjectNum(numPointsPerObject,
                                    sampleSize,
                                    int(sampleSize) - 1,
                                    numCorticalColumns,
                                    resultDirName,
                                    cpuCount)

  markers = ("-bo", "-ro", "-co", "-go")
  ploti = 0
  fig, ax = plt.subplots(2, 2)
  st = fig.suptitle(
    "Varying number of objects ({} cortical column{})"
      .format(numCorticalColumns, "s" if numCorticalColumns > 1 else ""
              ), fontsize="x-large"
  )

  for axi in (0, 1):
    for axj in (0, 1):
      ax[axi][axj].xaxis.set_major_locator(ticker.MultipleLocator(300))

  legendEntries = []
  for sampleSize in sampleSizeRange:
    activationThreshold = int(sampleSize) - 1

    resultFileName = _prepareResultsDir(
      "multiple_column_capacity_varying_object_num_synapses_{}_thresh_{}.csv"
      .format(sampleSize, activationThreshold),
      resultDirName=resultDirName
    )

    result = pd.read_csv(resultFileName)

    plotResults(result, ax, "numObjects", None, markers[ploti])
    ploti += 1
    legendEntries.append("# syn {}".format(sampleSize))
  plt.legend(legendEntries, loc=2)
  fig.tight_layout()

  # shift subplots down:
  st.set_y(0.95)
  fig.subplots_adjust(top=0.85)

  plt.savefig(
    os.path.join(
      plotDirName,
      "multiple_column_capacity_varying_object_num_summary.pdf"
    )
  )



def runExperiments(numCorticalColumns, resultDirName, plotDirName, cpuCount):

  # Varying number of pts per objects, two objects
  runExperiment1(numCorticalColumns=numCorticalColumns,
                 resultDirName=resultDirName,
                 plotDirName=plotDirName,
                 cpuCount=cpuCount)

  # 10 pts per object, varying number of objects
  runExperiment2(numCorticalColumns=numCorticalColumns,
                 resultDirName=resultDirName,
                 plotDirName=plotDirName,
                 cpuCount=cpuCount)



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--numCorticalColumns",
    default=DEFAULT_NUM_CORTICAL_COLUMNS,
    type=int,
    metavar="NUMBER"
  )
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

  runExperiments(numCorticalColumns=opts.numCorticalColumns,
                 resultDirName=opts.resultDirName,
                 plotDirName=opts.plotDirName,
                 cpuCount=opts.cpuCount)
