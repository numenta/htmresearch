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
This file is used to run Thing experiments using simulated sensations.
"""
import random
import os
import numpy as np
import pprint
import matplotlib.pyplot as plt
from sklearn import manifold, random_projection

from htmresearch.frameworks.layers.l2_l4_inference import (
  L4L2Experiment, rerunExperimentFromLogfile)

from htmresearch.frameworks.layers.object_machine_factory import (
  createObjectMachine
)


def getL4Params():
  """
  Returns a good default set of parameters to use in the L4 region.
  """
  return {
    "columnCount": 256,
    "cellsPerColumn": 16,
    "learn": True,
    "learnOnOneCell": False,
    "initialPermanence": 0.51,
    "connectedPermanence": 0.6,
    "permanenceIncrement": 0.1,
    "permanenceDecrement": 0.02,
    "minThreshold": 18,
    "predictedSegmentDecrement": 0.0,
    "activationThreshold": 18,
    "sampleSize": 30,
    "implementation": "etm",
  }



def getL2Params():
  """
  Returns a good default set of parameters to use in the L4 region.
  """
  return {
    "inputWidth": 2048 * 16,
    "cellCount": 4096,
    "sdrSize": 40,
    "synPermProximalInc": 0.5,
    "synPermProximalDec": 0.0,
    "initialProximalPermanence": 0.6,
    "minThresholdProximal": 6,
    "sampleSizeProximal": 10,
    "connectedPermanenceProximal": 0.5,
    "synPermDistalInc": 0.1,
    "synPermDistalDec": 0.001,
    "initialDistalPermanence": 0.41,
    "activationThresholdDistal": 13,
    "sampleSizeDistal": 30,
    "connectedPermanenceDistal": 0.5,
    "distalSegmentInhibitionFactor": 1.001,
    "learningMode": True,
  }


def loadThingObjects(numCorticalColumns=1, objDataPath='./data/'):
  """
  Load simulated sensation data on a number of different objects
  There is one file per object, each row contains one feature, location pairs
  The format is as follows
  [(-33.6705, 75.5003, 2.4207)/10] => [[list of active bits of location],
                                       [list of active bits of feature]]
  The content before "=>" is the true 3D location / sensation
  The number of active bits in the location and feature is listed after "=>".
  @return A simple object machine
  """
  # create empty simple object machine
  objects = createObjectMachine(
    machineType="simple",
    numInputBits=20,
    sensorInputSize=1024,
    externalInputSize=1024,
    numCorticalColumns=numCorticalColumns,
    numFeatures=0,
    numLocations=0,
  )

  for _ in range(numCorticalColumns):
    objects.locations.append([])
    objects.features.append([])

  objFiles = []
  for f in os.listdir(objDataPath):
    if os.path.isfile(os.path.join(objDataPath, f)):
      if '.log' in f:
        objFiles.append(f)

  idx = 0
  for f in objFiles:
    objName = f.split('.')[0]
    objName = objName[4:]
    objFile = open('{}/{}'.format(objDataPath, f))

    sensationList = []
    for line in objFile.readlines():
      # parse thing data file and extract feature/location vectors
      sense = line.split('=>')[1].strip(' ').strip('\n')
      location = sense.split('],[')[0].strip('[')
      feature = sense.split('],[')[1].strip(']')
      location = np.fromstring(location, sep=',', dtype=np.uint8)
      feature = np.fromstring(feature, sep=',', dtype=np.uint8)

      # add the current sensation to object Machine
      sensationList.append((idx, idx))
      for c in range(numCorticalColumns):
        objects.locations[c].append(set(location.tolist()))
        objects.features[c].append(set(feature.tolist()))
      idx += 1
    objects.addObject(sensationList, objName)
    print "load object file: {} object name: {} sensation # {}".format(
      f, objName, len(sensationList))
  return objects



def trainNetwork(objects, numColumns, l4Params, l2Params, verbose=False):
  objectNames = objects.objects.keys()
  numObjects = len(objectNames)

  print " Training sensorimotor network ..."
  exp = L4L2Experiment("shared_features",
                       L2Overrides=l2Params,
                       L4Overrides=l4Params,
                       numCorticalColumns=numColumns)
  exp.learnObjects(objects.provideObjectsToLearn())

  settlingTime = 3
  L2Representations = exp.objectL2Representations
  if verbose:
    print "Learned object representations:"
    pprint.pprint(L2Representations, width=400)
    print "=========================="

  # For inference, we will check and plot convergence for each object. For each
  # object, we create a sequence of random sensations for each column.  We will
  # present each sensation for settlingTime time steps to let it settle and
  # ensure it converges.

  overlapMat = np.zeros((numObjects, numObjects))
  numL2ActiveCells= np.zeros((numObjects, ))
  for objectIdx in range(numObjects):
    objectId = objectNames[objectIdx]
    obj = objects[objectId]

    objectSensations = {}
    for c in range(numColumns):
      objectSensations[c] = []

    if numColumns > 1:
      # Create sequence of random sensations for this object for all columns At
      # any point in time, ensure each column touches a unique loc,feature pair
      # on the object.  It is ok for a given column to sense a loc,feature pair
      # more than once. The total number of sensations is equal to the number of
      # points on the object.
      for sensationNumber in range(len(obj)):
        # Randomly shuffle points for each sensation
        objectCopy = [pair for pair in obj]
        random.shuffle(objectCopy)
        for c in range(numColumns):
          # stay multiple steps on each sensation
          for _ in xrange(settlingTime):
            objectSensations[c].append(objectCopy[c])

    else:
      # Create sequence of sensations for this object for one column. The total
      # number of sensations is equal to the number of points on the object. No
      # point should be visited more than once.
      objectCopy = [pair for pair in obj]
      # random.shuffle(objectCopy)
      for pair in objectCopy:
        # stay multiple steps on each sensation
        for _ in xrange(settlingTime):
          objectSensations[0].append(pair)

    inferConfig = {
      "object": objectId,
      "numSteps": len(objectSensations[0]),
      "pairs": objectSensations,
      "includeRandomLocation": False,
    }

    inferenceSDRs = objects.provideObjectToInfer(inferConfig)
    exp.infer(inferenceSDRs, objectName=objectId, reset=False)

    for c in range(numColumns):
      numL2ActiveCells[objectIdx] += len(exp.getL2Representations()[c])

    if verbose:
      print "Output for {}: {}".format(objectId, exp.getL2Representations())
      print "# L2 active cells {}: ".format(numL2ActiveCells[objectIdx])


    for i in range(numObjects):
      overlapMat[objectIdx, i] = len(exp.getL2Representations()[0] &
            L2Representations[objects.objects.keys()[i]][0])
      if verbose:
        print "Intersection with {}:{}".format(
          objectNames[i], overlapMat[objectIdx, i])

    exp.sendReset()

  expResult = {'overlapMat': overlapMat,
               'numL2ActiveCells': numL2ActiveCells}
  return expResult



def computeAccuracy(expResult, objects):
  objectNames = objects.objects.keys()
  overlapMat = expResult['overlapMat']
  numL2ActiveCells = expResult['numL2ActiveCells']
  numCorrect = 0
  numObjects = overlapMat.shape[0]
  numFound = 0

  percentOverlap = np.zeros(overlapMat.shape)
  for i in range(numObjects):
    for j in range(i, numObjects):
      percentOverlap[i, j] = overlapMat[i, j] # / np.min([numL2ActiveCells[i], numL2ActiveCells[j]])

  objectNames = np.array(objectNames)
  for i in range(numObjects):
    # idx = np.where(overlapMat[i, :]>confuseThresh)[0]
    idx = np.where(percentOverlap[i, :] == np.max(percentOverlap[i, :]))[0]
    print " {}, # sensations {}, best match is {}".format(
      objectNames[i], len(objects[objectNames[i]]), objectNames[idx])

    found = len(np.where(idx == i)[0]) > 0
    numFound += found
    if not found:
      print "<=========== {} was not detected ! ===========>".format(objectNames[i])
    if len(idx) > 1:
      continue

    if idx[0] == i:
      numCorrect += 1

  accuracy = float(numCorrect)/numObjects
  numPerfect = len(np.where(numL2ActiveCells<=40)[0])
  print "accuracy: {} ({}/{}) ".format(accuracy, numCorrect, numObjects)
  print "perfect retrival ratio: {} ({}/{}) ".format(
    float(numPerfect)/numObjects, numPerfect, numObjects)
  print "Object detection ratio {}/{} ".format(numFound, numObjects)
  return accuracy


def runExperimentAccuracyVsL4Thresh():
  accuracyVsThresh = []
  threshList = np.arange(13, 20)
  for thresh in threshList:
    numColumns = 1
    l2Params = getL2Params()
    l4Params = getL4Params()

    l4Params['minThreshold'] = thresh
    l4Params['activationThreshold'] = thresh

    objects = loadThingObjects(1, './data')
    expResult = trainNetwork(objects, numColumns, l4Params, l2Params, True)

    accuracy = computeAccuracy(expResult, objects)
    accuracyVsThresh.append(accuracy)

  plt.figure()
  plt.plot(threshList, accuracyVsThresh, '-o')
  plt.xlabel('L4 distal Threshold')
  plt.ylabel('Classification Accuracy')
  plt.savefig('accuracyVsL4Thresh.pdf')
  return threshList, accuracyVsThresh


if __name__ == "__main__":

  threshList, accuracyVsThresh = runExperimentAccuracyVsL4Thresh()

  # numColumns = 1
  # l2Params = getL2Params()
  # l4Params = getL4Params()
  #
  # objects = loadThingObjects(numColumns, './data')
  #
  # expResult = trainNetwork(objects, numColumns, l4Params, l2Params, True)
  #
  # accuracy = computeAccuracy(expResult, objects)
  #
  # objectNames = objects.objects.keys()
  # numObjects = len(objectNames)
  #
  # distanceMat = expResult['overlapMat']
  # numL2ActiveCells = expResult['numL2ActiveCells']
  #
  # # plot pairwise overlap
  # plt.figure()
  # plt.imshow(expResult['overlapMat'])
  # plt.xticks(range(numObjects), objectNames, rotation='vertical', fontsize=5)
  # plt.yticks(range(numObjects), objectNames, fontsize=5)
  # plt.title('pairwise overlap')
  # plt.tight_layout()
  # plt.savefig('overlap_matrix.pdf')
  #
  # # plot number of active cells for each object
  # plt.figure()
  # objectNamesSort = []
  # idx = np.argsort(expResult['numL2ActiveCells'])
  # for i in idx:
  #   objectNamesSort.append(objectNames[i])
  # plt.plot(numL2ActiveCells[idx])
  # plt.xticks(range(numObjects), objectNamesSort, rotation='vertical', fontsize=5)
  # plt.tight_layout()
  # plt.ylabel('Number of active L2 cells')
  # plt.savefig('number_of_active_l2_cells.pdf')

