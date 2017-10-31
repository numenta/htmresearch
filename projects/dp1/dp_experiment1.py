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
This is for running some very preliminary disjoint pooling experiments.
"""

import cPickle
from multiprocessing import Pool
import random
import time
import numpy

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

from htmresearch.frameworks.layers.l2_l4_inference import L4L2Experiment
from htmresearch.frameworks.layers.object_machine_factory import (
  createObjectMachine
)


def printColumnPoolerDiagnostics(pooler):
  print "sampleSizeProximal: ", pooler.sampleSizeProximal
  print "Average number of proximal synapses per cell:",
  print float(pooler.numberOfProximalSynapses()) / pooler.cellCount

  print "Average number of distal segments per cell:",
  print float(pooler.numberOfDistalSegments()) / pooler.cellCount

  print "Average number of connected distal synapses per cell:",
  print float(pooler.numberOfConnectedDistalSynapses()) / pooler.cellCount

  print "Average number of distal synapses per cell:",
  print float(pooler.numberOfDistalSynapses()) / pooler.cellCount


def runExperiment(args):
  """
  Run experiment.  args is a dict representing the parameters. We do it this way
  to support multiprocessing.

  The method returns the args dict updated with multiple additional keys
  representing accuracy metrics.
  """
  numObjects = args.get("numObjects", 10)
  numLocations = args.get("numLocations", 10)
  numFeatures = args.get("numFeatures", 10)
  numColumns = args.get("numColumns", 2)
  sensorInputSize = args.get("sensorInputSize", 300)
  networkType = args.get("networkType", "MultipleL4L2Columns")
  longDistanceConnections = args.get("longDistanceConnections", 0)
  locationNoise = args.get("locationNoise", 0.0)
  featureNoise = args.get("featureNoise", 0.0)
  numPoints = args.get("numPoints", 10)
  trialNum = args.get("trialNum", 42)
  plotInferenceStats = args.get("plotInferenceStats", True)
  settlingTime = args.get("settlingTime", 3)
  includeRandomLocation = args.get("includeRandomLocation", False)
  enableFeedback = args.get("enableFeedback", True)
  numAmbiguousLocations = args.get("numAmbiguousLocations", 0)
  numInferenceRpts = args.get("numInferenceRpts", 1)
  numLearningRpts = args.get("numLearningRpts", 3)
  l2Params = args.get("l2Params", None)
  l4Params = args.get("l4Params", None)

  # Create the objects
  objects = createObjectMachine(
    machineType="simple",
    numInputBits=20,
    sensorInputSize=sensorInputSize,
    externalInputSize=2400,
    numCorticalColumns=numColumns,
    numFeatures=numFeatures,
    numLocations=numLocations,
    seed=trialNum
  )

  objects.createRandomObjects(numObjects, numPoints=numPoints,
                                    numLocations=numLocations,
                                    numFeatures=numFeatures)

  r = objects.objectConfusion()
  print "Average common pairs in objects=", r[0],
  print ", locations=",r[1],", features=",r[2]

  # print "Total number of objects created:",len(objects.getObjects())
  # print "Objects are:"
  # for o in objects:
  #   pairs = objects[o]
  #   pairs.sort()
  #   print str(o) + ": " + str(pairs)

  # This object machine will simulate objects where each object is just one
  # unique feature/location pair. We will use this to pretrain L4/L2 with
  # individual pairs.
  pairObjects = createObjectMachine(
    machineType="simple",
    numInputBits=20,
    sensorInputSize=sensorInputSize,
    externalInputSize=2400,
    numCorticalColumns=numColumns,
    numFeatures=numFeatures,
    numLocations=numLocations,
    seed=trialNum
  )

  # Create "pair objects" consisting of all unique F/L pairs from our objects.
  # These pairs should have the same SDRs as the original objects.
  pairObjects.locations = objects.locations
  pairObjects.features = objects.features
  distinctPairs = objects.getDistinctPairs()
  print "Number of distinct feature/location pairs:",len(distinctPairs)
  for pairNumber,pair in enumerate(distinctPairs):
    pairObjects.addObject([pair], pairNumber)

  #####################################################
  #
  # Setup experiment and train the network
  name = "dp_O%03d_L%03d_F%03d_C%03d_T%03d" % (
    numObjects, numLocations, numFeatures, numColumns, trialNum
  )
  exp = L4L2Experiment(
    name,
    numCorticalColumns=numColumns,
    L2Overrides=l2Params,
    L4Overrides=l4Params,
    networkType = networkType,
    longDistanceConnections=longDistanceConnections,
    inputSize=sensorInputSize,
    externalInputSize=2400,
    numInputBits=20,
    seed=trialNum,
    enableFeedback=enableFeedback,
    numLearningPoints=numLearningRpts,
  )

  # Learn all FL pairs in each L4 and in each L2

  # Learning in L2 involves choosing a small random number of cells, growing
  # proximal synapses to L4 cells. Growing distal synapses to active cells in
  # each neighboring column. Each column gets its own distal segment.
  exp.learnObjects(pairObjects.provideObjectsToLearn())

  # Verify that all columns learned the pairs
  # numCorrectClassifications = 0
  # for pairId in pairObjects:
  #
  #   obj = pairObjects[pairId]
  #   objectSensations = {}
  #   for c in range(numColumns):
  #     objectSensations[c] = [obj[0]]*settlingTime
  #
  #   inferConfig = {
  #     "object": pairId,
  #     "numSteps": settlingTime,
  #     "pairs": objectSensations,
  #   }
  #
  #   inferenceSDRs = pairObjects.provideObjectToInfer(inferConfig)
  #
  #   exp.infer(inferenceSDRs, objectName=pairId, reset=False)
  #
  #   if exp.isObjectClassified(pairId, minOverlap=30):
  #     numCorrectClassifications += 1
  #
  #   exp.sendReset()
  #
  # print "Classification accuracy for pairs=",100.0*numCorrectClassifications/len(distinctPairs)

  ########################################################################
  #
  # Create "object representations" in L2 by simultaneously invoking the union
  # of all FL pairs in an object and doing some sort of spatial pooling to
  # create L2 representation.

  exp.resetStatistics()
  for objectId in objects:
    # Create one sensation per object consisting of the union of all features
    # and the union of locations.
    ul, uf = objects.getUniqueFeaturesLocationsInObject(objectId)
    print "Object",objectId,"Num unique features:",len(uf),"Num unique locations:",len(ul)
    objectSensations = {}
    for c in range(numColumns):
      objectSensations[c] = [(tuple(ul),  tuple(uf))]*settlingTime

    inferConfig = {
      "object": objectId,
      "numSteps": settlingTime,
      "pairs": objectSensations,
    }

    inferenceSDRs = objects.provideObjectToInfer(inferConfig)

    exp.infer(inferenceSDRs, objectName="Object "+str(objectId))


  # Compute confusion matrix between all objects as network settles
  for iteration in range(settlingTime):
    confusion = numpy.zeros((numObjects, numObjects))
    for o1 in objects:
      for o2 in objects:
        confusion[o1, o2] = len(set(exp.statistics[o1]["Full L2 SDR C0"][iteration]) &
                                set(exp.statistics[o2]["Full L2 SDR C0"][iteration]) )

    plt.figure()
    plt.imshow(confusion)
    plt.xlabel('Object #')
    plt.ylabel('Object #')
    plt.title("Object overlaps")
    plt.colorbar()
    plt.savefig("confusion_random_10L_5F_"+str(iteration)+".pdf")
    plt.close()


  for col in range(numColumns):
    print "Diagnostics for column",col
    printColumnPoolerDiagnostics(exp.getAlgorithmInstance(column=col))
    print

  return args


  # Show average overlap as a function of number of shared FL pairs,
  # shared locations, shared features

  # Compute confusion matrix showing number of shared FL pairs

  # Compute confusion matrix using our normal method



def runExperimentPool(numObjects,
                      numLocations,
                      numFeatures,
                      numColumns,
                      longDistanceConnectionsRange = [0.0],
                      numWorkers=7,
                      nTrials=1,
                      numPoints=10,
                      locationNoiseRange=[0.0],
                      featureNoiseRange=[0.0],
                      enableFeedback=[True],
                      ambiguousLocationsRange=[0],
                      numInferenceRpts=1,
                      settlingTime=3,
                      l2Params=None,
                      l4Params=None,
                      resultsName="convergence_results.pkl"):
  """
  Allows you to run a number of experiments using multiple processes.
  For each parameter except numWorkers, pass in a list containing valid values
  for that parameter. The cross product of everything is run, and each
  combination is run nTrials times.

  Returns a list of dict containing detailed results from each experiment.
  Also pickles and saves the results in resultsName for later analysis.

  Example:
    results = runExperimentPool(
                          numObjects=[10],
                          numLocations=[5],
                          numFeatures=[5],
                          numColumns=[2,3,4,5,6],
                          numWorkers=8,
                          nTrials=5)
  """
  # Create function arguments for every possibility
  args = []

  for c in reversed(numColumns):
    for o in reversed(numObjects):
      for l in numLocations:
        for f in numFeatures:
          for p in longDistanceConnectionsRange:
            for t in range(nTrials):
              for locationNoise in locationNoiseRange:
                for featureNoise in featureNoiseRange:
                  for ambiguousLocations in ambiguousLocationsRange:
                    for feedback in enableFeedback:
                      args.append(
                        {"numObjects": o,
                         "numLocations": l,
                         "numFeatures": f,
                         "numColumns": c,
                         "trialNum": t,
                         "numPoints": numPoints,
                         "longDistanceConnections" : p,
                         "plotInferenceStats": False,
                         "locationNoise": locationNoise,
                         "featureNoise": featureNoise,
                         "enableFeedback": feedback,
                         "numAmbiguousLocations": ambiguousLocations,
                         "numInferenceRpts": numInferenceRpts,
                         "l2Params": l2Params,
                         "l4Params": l4Params,
                         "settlingTime": settlingTime,
                         }
              )
  numExperiments = len(args)
  print "{} experiments to run, {} workers".format(numExperiments, numWorkers)
  # Run the pool
  if numWorkers > 1:
    pool = Pool(processes=numWorkers)
    rs = pool.map_async(runExperiment, args, chunksize=1)
    while not rs.ready():
      remaining = rs._number_left
      pctDone = 100.0 - (100.0*remaining) / numExperiments
      print "    =>", remaining, "experiments remaining, percent complete=",pctDone
      time.sleep(5)
    pool.close()  # No more work
    pool.join()
    result = rs.get()
  else:
    result = []
    for arg in args:
      result.append(runExperiment(arg))

  # print "Full results:"
  # pprint.pprint(result, width=150)

  # Pickle results for later use
  with open(resultsName,"wb") as f:
    cPickle.dump(result,f)

  return result



if __name__ == "__main__":

  # This is how you run a specific experiment in single process mode. Useful
  # for debugging, profiling, etc.
  results = runExperiment(
      {
        "numObjects": 20,
        "numPoints": 10,
        "numLocations": 10,
        "numFeatures": 5,
        "numColumns": 1,
        "trialNum": 4,
        "settlingTime": 3,
        "plotInferenceStats": False,  # Outputs detailed graphs
      }
  )
