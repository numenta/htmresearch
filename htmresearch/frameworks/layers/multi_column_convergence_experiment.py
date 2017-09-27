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
This is an overall script to run various convergence experiments. It checks the
convergence of L4-L2 as you increase the number of columns under various
scenarios.
"""

import cPickle
from multiprocessing import Pool
import random
import time
import numpy

from htmresearch.frameworks.layers.l2_l4_inference import L4L2Experiment
from htmresearch.frameworks.layers.object_machine_factory import (
  createObjectMachine
)


def runExperiment(args):
  """
  Run experiment.  What did you think this does?

  args is a dict representing the parameters. We do it this way to support
  multiprocessing. args contains one or more of the following keys:

  @param featureNoise (float) Noise level to add to the features
                             during inference. Default: None
  @param locationNoise (float) Noise level to add to the locations
                             during inference. Default: None
  @param numObjects  (int)   The number of objects we will train.
                             Default: 10
  @param numPoints   (int)   The number of points on each object.
                             Default: 10
  @param numLocations (int)  For each point, the number of locations to choose
                             from.  Default: 10
  @param numFeatures (int)   For each point, the number of features to choose
                             from.  Default: 10
  @param numColumns  (int)   The total number of cortical columns in network.
                             Default: 2
  @param networkType (string)The type of network to use.  Options are:
                             "MultipleL4L2Columns",
                             "MultipleL4L2ColumnsWithTopology" and
                             "MultipleL4L2ColumnsWithRandomTopology".
                             Default: "MultipleL4L2Columns"
  @param longDistanceConnections (float) The probability that a column will
                             connect to a distant column.  Only relevant when
                             using the random topology network type.
                             If > 1, will instead be taken as desired number
                             of long-distance connections per column.
  @param settlingTime (int)  Number of iterations we wait to let columns
                             stabilize. Important for multicolumn experiments
                             with lateral connections.
  @param includeRandomLocation (bool) If True, a random location SDR will be
                             generated during inference for each feature.
  @param enableFeedback (bool) If True, enable feedback, default is True
  @param numAmbiguousLocations (int) number of ambiguous locations. Ambiguous
                             locations will present during inference if this
                             parameter is set to be a positive number

  The method returns the args dict updated with multiple additional keys
  representing accuracy metrics.
  """
  numObjects = args.get("numObjects", 10)
  numLocations = args.get("numLocations", 10)
  numFeatures = args.get("numFeatures", 10)
  numColumns = args.get("numColumns", 2)
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
  l2Params = args.get("l2Params", None)
  l4Params = args.get("l4Params", None)

  # Create the objects
  objects = createObjectMachine(
    machineType="simple",
    numInputBits=20,
    sensorInputSize=150,
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

  # Setup experiment and train the network
  name = "convergence_O%03d_L%03d_F%03d_C%03d_T%03d" % (
    numObjects, numLocations, numFeatures, numColumns, trialNum
  )
  exp = L4L2Experiment(
    name,
    numCorticalColumns=numColumns,
    L2Overrides=l2Params,
    L4Overrides=l4Params,
    networkType = networkType,
    longDistanceConnections=longDistanceConnections,
    inputSize=150,
    externalInputSize=2400,
    numInputBits=20,
    seed=trialNum,
    enableFeedback=enableFeedback,
  )

  exp.learnObjects(objects.provideObjectsToLearn())

  # For inference, we will check and plot convergence for each object. For each
  # object, we create a sequence of random sensations for each column.  We will
  # present each sensation for settlingTime time steps to let it settle and
  # ensure it converges.
  numCorrectClassifications=0
  classificationPerSensation = numpy.zeros(settlingTime*numPoints)
  for objectId in objects:
    exp.sendReset()

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
      random.shuffle(objectCopy)
      for pair in objectCopy:
        # stay multiple steps on each sensation
        for _ in xrange(settlingTime):
          objectSensations[0].append(pair)

    inferConfig = {
      "object": objectId,
      "numSteps": len(objectSensations[0]),
      "pairs": objectSensations,
      "noiseLevel": featureNoise,
      "locationNoise": locationNoise,
      "includeRandomLocation": includeRandomLocation,
      "numAmbiguousLocations": numAmbiguousLocations,
    }

    inferenceSDRs = objects.provideObjectToInfer(inferConfig)

    exp.infer(inferenceSDRs, objectName=objectId, reset=False)

    classificationPerSensation += numpy.array(
      exp.statistics[objectId]["Correct classification"])

    if exp.isObjectClassified(objectId, minOverlap=30):
      numCorrectClassifications += 1

    if plotInferenceStats:
      exp.plotInferenceStats(
        fields=["L2 Representation",
                "Overlap L2 with object",
                "L4 Representation"],
        experimentID=objectId,
        onePlot=False,
      )


  convergencePoint, accuracy = exp.averageConvergencePoint("L2 Representation",
                                                 30, 40, settlingTime)
  classificationAccuracy = float(numCorrectClassifications) / numObjects
  classificationPerSensation = classificationPerSensation / numObjects

  print "# objects {} # features {} # locations {} # columns {} trial # {} network type {}".format(
    numObjects, numFeatures, numLocations, numColumns, trialNum, networkType)
  print "Average convergence point=",convergencePoint
  print "Classification accuracy=",classificationAccuracy
  print

  # Return our convergence point as well as all the parameters and objects
  args.update({"objects": objects.getObjects()})
  args.update({"convergencePoint":convergencePoint})
  args.update({"classificationAccuracy":classificationAccuracy})
  args.update({"classificationPerSensation":classificationPerSensation.tolist()})

  # Can't pickle experiment so can't return it for batch multiprocessing runs.
  # However this is very useful for debugging when running in a single thread.
  if plotInferenceStats:
    args.update({"experiment": exp})
  return args


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

