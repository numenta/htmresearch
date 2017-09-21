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

import random

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
  @param pointRange  (int)   Creates objects each with points ranging from
                             [numPoints,...,numPoints+pointRange-1]
                             A total of numObjects * pointRange objects will be
                             created.
                             Default: 1
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

  The method returns the args dict updated with two additional keys:
    convergencePoint (int)   The average number of iterations it took
                             to converge across all objects
    objects          (pairs) The list of objects we trained on
  """
  numObjects = args.get("numObjects", 10)
  numLocations = args.get("numLocations", 10)
  numFeatures = args.get("numFeatures", 10)
  numColumns = args.get("numColumns", 2)
  networkType = args.get("networkType", "MultipleL4L2Columns")
  longDistanceConnections = args.get("longDistanceConnections", 0)
  profile = args.get("profile", False)
  locationNoise = args.get("locationNoise", 0.0)
  featureNoise = args.get("featureNoise", 0.0)
  numPoints = args.get("numPoints", 10)
  trialNum = args.get("trialNum", 42)
  pointRange = args.get("pointRange", 1)
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

  for p in range(pointRange):
    objects.createRandomObjects(numObjects, numPoints=numPoints+p,
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
  if profile:
    exp.printProfile(reset=True)

  # For inference, we will check and plot convergence for each object. For each
  # object, we create a sequence of random sensations for each column.  We will
  # present each sensation for settlingTime time steps to let it settle and
  # ensure it converges.
  numCorrectClassifications=0
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

    classificationResults = exp.getCurrentClassification(30, includeZeros=False)
    # print "Classification for object",objectId, "=", classificationResults
    if (classificationResults.get(objectId, 0.0)==1.0 and
        len(classificationResults)==1):
      numCorrectClassifications += 1

    if plotInferenceStats:
      exp.plotInferenceStats(
        fields=["L2 Representation",
                "Overlap L2 with object",
                "L4 Representation"],
        experimentID=objectId,
        onePlot=False,
      )


  convergencePoint = exp.averageConvergencePoint("L2 Representation", 30, 40,
                                                 settlingTime)
  classificationAccuracy = float(numCorrectClassifications) / numObjects

  print "# objects {} # features {} # locations {} # columns {} trial # {} network type {}".format(
    numObjects, numFeatures, numLocations, numColumns, trialNum, networkType)
  print "Average convergence point=",convergencePoint
  print "Classification accuracy=",classificationAccuracy
  print

  # Return our convergence point as well as all the parameters and objects
  args.update({"objects": objects.getObjects()})
  args.update({"convergencePoint":convergencePoint})
  args.update({"classificationAccuracy":classificationAccuracy})

  # Can't pickle experiment so can't return it for batch multiprocessing runs.
  # However this is very useful for debugging when running in a single thread.
  if plotInferenceStats:
    args.update({"experiment": exp})
  return args

