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
This file plots the convergence of L4-L2 as you increase the number of columns,
or adjust the confusion between objects.
"""

import random
import pprint
import numpy
from multiprocessing import Pool

from htmresearch.frameworks.layers.l2_l4_inference import L4L2Experiment

def locateConvergencePoint(stats, targetValue):
  """
  Walk backwards through stats until you locate the first point that diverges
  from targetValue.  We need this to handle cases where it might get to
  targetValue, diverge, and then get back again.  We want the last convergence
  point.
  """
  for i,v in enumerate(stats[::-1]):
    if v != targetValue:
      return len(stats)-i

  # Never differs - converged right away
  return 0


def averageConvergencePoint(inferenceStats, prefix, targetValue):
  """
  Given inference statistics for a bunch of runs, locate all traces with the
  given prefix. For each trace locate the iteration where it finally settles
  on targetValue. Return the average settling iteration across all runs.
  """
  itSum = 0
  itNum = 0
  for stats in inferenceStats:
    for key in stats.iterkeys():
      if prefix in key:
        itSum += locateConvergencePoint(stats[key], targetValue)
        itNum += 1

  return float(itSum)/itNum


def runExperiment(args):
  """
  Run experiment.  What did you think this does?

  args is a dict representing the parameters. We do it this way to support
  multiprocessing. args contains one or more of the following keys:

  @param noiseLevel  (float) Noise level to add to the locations and features
                             during inference. Default: None
  @param profile     (bool)  If True, the network will be profiled after
                             learning and inference. Default: False
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

  """
  numObjects = args.get("numObjects", 10)
  numLocations = args.get("numLocations", 10)
  numFeatures = args.get("numFeatures", 10)
  numColumns = args.get("numColumns", 2)
  profile = args.get("profile", False)
  noiseLevel = args.get("noiseLevel", None)
  numPoints = args.get("numPoints", 10)

  # print "\n==============\nRunning experiment with params:"
  pprint.pprint(args)
  name = "convergence_O%03d_L%03d_F%03d_C%03d" % (
    numObjects, numLocations, numFeatures, numColumns
  )
  exp = L4L2Experiment(
    name,
    numCorticalColumns=numColumns,
  )

  # Create the objects and train the network
  objects = exp.createRandomObjects(numObjects, numPoints=numPoints,
                                    numLocations=numLocations,
                                    numFeatures=numFeatures)
  # print "Objects are:"
  for obj, pairs in objects.iteritems():
    pairs.sort()
    # print str(obj) + ": " + str(pairs)

  exp.learnObjects(objects)
  if profile:
    exp.printProfile(reset=True)

  # For inference, we will check and plot convergence for each object. For each
  # object, we create a sequence of random sensations for each column.  We will
  # present each sensation for 3 time steps to let it settle and ensure it
  # converges.
  for objectId, obj in objects.iteritems():
    # Create sequence of sensations for this object for all columns
    objectSensations = {}
    for c in range(numColumns):
      objectCopy = [pair for pair in obj]
      random.shuffle(objectCopy)
      # stay multiple steps on each sensation
      sensations = []
      for pair in objectCopy:
        for _ in xrange(3):
          sensations.append(pair)
      objectSensations[c] = sensations

    inferConfig = {
      "object": objectId,
      "numSteps": len(objectSensations[0]),
      "pairs": objectSensations
    }

    exp.infer(inferConfig, noise=noiseLevel)
    if profile:
      exp.printProfile(reset=True)

    exp.plotInferenceStats(
      fields=["L2 Representation",
              "Overlap L2 with object",
              "L4 Representation"],
      experimentID=objectId,
      onePlot=False,
    )

  convergencePoint = averageConvergencePoint(
    exp.getInferenceStats(),"L2 Representation", 40)
  print "Average convergence point=",convergencePoint

  # Return our convergence point as well as all the parameters and objects
  args.update({"objects": objects})
  args.update({"convergencePoint":convergencePoint})
  return args


def runExperimentPool(numObjects,
                      numLocations,
                      numFeatures,
                      numColumns,
                      numWorkers=8):
  """
  Allows you to run a number of experiments using multiple processes.
  For each parameter except numWorkers, pass in a list containing valid values
  for that parameter. The cross product of everything is run.

  Returns a dict containing detailed results from each experiment

  Example:
    results = runExperimentPool(
                          numObjects=[10],
                          numLocations=[5],
                          numFeatures=[5],
                          numColumns=[2,3,4,5,6],
                          numWorkers=8)
  """
  # Create function arguments for every possibility
  args = []
  for c in numColumns:
    for o in numObjects:
      for l in numLocations:
        for f in numFeatures:
          args.append(
            {"numObjects": o,
             "numLocations": l,
             "numFeatures": f,
             "numColumns": c
             }
          )

  # Run the pool
  pool = Pool(processes=numWorkers)
  result = pool.map(runExperiment, args)

  return result


if __name__ == "__main__":

  results = runExperimentPool(
                    numObjects=[10],
                    numLocations=[50],
                    numFeatures=[50],
                    numColumns=[2,3,4,5,6],
                    numWorkers=8)

  print "Full results:"
  pprint.pprint(results, width=150)

  # Accumulate all the results per column in a numpy array, and return it as
  # well as raw results
  # convergence = numpy.zeros(7)
  # for r in results:
  #   convergence[r["numColumns"]] = r["convergencePoint"]
  # print "Convergence array=",convergence[1:]
