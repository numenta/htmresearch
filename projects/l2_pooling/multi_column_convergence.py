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
import os
import pprint
import numpy
import cPickle
from multiprocessing import Pool
import matplotlib.pyplot as plt

from htmresearch.frameworks.layers.l2_l4_inference import L4L2Experiment
from htmresearch.frameworks.layers.object_machine_factory import (
  createObjectMachine
)

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


def objectConfusion(objects):
  """
  For debugging, print overlap between each pair of objects.
  """
  for o1,s1 in objects.iteritems():
    for o2,s2 in objects.iteritems():
      # Count number of common locations id's and common feature id's
      commonLocations = 0
      commonFeatures = 0
      for pair1 in s1:
        for pair2 in s2:
          if pair1[0] == pair2[0]: commonLocations += 1
          if pair1[1] == pair2[1]: commonFeatures += 1

      print "Confusion",o1,o2,", common pairs=",len(set(s1)&set(s2)),
      print ", locations=",commonLocations,"features=",commonFeatures


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

  The method returns the args dict updated with two additional keys:
    convergencePoint (int)   The average number of iterations it took
                             to converge across all objects
    objects          (pairs) The list of objects we trained on
  """
  numObjects = args.get("numObjects", 10)
  numLocations = args.get("numLocations", 10)
  numFeatures = args.get("numFeatures", 10)
  numColumns = args.get("numColumns", 2)
  profile = args.get("profile", False)
  noiseLevel = args.get("noiseLevel", None)  # TODO: implement this?
  numPoints = args.get("numPoints", 10)
  trialNum = args.get("trialNum", 42)
  plotInferenceStats = args.get("plotInferenceStats", True)

  # Create the objects
  objects = createObjectMachine(
    machineType="simple",
    numInputBits=20,
    sensorInputSize=1024,
    externalInputSize=1024,
    numCorticalColumns=numColumns,
  )
  objects.createRandomObjects(numObjects, numPoints=numPoints,
                                    numLocations=numLocations,
                                    numFeatures=numFeatures)

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
    seed=trialNum
  )

  exp.learnObjects(objects.provideObjectsToLearn())
  if profile:
    exp.printProfile(reset=True)

  # For inference, we will check and plot convergence for each object. For each
  # object, we create a sequence of random sensations for each column.  We will
  # present each sensation for 3 time steps to let it settle and ensure it
  # converges.
  for objectId in objects:
    obj = objects[objectId]

    # Create sequence of sensations for this object for all columns
    objectSensations = {}
    for c in range(numColumns):
      objectCopy = [pair for pair in obj]
      random.shuffle(objectCopy)
      # stay multiple steps on each sensation
      sensations = []
      for pair in objectCopy:
        for _ in xrange(2):
          sensations.append(pair)
      objectSensations[c] = sensations

    inferConfig = {
      "object": objectId,
      "numSteps": len(objectSensations[0]),
      "pairs": objectSensations
    }

    exp.infer(objects.provideObjectToInfer(inferConfig), objectName=objectId)
    if profile:
      exp.printProfile(reset=True)

    if plotInferenceStats:
      exp.plotInferenceStats(
        fields=["L2 Representation",
                "Overlap L2 with object",
                "L4 Representation"],
        experimentID=objectId,
        onePlot=False,
      )

  convergencePoint = averageConvergencePoint(
    exp.getInferenceStats(),"L2 Representation", 40)

  print
  print "# objects {} # features {} # locations {} # columns {} trial # {}".format(
    numObjects, numFeatures, numLocations, numColumns, trialNum)
  print "Average convergence point=",convergencePoint

  # Return our convergence point as well as all the parameters and objects
  args.update({"objects": objects.getObjects()})
  args.update({"convergencePoint":convergencePoint})

  # Can't pickle experiment so can't return it. However this is very useful
  # for debugging when running in a single thread.
  # args.update({"experiment": exp})
  return args


def runExperimentPool(numObjects,
                      numLocations,
                      numFeatures,
                      numColumns,
                      numWorkers=7,
                      nTrials=1,
                      resultsName="convergence_results.pkl"):
  """
  Allows you to run a number of experiments using multiple processes.
  For each parameter except numWorkers, pass in a list containing valid values
  for that parameter. The cross product of everything is run, and each
  combination is run nTrials times.

  Returns a list of dict containing detailed results from each experiment.
  Also pickles the results in resultsName for later analysis.

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
  for t in range(nTrials):
    for c in numColumns:
      for o in numObjects:
        for l in numLocations:
          for f in numFeatures:
            args.append(
              {"numObjects": o,
               "numLocations": l,
               "numFeatures": f,
               "numColumns": c,
               "trialNum": t,
               "plotInferenceStats": False,
               }
            )

  print "{} experiments to run, {} workers".format(len(args), numWorkers)
  # Run the pool
  if numWorkers > 1:
    pool = Pool(processes=numWorkers)
    result = pool.map(runExperiment, args)
  else:
    result = []
    for arg in args:
      result.append(runExperiment(arg))

  print "Full results:"
  pprint.pprint(result, width=150)

  # Pickle results for later use
  with open(resultsName,"wb") as f:
    cPickle.dump(result,f)

  return result


def plotConvergenceStats(convergence, columnRange, featureRange):
  """
  Plots the convergence graph

  Convergence[f,c] = how long it took it to converge with f unique features
  and c columns.

  Features: the list of features we want to plot
  """
  plt.figure()
  plotPath = os.path.join("plots", "convergence_1.png")

  # Plot each curve
  colorList = {3: 'r', 5: 'b', 7: 'g', 11: 'k'}
  markerList = {3: 'o', 5: 'D', 7: '*', 11: 'x'}
  for f in featureRange:
    print columnRange
    print convergence[f-1,columnRange]
    plt.plot(columnRange, convergence[f-1,columnRange],
             color=colorList[f],
             marker=markerList[f])

  # format
  plt.legend(['Unique features=3', 'Unique features=5',
              'Unique features=7', 'Unique features=11'], loc="upper right")
  plt.xlabel("Columns")
  plt.xticks(columnRange)
  plt.ylabel("Number of sensations")
  plt.title("Convergence")

    # save
  plt.savefig(plotPath)
  plt.close()


if __name__ == "__main__":

  # This is how you run a specific experiment in single process mode. Useful
  # for debugging, profiling, etc.
  # results = runExperiment(
  #               {
  #                 "numObjects": 10,
  #                 "numLocations": 10,
  #                 "numFeatures": 7,
  #                 "numColumns": 3,
  #                 "trialNum": 0
  #               }
  # )


  # This is how you run a bunch of experiments in a process pool

  # Here we want to see how the number of columns affects convergence.
  # We run 10 trials for each column number and then analyze results
  numTrials = 4
  columnRange = [2,3,4,5,6,7]
  featureRange = [3,5,7,11]
  # Comment this out if you are re-running analysis on an already saved set of
  # results
  results = runExperimentPool(
                    numObjects=[10],
                    numLocations=[10],
                    numFeatures=featureRange,
                    numColumns=columnRange,
                    nTrials=numTrials)

  # Analyze results
  with open("convergence_results.pkl","rb") as f:
    results = cPickle.load(f)

  # Accumulate all the results per column in a numpy array, and print it as
  # well as raw results.  This part can be specific to each experiment
  convergence = numpy.zeros((max(featureRange), max(columnRange)+1))
  for r in results:
    convergence[r["numFeatures"]-1,
                r["numColumns"]] += r["convergencePoint"]/2.0

  convergence = convergence/numTrials + 1.0

  # For each column, print convergence as fct of number of unique features
  for c in range(2,max(columnRange)+1):
    print c,convergence[:, c]

  # Print everything anyway for debugging
  print "Average convergence array=",convergence

  plotConvergenceStats(convergence, columnRange, featureRange)

