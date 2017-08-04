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
This file plots the behavior of L4-L2-TM network as you train it on sequences.
"""

import random
import os
from math import ceil
import numpy
import cPickle
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

from htmresearch.frameworks.layers.combined_sequence_experiment import (
  L4TMExperiment
)
from htmresearch.frameworks.layers.object_machine_factory import (
  createObjectMachine
)


def locateConvergencePoint(stats, minOverlap, maxOverlap):
  """
  Walk backwards through stats until you locate the first point that diverges
  from target overlap values.  We need this to handle cases where it might get
  to target values, diverge, and then get back again.  We want the last
  convergence point.
  """
  for i,v in enumerate(stats[::-1]):
    if not (v >= minOverlap and v <= maxOverlap):
      return len(stats)-i + 1

  # Never differs - converged in one iteration
  return 1


def averageConvergencePoint(inferenceStats, prefix, minOverlap, maxOverlap,
                            settlingTime):
  """
  inferenceStats contains activity traces while the system visits each object.

  Given the i'th object, inferenceStats[i] contains activity statistics for
  each column for each region for the entire sequence of sensations.

  For each object, compute the convergence time - the first point when all
  L2 columns have converged.

  Return the average convergence time across all objects.

  Given inference statistics for a bunch of runs, locate all traces with the
  given prefix. For each trace locate the iteration where it finally settles
  on targetValue. Return the average settling iteration across all runs.
  """
  convergenceSum = 0.0

  # For each object
  for stats in inferenceStats:

    # For each L2 column locate convergence time
    convergencePoint = 0.0
    for key in stats.iterkeys():
      if prefix in key:
        columnConvergence = locateConvergencePoint(
          stats[key], minOverlap, maxOverlap)

        # Ensure this column has converged by the last iteration
        # assert(columnConvergence <= len(stats[key]))

        convergencePoint = max(convergencePoint, columnConvergence)

    convergenceSum += ceil(float(convergencePoint)/settlingTime)

  return convergenceSum/len(inferenceStats)


def runExperiment(args):
  """
  Runs the experiment.  What did you think this does?

  args is a dict representing the parameters. We do it this way to support
  multiprocessing. args contains one or more of the following keys:

  @param noiseLevel  (float) Noise level to add to the locations and features
                             during inference. Default: None
  @param profile     (bool)  If True, the network will be profiled after
                             learning and inference. Default: False
  @param numSequences (int)  The number of objects (sequences) we will train.
                             Default: 10
  @param seqLength   (int)   The number of points on each object (length of
                             each sequence).
                             Default: 10
  @param numFeatures (int)   For each point, the number of features to choose
                             from.  Default: 10
  @param numColumns  (int)   The total number of cortical columns in network.
                             Default: 2

  The method returns the args dict updated with two additional keys:
    convergencePoint (int)   The average number of iterations it took
                             to converge across all objects
    objects          (pairs) The list of objects we trained on
  """
  numSequences = args.get("numSequences", 10)
  numFeatures = args.get("numFeatures", 10)
  numColumns = args.get("numColumns", 2)
  networkType = args.get("networkType", "L4L2TMColumn")
  profile = args.get("profile", False)
  noiseLevel = args.get("noiseLevel", None)  # TODO: implement this?
  seqLength = args.get("seqLength", 10)
  trialNum = args.get("trialNum", 42)
  plotInferenceStats = args.get("plotInferenceStats", True)


  # Create the objects
  objects = createObjectMachine(
    machineType="sequence",
    numInputBits=20,
    sensorInputSize=150,
    externalInputSize=2400,
    numCorticalColumns=numColumns,
    numFeatures=numFeatures,
    seed=trialNum
  )
  objects.createRandomSequences(numSequences, seqLength)

  # print "Sequences:"
  # print objects.getObjects()

  r = objects.objectConfusion()
  print "Average common pairs=", r[0],
  print ", features=",r[2]

  # Setup experiment and train the network
  name = "sequence_convergence_S%03d_F%03d_C%03d_T%03d" % (
    numSequences, numFeatures, numColumns, trialNum
  )
  exp = L4TMExperiment(
    name=name,
    numCorticalColumns=numColumns,
    networkType = networkType,
    inputSize=150,
    externalInputSize=2400,
    numInputBits=20,
    seed=trialNum,
    logCalls=False
  )

  # Train the network on all the SDRs for all the objects
  objectSDRs = objects.provideObjectsToLearn()

  # Make sure we learn enough times to deal with high order sequences and
  # remove extra predictions
  for _ in range(3*seqLength):
    exp.learnObjects(objectSDRs)
  if profile:
    exp.printProfile(reset=True)

  # For inference, we will check and plot convergence for each sequence. We
  # don't want to shuffle them!
  for objectId in objects:
    obj = objects[objectId]

    objectSensations = {}
    for c in range(numColumns):
      objectSensations[c] = []

    # Create sequence of sensations for this object for one column. The total
    # number of sensations is equal to the number of points on the object. No
    # point should be visited more than once.
    objectCopy = [pair for pair in obj]
    for pair in objectCopy:
      objectSensations[0].append(pair)

    inferConfig = {
      "object": objectId,
      "numSteps": len(objectSensations[0]),
      "pairs": objectSensations,
    }

    inferenceSDRs = objects.provideObjectToInfer(inferConfig)
    # print "Inference SDRs", inferenceSDRs

    exp.infer(inferenceSDRs, objectName=objectId)

    if plotInferenceStats:
      exp.plotInferenceStats(
        fields=[
                # "L2 Representation",
                "Overlap L2 with object",
                "TM Basal Segments",
                # "L4 Representation",
                "TM PredictedActive",
                ],
        experimentID=objectId,
        onePlot=False,
      )

  # Compute overall inference statistics
  infStats = exp.getInferenceStats()
  convergencePoint = averageConvergencePoint(
    infStats,"L2 Representation", 30, 40, 1)

  numPredictions = 0.0
  sumPredictions = 0.0
  for stat in infStats:
    predictedActiveTrace = stat["TM PredictedActive C0"]
    # print predictedActiveTrace
    numPredictions += len(predictedActiveTrace)
    sumPredictions += sum(predictedActiveTrace)
  averagePredictions = sumPredictions / numPredictions

  print "# Sequences {} # features {} # columns {} trial # {} network type {}".format(
    numSequences, numFeatures, numColumns, trialNum, networkType)
  print "Average convergence point=",convergencePoint
  print

  # Return our convergence point as well as all the parameters and objects
  args.update({"objects": objects.getObjects()})
  args.update({"convergencePoint":convergencePoint})
  args.update({"averagePredictions": averagePredictions})

  # Can't pickle experiment so can't return it for batch multiprocessing runs.
  # However this is very useful for debugging when running in a single thread.
  # if plotInferenceStats:
  #   args.update({"experiment": exp})
  return args


def runExperimentPool(numSequences,
                      numFeatures,
                      numColumns,
                      networkType=["L4L2TMColumn"],
                      numWorkers=7,
                      nTrials=1,
                      seqLength=10,
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
                          numSequences=[10],
                          numFeatures=[5],
                          numColumns=[2,3,4,5,6],
                          numWorkers=8,
                          nTrials=5)
  """
  # Create function arguments for every possibility
  args = []

  for c in reversed(numColumns):
    for o in reversed(numSequences):
        for f in numFeatures:
          for n in networkType:
            for t in range(nTrials):
              args.append(
                {"numSequences": o,
                 "numFeatures": f,
                 "numColumns": c,
                 "trialNum": t,
                 "seqLength": seqLength,
                 "networkType" : n,
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

  # Pickle results for later use
  with open(resultsName,"wb") as f:
    cPickle.dump(result,f)

  return result


def plotConvergenceBySequence(results, objectRange, featureRange, numTrials):
  """
  Plots the convergence graph: iterations vs number of objects.
  Each curve shows the convergence for a given number of unique features.
  """
  ########################################################################
  #
  # Accumulate all the results per column in a convergence array.
  #
  # Convergence[f,o] = how long it took it to converge with f unique features
  # and o objects.

  convergence = numpy.zeros((max(featureRange), max(objectRange) + 1))
  for r in results:
    if r["numFeatures"] in featureRange:
      convergence[r["numFeatures"] - 1, r["numSequences"]] += r["convergencePoint"]

  convergence /= numTrials

  ########################################################################
  #
  # Create the plot. x-axis=
  plt.figure()
  plotPath = os.path.join("plots", "convergence_by_sequence.pdf")

  # Plot each curve
  legendList = []
  colorList = ['r', 'b', 'g', 'm', 'c', 'k', 'y']

  for i in range(len(featureRange)):
    f = featureRange[i]
    print "features={} objectRange={} convergence={}".format(
      f,objectRange, convergence[f-1,objectRange])
    legendList.append('Unique features={}'.format(f))
    plt.plot(objectRange, convergence[f-1,objectRange],
             color=colorList[i])

  # format
  plt.legend(legendList, loc="lower right", prop={'size':10})
  plt.xlabel("Number of sequences in training set")
  plt.xticks(range(0,max(objectRange)+1,10))
  plt.yticks(range(0,int(convergence.max())+2))
  plt.ylabel("Average number of touches")
  plt.title("Number of touches to recognize a sequence")

    # save
  plt.savefig(plotPath)
  plt.close()


def plotPredictionsBySequence(results, objectRange, featureRange, numTrials):
  """
  Plots the convergence graph: iterations vs number of objects.
  Each curve shows the convergence for a given number of unique features.
  """
  ########################################################################
  #
  # Accumulate all the results per column in a convergence array.
  #
  # predictions[f,o] = how long it took it to converge with f unique features
  # and o objects.

  predictions = numpy.zeros((max(featureRange), max(objectRange) + 1))
  for r in results:
    if r["numFeatures"] in featureRange:
      predictions[r["numFeatures"] - 1, r["numSequences"]] += r["averagePredictions"]

  predictions /= numTrials

  ########################################################################
  #
  # Create the plot. x-axis=
  plt.figure()
  plotPath = os.path.join("plots", "predictions_by_sequence.pdf")

  # Plot each curve
  legendList = []
  colorList = ['r', 'b', 'g', 'm', 'c', 'k', 'y']

  for i in range(len(featureRange)):
    f = featureRange[i]
    print "features={} objectRange={} convergence={}".format(
      f,objectRange, predictions[f-1,objectRange])
    legendList.append('Unique features={}'.format(f))
    plt.plot(objectRange, predictions[f-1,objectRange],
             color=colorList[i])

  # format
  plt.legend(legendList, loc="center right", prop={'size':10})
  plt.xlabel("Number of sequences in training set")
  plt.xticks(range(0,max(objectRange)+1,10))
  plt.yticks(range(0,int(predictions.max())+2,10))
  plt.ylabel("Average number of predicted cells")
  plt.title("Predictions in TM while inferring sequences")

    # save
  plt.savefig(plotPath)
  plt.close()

if __name__ == "__main__":

  # This is how you run a specific experiment in single process mode. Useful
  # for debugging, profiling, etc.
  if True:
    results = runExperiment(
                  {
                    "numSequences": 10,
                    "seqLength": 10,
                    "numFeatures": 100,
                    "numColumns": 1,
                    "trialNum": 4,
                    "plotInferenceStats": True,  # Outputs detailed graphs
                  }
              )
  # Pickle results for later use
  with open("one_sequence_convergence_results.pkl","wb") as f:
    cPickle.dump(results,f)


  # Here we want to see how the number of objects affects convergence for a
  # single column.
  # This experiment is run using a process pool
  if False:
    # We run 10 trials for each column number and then analyze results
    numTrials = 10
    columnRange = [1]
    featureRange = [10, 100, 1000, 10000]
    seqRange = [2,5,10,20,30]

    # Comment this out if you are re-running analysis on already saved results.
    # Very useful for debugging the plots
    runExperimentPool(
                      numSequences=seqRange,
                      numFeatures=featureRange,
                      numColumns=columnRange,
                      seqLength=10,
                      nTrials=numTrials,
                      numWorkers=cpu_count() - 1,
                      resultsName="sequence_convergence_results.pkl")

    # Analyze results
    with open("sequence_convergence_results.pkl","rb") as f:
      results = cPickle.load(f)

    plotConvergenceBySequence(results, seqRange, featureRange, numTrials)

    plotPredictionsBySequence(results, seqRange, featureRange, numTrials)

