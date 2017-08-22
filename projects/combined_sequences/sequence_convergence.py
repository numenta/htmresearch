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

import os
from math import ceil
import numpy
import time
import cPickle
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

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
  numCorrect = 0.0
  inferenceLength = 1000000

  # For each object
  for stats in inferenceStats:

    # For each L2 column locate convergence time
    convergencePoint = 0.0
    for key in stats.iterkeys():
      if prefix in key:
        inferenceLength = len(stats[key])
        columnConvergence = locateConvergencePoint(
          stats[key], minOverlap, maxOverlap)

        # Ensure this column has converged by the last iteration
        # assert(columnConvergence <= len(stats[key]))

        convergencePoint = max(convergencePoint, columnConvergence)

    convergenceSum += ceil(float(convergencePoint)/settlingTime)

    if ceil(float(convergencePoint)/settlingTime) <= inferenceLength:
      numCorrect += 1

  return convergenceSum/len(inferenceStats), numCorrect/len(inferenceStats)


def runExperiment(args):
  """
  Runs the experiment.  What did you think this does?

  args is a dict representing the parameters. We do it this way to support
  multiprocessing. args contains one or more of the following keys:

  @param noiseLevel  (float) Noise level to add to the locations and features
                             during inference. Default: None
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
  numColumns = args.get("numColumns", 1)
  networkType = args.get("networkType", "L4L2TMColumn")
  noiseLevel = args.get("noiseLevel", None)  # TODO: implement this?
  seqLength = args.get("seqLength", 10)
  trialNum = args.get("trialNum", 42)
  plotInferenceStats = args.get("plotInferenceStats", True)
  inputSize = args.get("inputSize", 512)
  numLocations = args.get("numLocations", 100000)


  # Create the objects
  objects = createObjectMachine(
    machineType="sequence",
    numInputBits=20,
    sensorInputSize=inputSize,
    externalInputSize=1024,
    numCorticalColumns=numColumns,
    numFeatures=numFeatures,
    numLocations=numLocations,
    seed=trialNum
  )
  objects.createRandomSequences(numSequences, seqLength)

  # print "Sequences:"
  # for i in objects:
  #   print i,objects[i]

  r = objects.objectConfusion()
  print "Average common pairs=", r[0],
  print ", features=",r[2]

  # Setup experiment and train the network
  name = "sequences_S%03d_F%03d_L%03d_T%03d" % (
    numSequences, numFeatures, numLocations, trialNum
  )
  exp = L4TMExperiment(
    name=name,
    numCorticalColumns=numColumns,
    networkType = networkType,
    inputSize=inputSize,
    externalInputSize=1024,
    numInputBits=20,
    seed=trialNum,
    L4Overrides={"initialPermanence": 0.41,
                 "activationThreshold": 18,
                 "minThreshold": 18},
    logCalls=False
  )

  # Train the network on all the sequences
  for seqName in objects:

    # Make sure we learn enough times to deal with high order sequences and
    # remove extra predictions.
    for p in range(3*seqLength):

      # Ensure we generate new random location for each sequence presentation
      objectSDRs = objects.provideObjectsToLearn([seqName])
      exp.learnObjects(objectSDRs, reset=False)

      # TM needs reset between sequences, but not other regions
      exp.TMColumns[0].reset()

    # L2 needs resets when we switch to new object
    exp.sendReset()

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

    exp.infer(inferenceSDRs, objectName=objectId)

    if plotInferenceStats:
      plotOneInferenceRun(
        exp.statistics[objectId],
        fields=[
          # ("L4 Predicted", "Predicted sensorimotor cells"),
          # ("L2 Representation", "L2 Representation"),
          # ("L4 Representation", "Active sensorimotor cells"),
          ("L4 PredictedActive", "Predicted active cells in sensorimotor layer"),
          ("TM Predicted", "Predicted cells in temporal sequence layer"),
          ("TM PredictedActive", "Predicted active cells in temporal sequence layer"),
        ],
        basename=exp.name,
        experimentID=objectId,
        plotDir=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "detailed_plots")
      )

  # Compute overall inference statistics
  infStats = exp.getInferenceStats()
  convergencePoint, accuracy = averageConvergencePoint(
    infStats,"L2 Representation", 30, 40, 1)

  predictedActive = numpy.zeros(len(infStats))
  predicted = numpy.zeros(len(infStats))
  predictedActiveL4 = numpy.zeros(len(infStats))
  predictedL4 = numpy.zeros(len(infStats))
  for i,stat in enumerate(infStats):
    predictedActive[i] = float(sum(stat["TM PredictedActive C0"][2:])) / len(stat["TM PredictedActive C0"][2:])
    predicted[i] = float(sum(stat["TM Predicted C0"][2:])) / len(stat["TM Predicted C0"][2:])

    predictedActiveL4[i] = float(sum(stat["L4 PredictedActive C0"])) / len(stat["L4 PredictedActive C0"])
    predictedL4[i] = float(sum(stat["L4 Predicted C0"])) / len(stat["L4 Predicted C0"])

  print "# Sequences {} # features {} # columns {} trial # {} network type {}".format(
    numSequences, numFeatures, numColumns, trialNum, networkType)
  print "Average convergence point=",convergencePoint,
  print "Accuracy:", accuracy
  print

  # Return our convergence point as well as all the parameters and objects
  args.update({"objects": objects.getObjects()})
  args.update({"convergencePoint":convergencePoint})
  args.update({"sensorimotorAccuracyPct": accuracy})
  args.update({"averagePredictions": predicted.mean()})
  args.update({"averagePredictedActive": predictedActive.mean()})
  args.update({"averagePredictionsL4": predictedL4.mean()})
  args.update({"averagePredictedActiveL4": predictedActiveL4.mean()})

  # Can't pickle experiment so can't return it for batch multiprocessing runs.
  # However this is very useful for debugging when running in a single thread.
  # if plotInferenceStats:
  #   args.update({"experiment": exp})
  return args


def runExperimentPool(numSequences,
                      numFeatures,
                      numLocations,
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

  for o in reversed(numSequences):
    for l in numLocations:
      for f in numFeatures:
        for n in networkType:
          for t in range(nTrials):
            args.append(
              {"numSequences": o,
               "numFeatures": f,
               "trialNum": t,
               "seqLength": seqLength,
               "networkType" : n,
               "numLocations": l,
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
  plotPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "plots", "convergence_by_sequence.pdf")

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


def plotSensorimotorAccuracy(results, locationRange, featureRange,
                             seqRange, title="", yaxis=""):
  """
  Plot accuracy vs number of locations
  """

  ########################################################################
  #
  # Accumulate all the results per column in a convergence array.
  #
  # accuracy[f,l] = how long it took it to converge with f unique features
  # and l locations on average.
  accuracy = numpy.zeros((max(featureRange)+1, max(locationRange) + 1))
  totals = numpy.zeros((max(featureRange)+1, max(locationRange) + 1))
  for r in results:
    if r["numFeatures"] in featureRange and r["numSequences"] in seqRange:
      accuracy[r["numFeatures"], r["numLocations"]] += r["sensorimotorAccuracyPct"]
      totals[r["numFeatures"], r["numLocations"]] += 1

  for f in featureRange:
    for l in locationRange:
      accuracy[f, l] = 100.0*accuracy[f, l] / totals[f, l]

  ########################################################################
  #
  # Create the plot.
  plt.figure()
  plotPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "plots", "sensorimotorAccuracy_by_sequence.pdf")

  # Plot each curve
  legendList = []
  colorList = ['r', 'b', 'g', 'm', 'c', 'k', 'y']

  for i in range(len(featureRange)):
    f = featureRange[i]
    print "features={} locationRange={} accuracy={}".format(
      f,locationRange, accuracy[f,locationRange]),
    print totals[f,locationRange]
    legendList.append('Sensorimotor layer, feature pool size: {}'.format(f))
    plt.plot(locationRange, accuracy[f,locationRange],
             color=colorList[i])

  plt.plot(locationRange, [100] * len(locationRange),
           color=colorList[len(featureRange)])
  legendList.append('Temporal sequence layer')

  # format
  plt.legend(legendList, bbox_to_anchor=(0., 0.65, 1., .102), loc="right", prop={'size':10})
  plt.xlabel("Size of location pool")
  # plt.xticks(range(0,max(locationRange)+1,10))
  # plt.yticks(range(0,int(accuracy.max())+2,10))
  plt.ylim(-10.0, 110.0)
  plt.ylabel(yaxis)
  plt.title(title)

    # save
  plt.savefig(plotPath)
  plt.close()


def plotPredictionsBySequence(results, objectRange, featureRange, numTrials,
                            key="", title="", yaxis=""):
  """
  Plots the convergence graph: iterations vs number of objects.
  Each curve shows the convergence for a given number of unique features.
  """

  ########################################################################
  #
  # Accumulate all the results per column in a convergence array.
  #
  # predictions[f,s] = how long it took it to converge with f unique features
  # and s sequences on average.
  predictions = numpy.zeros((max(featureRange), max(objectRange) + 1))
  for r in results:
    if r["numFeatures"] in featureRange:
      predictions[r["numFeatures"] - 1, r["numSequences"]] += r[key]

  predictions /= numTrials

  ########################################################################
  #
  # Create the plot.
  plt.figure()
  plotPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "plots", key+"by_sequence.pdf")

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
  plt.xlabel("Number of sequences learned")
  plt.xticks(range(0,max(objectRange)+1,10))
  plt.yticks(range(0,int(predictions.max())+2,10))
  plt.ylabel(yaxis)
  plt.title(title)

    # save
  plt.savefig(plotPath)
  plt.close()


def plotOneInferenceRun(stats,
                       fields,
                       basename,
                       plotDir="plots",
                       experimentID=0):
  """
  Plots individual inference runs.
  """
  if not os.path.exists(plotDir):
    os.makedirs(plotDir)

  plt.figure()
  objectName = stats["object"]

  # plot request stats
  for field in fields:
    fieldKey = field[0] + " C0"
    plt.plot(stats[fieldKey], marker='+', label=field[1])

  # format
  plt.legend(loc="upper right")
  plt.xlabel("Input number")
  plt.xticks(range(stats["numSteps"]))
  plt.ylabel("Number of cells")
  plt.ylim(-5, 100)
  # plt.ylim(plt.ylim()[0] - 5, plt.ylim()[1] + 5)
  plt.title("Activity while inferring a single sequence".format(objectName))

  # save
  relPath = "{}_exp_{}.pdf".format(basename, experimentID)
  path = os.path.join(plotDir, relPath)
  plt.savefig(path)
  plt.close()




if __name__ == "__main__":

  startTime = time.time()
  dirName = os.path.dirname(os.path.realpath(__file__))

  # This is how you run a specific experiment in single process mode. Useful
  # for debugging, profiling, etc.
  if False:
    results = runExperiment(
                  {
                    "numSequences": 5,
                    "seqLength": 10,
                    "numFeatures": 10,
                    "numColumns": 1,
                    "trialNum": 4,
                    "numLocations": 100,
                    "plotInferenceStats": True,  # Outputs detailed graphs
                  }
              )


  # Here we want to check accuracy of the L2/L4 networks in classifying the
  # sequences. This experiment is run using a process pool
  if True:
    # We run 10 trials for each column number and then analyze results
    numTrials = 10
    featureRange = [5, 10, 100]
    seqRange = [50]
    locationRange = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900,
                     1000, 1100, 1200, 1300, 1400, 1500, 1600]
    resultsName = os.path.join(dirName, "sensorimotor_accuracy_results_5_10_100.pkl")

    # Comment this out if you  are re-running analysis on already saved results.
    # Very useful for debugging the plots
    # runExperimentPool(
    #                   numSequences=seqRange,
    #                   numFeatures=featureRange,
    #                   numLocations=locationRange,
    #                   seqLength=10,
    #                   nTrials=numTrials,
    #                   numWorkers=cpu_count() - 1,
    #                   resultsName=resultsName)

    # Analyze results
    with open(resultsName,"rb") as f:
      results = cPickle.load(f)

    plotSensorimotorAccuracy(results, locationRange, featureRange, seqRange,
      title="Relative performance of layers while inferring temporal sequences",
      yaxis="Accuracy (%)")


  # Here we want to see how the number of objects affects convergence for a
  # single column.
  # This experiment is run using a process pool
  if False:
    # We run 10 trials for each column number and then analyze results
    numTrials = 10
    featureRange = [5, 10, 100, 1000]
    seqRange = [2,5,10,20,30,50]
    locationRange = [10, 100, 500, 1000]
    resultsName = os.path.join(dirName, "sequence_convergence_results.pkl")

    # Comment this out if you  are re-running analysis on already saved results.
    # Very useful for debugging the plots
    runExperimentPool(
                      numSequences=seqRange,
                      numFeatures=featureRange,
                      numLocations=locationRange,
                      seqLength=10,
                      nTrials=numTrials,
                      numWorkers=cpu_count() - 1,
                      resultsName=resultsName)

    # Analyze results
    with open(resultsName,"rb") as f:
      results = cPickle.load(f)

    plotConvergenceBySequence(results, seqRange, featureRange, numTrials)

    plotPredictionsBySequence(results, seqRange, featureRange, numTrials,
                              key="averagePredictions",
                              title="Predictions in temporal sequence layer",
                              yaxis="Average number of predicted cells")

    plotPredictionsBySequence(results, seqRange, featureRange, numTrials,
                            key="averagePredictedActive",
                            title="Correct predictions in temporal sequence layer",
                            yaxis="Average number of correctly predicted cells"
                            )

    plotPredictionsBySequence(results, seqRange, featureRange, numTrials,
                              key="averagePredictedActiveL4",
                              title="Correct predictions in sensorimotor layer",
                              yaxis="Average number of correctly predicted cells"
                              )

    plotPredictionsBySequence(results, seqRange, featureRange, numTrials,
                              key="averagePredictionsL4",
                              title="Predictions in sensorimotor layer",
                              yaxis="Average number of predicted cells")


  print "Actual runtime=",time.time() - startTime

