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
This file plots the behavior of L4-L2-TM network as you train it on objects.
"""

import random
import time
import os
from math import ceil
import numpy
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
        columnConvergence = locateConvergencePoint(
          stats[key], minOverlap, maxOverlap)

        # Ensure this column has converged by the last iteration
        # assert(columnConvergence <= len(stats[key]))

        convergencePoint = max(convergencePoint, columnConvergence)

    convergenceSum += ceil(float(convergencePoint)/settlingTime)

    if ceil(float(convergencePoint)/settlingTime) <= inferenceLength:
      numCorrect += 1

  return convergenceSum/len(inferenceStats), numCorrect/len(inferenceStats)


def averageSequenceAccuracy(inferenceStats, minOverlap, maxOverlap):
  """
  inferenceStats contains activity traces while the system visits each object.

  Given the i'th object, inferenceStats[i] contains activity statistics for
  each column for each region for the entire sequence of sensations.

  For each object, decide whether the TM uniquely classified it by checking that
  the number of predictedActive cells are in an acceptable range.
  """
  numCorrect = 0.0
  numStats = 0.0
  prefix = "TM PredictedActive"

  # For each object
  for stats in inferenceStats:

    # Keep running total of how often the number of predictedActive cells are
    # in the range.
    for key in stats.iterkeys():
      if prefix in key:
        for numCells in stats[key]:
          numStats += 1.0
          if numCells in range(minOverlap, maxOverlap+1):
            numCorrect += 1.0

  return numCorrect / numStats


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
  @param settlingTime (int)  Number of iterations we wait to let columns
                             stabilize. Important for multicolumn experiments
                             with lateral connections.
  @param includeRandomLocation (bool) If True, a random location SDR will be
                             generated during inference for each feature.

  The method returns the args dict updated with two additional keys:
    convergencePoint (int)   The average number of iterations it took
                             to converge across all objects
    objects          (pairs) The list of objects we trained on
  """
  numObjects = args.get("numObjects", 10)
  numLocations = args.get("numLocations", 10)
  numFeatures = args.get("numFeatures", 10)
  numColumns = args.get("numColumns", 1)
  networkType = args.get("networkType", "L4L2TMColumn")
  profile = args.get("profile", False)
  noiseLevel = args.get("noiseLevel", None)  # TODO: implement this?
  numPoints = args.get("numPoints", 10)
  trialNum = args.get("trialNum", 42)
  pointRange = args.get("pointRange", 1)
  plotInferenceStats = args.get("plotInferenceStats", True)
  settlingTime = args.get("settlingTime", 3)
  includeRandomLocation = args.get("includeRandomLocation", False)
  inputSize = args.get("inputSize", 512)
  numInputBits = args.get("inputBits", 20)


  # Create the objects
  objects = createObjectMachine(
    machineType="simple",
    numInputBits=numInputBits,
    sensorInputSize=inputSize,
    externalInputSize=1024,
    numCorticalColumns=numColumns,
    numFeatures=numFeatures,
    seed=trialNum
  )

  for p in range(pointRange):
    objects.createRandomObjects(numObjects, numPoints=numPoints+p,
                                      numLocations=numLocations,
                                      numFeatures=numFeatures)

  r = objects.objectConfusion()
  print "Average common pairs=", r[0],
  print ", locations=",r[1],
  print ", features=",r[2]

  # print "Total number of objects created:",len(objects.getObjects())
  # print "Objects are:"
  # for o in objects:
  #   pairs = objects[o]
  #   pairs.sort()
  #   print str(o) + ": " + str(pairs)

  # Setup experiment and train the network. Ensure both TM layers have identical
  # parameters.
  name = "convergence_O%03d_L%03d_F%03d_T%03d" % (
    numObjects, numLocations, numFeatures, trialNum
  )
  exp = L4TMExperiment(
    name=name,
    numCorticalColumns=numColumns,
    networkType = networkType,
    inputSize=inputSize,
    numInputBits=numInputBits,
    externalInputSize=1024,
    numExternalInputBits=numInputBits,
    seed=trialNum,
    L4Overrides={"initialPermanence": 0.41,
                 "activationThreshold": 18,
                 "minThreshold": 18},
  )

  # We want to traverse the features of each object randomly a few times before
  # moving on to the next time. Create the SDRs that we need for this.
  objectsToLearn = objects.provideObjectsToLearn()
  objectTraversals = {}
  for objectId in objectsToLearn:
    objectTraversals[objectId] = objects.randomTraversal(
      objectsToLearn[objectId], settlingTime)

  # Train the network on all the SDRs for all the objects
  exp.learnObjects(objectTraversals)
  if profile:
    exp.printProfile(reset=True)

  # For inference, we will check and plot convergence for each object. For each
  # object, we create a sequence of random sensations for each column.  We will
  # present each sensation for settlingTime time steps to let it settle and
  # ensure it converges.
  for objectId in objects:
    obj = objects[objectId]

    objectSensations = {}
    for c in range(numColumns):
      objectSensations[c] = []

    assert numColumns == 1

    # Create sequence of sensations for this object for one column. The total
    # number of sensations is equal to the number of points on the object. No
    # point should be visited more than once.
    objectCopy = [pair for pair in obj]
    random.shuffle(objectCopy)
    for pair in objectCopy:
        objectSensations[0].append(pair)

    inferConfig = {
      "object": objectId,
      "numSteps": len(objectSensations[0]),
      "pairs": objectSensations,
      "includeRandomLocation": includeRandomLocation,
    }

    inferenceSDRs = objects.provideObjectToInfer(inferConfig)

    exp.infer(inferenceSDRs, objectName=objectId)
    if profile:
      exp.printProfile(reset=True)

    if plotInferenceStats:
      plotOneInferenceRun(
        exp.statistics[objectId],
        fields=[
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
  convergencePoint, sensorimotorAccuracy = averageConvergencePoint(
    infStats,"L2 Representation", 30, 40, settlingTime)

  sequenceAccuracy = averageSequenceAccuracy(infStats, 15, 25)

  predictedActive = numpy.zeros(len(infStats))
  predicted = numpy.zeros(len(infStats))
  predictedActiveL4 = numpy.zeros(len(infStats))
  predictedL4 = numpy.zeros(len(infStats))
  for i,stat in enumerate(infStats):
    predictedActive[i] = float(sum(stat["TM PredictedActive C0"][2:])) / len(stat["TM PredictedActive C0"][2:])
    predicted[i] = float(sum(stat["TM Predicted C0"][2:])) / len(stat["TM Predicted C0"][2:])

    predictedActiveL4[i] = float(sum(stat["L4 PredictedActive C0"])) / len(stat["L4 PredictedActive C0"])
    predictedL4[i] = float(sum(stat["L4 Predicted C0"])) / len(stat["L4 Predicted C0"])

  print "# objects {} # features {} # locations {} # columns {} trial # {} network type {}".format(
    numObjects, numFeatures, numLocations, numColumns, trialNum, networkType)
  print "Average convergence point=",convergencePoint,
  print "Accuracy:", sensorimotorAccuracy
  print "Sequence accuracy:", sequenceAccuracy
  print

  # Return our convergence point as well as all the parameters and objects
  args.update({"objects": objects.getObjects()})
  args.update({"convergencePoint":convergencePoint})
  args.update({"sensorimotorAccuracyPct": sensorimotorAccuracy})
  args.update({"sequenceAccuracyPct": sequenceAccuracy})
  args.update({"averagePredictions": predicted.mean()})
  args.update({"averagePredictedActive": predictedActive.mean()})
  args.update({"averagePredictionsL4": predictedL4.mean()})
  args.update({"averagePredictedActiveL4": predictedActiveL4.mean()})

  # Can't pickle experiment so can't return it for batch multiprocessing runs.
  # However this is very useful for debugging when running in a single thread.
  if plotInferenceStats:
    args.update({"experiment": exp})
  return args


def runExperimentPool(numObjects,
                      numLocations,
                      numFeatures,
                      numWorkers=7,
                      nTrials=1,
                      pointRange=1,
                      numPoints=10,
                      includeRandomLocation=False,
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

  for o in reversed(numObjects):
    for l in numLocations:
      for f in numFeatures:
        for t in range(nTrials):
          args.append(
            {"numObjects": o,
             "numLocations": l,
             "numFeatures": f,
             "trialNum": t,
             "pointRange": pointRange,
             "numPoints": numPoints,
             "plotInferenceStats": False,
             "includeRandomLocation": includeRandomLocation,
             "settlingTime": 3,
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


def plotConvergenceByObject(results, objectRange, featureRange, numTrials):
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
      convergence[r["numFeatures"] - 1, r["numObjects"]] += r["convergencePoint"]

  convergence /= numTrials

  ########################################################################
  #
  # Create the plot. x-axis=
  plt.figure()
  plotPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "plots", "convergence_by_object.pdf")

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
  plt.xlabel("Number of objects in training set")
  plt.xticks(range(0,max(objectRange)+1,10))
  plt.yticks(range(0,int(convergence.max())+2))
  plt.ylabel("Average number of touches")
  plt.title("Number of touches to recognize one object (single column)")

    # save
  plt.savefig(plotPath)
  plt.close()


def plotPredictionsByObject(results, objectRange, featureRange, numTrials,
                            key="", title="", yaxis=""):
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
      predictions[r["numFeatures"] - 1, r["numObjects"]] += r[key]

  predictions /= numTrials

  ########################################################################
  #
  # Create the plot. x-axis=
  plt.figure()
  plotPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "plots", key+"_by_object.pdf")

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
  plt.xlabel("Number of objects in training set")
  plt.xticks(range(0,max(objectRange)+1,10))
  plt.yticks(range(0,int(predictions.max())+2,10))
  plt.ylabel(yaxis)
  plt.title(title)

    # save
  plt.savefig(plotPath)
  plt.close()


def plotSequenceAccuracy(results, featureRange, objectRange,
                         title="", yaxis=""):
  """
  Plot accuracy vs number of features
  """

  ########################################################################
  #
  # Accumulate all the results per column in a convergence array.
  #
  # accuracy[o,f] = accuracy with o objects in training
  # and f unique features.
  accuracy = numpy.zeros((max(objectRange)+1, max(featureRange) + 1))
  totals = numpy.zeros((max(objectRange)+1, max(featureRange) + 1))
  for r in results:
    if r["numFeatures"] in featureRange and r["numObjects"] in objectRange:
      accuracy[r["numObjects"], r["numFeatures"]] += r["sequenceAccuracyPct"]
      totals[r["numObjects"], r["numFeatures"]] += 1

  for o in objectRange:
    for f in featureRange:
      accuracy[o, f] = 100.0 * accuracy[o, f] / totals[o, f]

  ########################################################################
  #
  # Create the plot.
  plt.figure()
  plotPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "plots", "sequenceAccuracy_by_object.pdf")

  # Plot each curve
  legendList = []
  colorList = ['r', 'b', 'g', 'm', 'c', 'k', 'y']

  for i in range(len(featureRange)):
    f = featureRange[i]
    print "features={} objectRange={} accuracy={}".format(
      f,objectRange, accuracy[objectRange, f])
    print "Totals=",totals[objectRange, f]
    legendList.append('Sequence layer, feature pool size: {}'.format(f))
    plt.plot(objectRange, accuracy[objectRange, f], color=colorList[i])

  plt.plot(objectRange, [100] * len(objectRange),
           color=colorList[len(featureRange)])
  legendList.append('Sensorimotor layer')

  # format
  plt.legend(legendList, bbox_to_anchor=(0., 0.6, 1., .102), loc="right", prop={'size':10})
  plt.xlabel("Number of objects")
  # plt.xticks(range(0,max(locationRange)+1,10))
  # plt.yticks(range(0,int(accuracy.max())+2,10))
  plt.ylim(-10.0, 110.0)
  plt.ylabel(yaxis)
  plt.title(title)

    # save
  plt.savefig(plotPath)
  plt.close()


def plotSequenceAccuracyBargraph(results, featureRange, objectRange,
                         title="", yaxis=""):
  """
  Plot accuracy vs number of features
  """

  ########################################################################
  #
  # Accumulate all the results per column in a convergence array.
  #
  # accuracy[o,f] = accuracy with o objects in training
  # and f unique features.
  accuracy = numpy.zeros(max(featureRange) + 1)
  totals = numpy.zeros(max(featureRange) + 1)
  for r in results:
    if r["numFeatures"] in featureRange and r["numObjects"] in objectRange:
      accuracy[r["numFeatures"]] += r["sequenceAccuracyPct"]
      totals[r["numFeatures"]] += 1

  for f in featureRange:
    accuracy[f] = 100.0 * accuracy[f] / totals[f]

  ########################################################################
  #
  # Create the plot.
  plt.figure()
  plotPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "plots", "sequenceAccuracy_by_object_bar.pdf")

  # Plot each curve
  legendList = []
  colorList = ['r', 'b', 'g', 'm', 'c', 'k', 'y']

  ind = numpy.arange(len(featureRange))
  width = 0.35

  for i in range(len(featureRange)):
    f = featureRange[i]
    print "features={} objectRange={} accuracy={}".format(
      f,objectRange, accuracy[f])
    print "Totals=",totals[f]
    plt.bar(i, 100.0, width, color='black')
    plt.bar(i+width, accuracy[f], width, color='white', edgecolor='black')

  legendList.append("Sensorimotor layer")
  legendList.append("Sequence layer")
  plt.legend(legendList, bbox_to_anchor=(0., 0.87, 1.0, .102), loc="right", prop={'size':10})
  plt.xlabel("Number of objects")
  plt.xticks(ind + width / 2, featureRange)
  plt.ylim(0.0, 119.0)
  plt.ylabel(yaxis)
  plt.title(title)
  #
  #   # save
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
  # plt.ylim(plt.ylim()[0] - 5, plt.ylim()[1] + 5)
  plt.ylim(-5, 50)
  plt.title("Activity while inferring a single object".format(objectName))

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
                    "numObjects": 50,
                    "numPoints": 10,
                    "numLocations": 100,
                    "numFeatures": 25,
                    "trialNum": 4,
                    "pointRange": 1,
                    "plotInferenceStats": True,  # Outputs detailed graphs
                    "settlingTime": 3,
                  }
              )


  # Here we want to check accuracy of the TM network in classifying the
  # objects.
  if True:
    # We run 10 trials for each column number and then analyze results
    numTrials = 10
    featureRange = [5, 10, 50]
    objectRange = [2, 5, 10, 20, 30, 40, 50, 70]
    locationRange = [100]
    resultsName = os.path.join(dirName, "sequence_accuracy_results.pkl")

    # Comment this out if you  are re-running analysis on already saved results.
    # Very useful for debugging the plots
    # runExperimentPool(
    #                   numObjects=objectRange,
    #                   numFeatures=featureRange,
    #                   numLocations=locationRange,
    #                   numPoints=10,
    #                   nTrials=numTrials,
    #                   numWorkers=cpu_count() - 1,
    #                   resultsName=resultsName)

    # Analyze results
    with open(resultsName,"rb") as f:
      results = cPickle.load(f)

    plotSequenceAccuracy(results, featureRange, objectRange,
      title="Relative performance of layers during sensorimotor inference",
      yaxis="Accuracy (%)")

    # plotSequenceAccuracyBargraph(results, featureRange, objectRange,
    #                      title="Performance while inferring objects",
    #                      yaxis="Accuracy (%)")

  # Here we want to see how the number of objects affects convergence for a
  # single column.
  # This experiment is run using a process pool
  if False:
    # We run 10 trials for each column number and then analyze results
    numTrials = 10
    featureRange = [5, 10, 100, 1000]
    objectRange = [2, 5, 10, 20, 30, 50]
    locationRange = [10, 100, 500, 1000]
    resultsName = os.path.join(dirName, "object_convergence_results.pkl")

    # Comment this out if you are re-running analysis on already saved results.
    # Very useful for debugging the plots
    runExperimentPool(
                      numObjects=objectRange,
                      numLocations=locationRange,
                      numFeatures=featureRange,
                      numPoints=10,
                      nTrials=numTrials,
                      numWorkers=cpu_count() - 1,
                      resultsName=resultsName)

    # Analyze results
    with open(resultsName,"rb") as f:
      results = cPickle.load(f)

    plotConvergenceByObject(results, objectRange, featureRange, numTrials)

    plotPredictionsByObject(results, objectRange, featureRange, numTrials,
                            key="averagePredictions",
                            title="Predictions in temporal sequence layer",
                            yaxis="Average number of predicted cells")

    plotPredictionsByObject(results, objectRange, featureRange, numTrials,
                            key="averagePredictedActive",
                            title="Correct predictions in temporal sequence layer",
                            yaxis="Average number of correctly predicted cells"
                            )


    plotPredictionsByObject(results, objectRange, featureRange, numTrials,
                            key="averagePredictedActiveL4",
                            title="Correct predictions in sensorimotor layer",
                            yaxis="Average number of correctly predicted cells"
                            )
    plotPredictionsByObject(results, objectRange, featureRange, numTrials,
                            key="averagePredictionsL4",
                            title="Predictions in sensorimotor layer",
                            yaxis="Average number of predicted cells")

  print "Actual runtime=",time.time() - startTime
