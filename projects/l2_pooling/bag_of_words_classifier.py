# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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
This file is used to run Thing experiments using simulated sensations with
a simple logistic encoder, with/without location signal


Example usage

Train a bag of words classifier with/without location
python bag_of_words_classifier.py --location 1

python bag_of_words_classifier.py  --location 0
"""

import cPickle
from optparse import OptionParser
import os
import random
import numpy
import numpy as np
import  matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from htmresearch.frameworks.layers.object_machine_factory import (
  createObjectMachine
)
from multi_column_convergence import runExperimentPool

plt.ion()
plt.close('all')
def _getArgs():
  parser = OptionParser(usage="Train BoW classifier on Thing data")

  parser.add_option("-l",
                    "--location",
                    type=int,
                    default=1,
                    dest="useLocation",
                    help="Whether to use location signal")


  (options, remainder) = parser.parse_args()
  return options, remainder


def findWordInVocabulary(input, wordList):
  findWord = None
  for j in range(wordList.shape[0]):
    numBitDiff = np.sum(np.abs(input - wordList[j, :]))
    if numBitDiff == 0:
      findWord = j
  return findWord



def bowClassifierPredict(input, bowVectors, distance="overlap"):
  numClasses = bowVectors.shape[0]
  output = np.zeros((numClasses,))

  # normalize input
  if distance == "L1":
    for i in range(numClasses):
      output[i] = np.sum(np.abs(input - bowVectors[i, :]))

  elif distance == "dotProduct":
    # normalize input
    input = input / np.linalg.norm(input)
    for i in range(numClasses):
      bowVectors[i, :] = bowVectors[i, :]/np.linalg.norm(bowVectors[i, :])
      output[i] = np.dot(input, bowVectors[i, :])
    output = 1 - output
  elif distance == "overlap":
    for i in range(numClasses):
      output[i] = -np.sum(np.minimum(input, bowVectors[i, :]))
  return output



def locateConvergencePoint(stats):
  """
  Walk backwards through stats until you locate the first point that diverges
  from target overlap values.  We need this to handle cases where it might get
  to target values, diverge, and then get back again.  We want the last
  convergence point.
  """
  n = len(stats)
  for i in range(n):
    if stats[n-i-1] < 1:
      return n-i

  # Never differs - converged in one iteration
  return 1



def run_bag_of_words_classifier(args):
  numObjects = args.get("numObjects", 10)
  numLocations = args.get("numLocations", 10)
  numFeatures = args.get("numFeatures", 10)
  numPoints = args.get("numPoints", 10)
  trialNum = args.get("trialNum", 42)
  pointRange = args.get("pointRange", 1)
  useLocation = args.get("useLocation", 1)
  numColumns = args.get("numColumns", 1)
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
  random.seed(trialNum)

  for p in range(pointRange):
    objects.createRandomObjects(numObjects, numPoints=numPoints + p,
                                numLocations=numLocations,
                                numFeatures=numFeatures)

  objects = objects.provideObjectsToLearn()
  objectNames = objects.keys()
  numObjs = len(objectNames)
  featureWidth = 150
  locationWidth = 2400

  # compute the number of sensations across all objects
  numInputVectors = 0
  for i in range(numObjs):
    numInputVectors += len(objects[objectNames[i]])

  if useLocation:
    inputWidth = featureWidth + locationWidth
  else:
    inputWidth = featureWidth

  # create "training" dataset
  data = np.zeros((numInputVectors, inputWidth))
  label = np.zeros((numInputVectors, numObjs))
  k = 0
  for i in range(numObjs):
    # print "converting object {} ...".format(i)
    numSenses = len(objects[objectNames[i]])
    for j in range(numSenses):
      activeBitsFeature = np.array(list(objects[objectNames[i]][j][0][1]))
      data[k, activeBitsFeature] = 1
      if useLocation:
        activeBitsLocation = np.array(list(objects[objectNames[i]][j][0][0]))
        data[k, featureWidth + activeBitsLocation] = 1
      label[k, i] = 1
      k += 1

  # enumerate number of distinct "words"
  wordList = np.zeros((0, inputWidth), dtype='int32')
  featureList = np.zeros((data.shape[0],))
  for i in range(data.shape[0]):
    findWord = False
    for j in range(wordList.shape[0]):
      index = findWordInVocabulary(data[i, :], wordList)
      if index is not None:
        featureList[i] = index
        findWord = True
        break

    if findWord is False:
      newWord = np.zeros((1, inputWidth), dtype='int32')
      newWord[0, :] = data[i, :]
      wordList = np.concatenate((wordList, newWord))
      featureList[i] = wordList.shape[0] - 1

  numWords = wordList.shape[0]
  # wordList = wordList[np.random.permutation(np.arange(numWords)), :]
  print "object # {} feature #  {} location # {} distinct words # {} numColumns {}".format(
    numObjects, numFeatures, numLocations, numWords, numColumns)

  # convert objects to BOW representations
  bowVectors = np.zeros((numObjs, numWords), dtype=np.int32)
  k = 0
  for i in range(numObjs):
    numSenses = len(objects[objectNames[i]])
    for j in range(numSenses):
      index = findWordInVocabulary(data[k, :], wordList)
      bowVectors[i, index] += 1
      k += 1

  # plt.figure()
  # plt.imshow(np.transpose(bowVectors))
  # plt.xlabel('Object #')
  # plt.ylabel('Word #')
  # plt.title("BoW representations")

  objects = []
  for i in range(numObjs):
    senseList = []
    senses = np.where(bowVectors[i, :])[0]
    for sense in senses.tolist():
      # print bowVectors[i, sense]
      for _ in range(bowVectors[i, sense]):
        senseList.append(sense)
    random.shuffle(senseList)
    objects.append(senseList)

  # plot accuracy as a function of number of sensations
  accuracyList = []
  classificationOutcome = np.zeros((numObjs, 11))
  for maxSenses in range(1, 11):
    bowVectorsTest = np.zeros((numObjs, numWords), dtype=np.int32)
    offset = 0
    for i in range(numObjs):
      numSenses = min(len(objects[objectNames[i]]), maxSenses)
      #
      # # sensations for object i
      # senses = np.where(bowVectors[i, :])[0]
      senses = objects[i]
      # if i==42:
      #   print senses
      for c in range(numColumns):
        for j in range(numSenses):
          # index = np.random.choice(senses)
          index = senses[j]
          # index = findWordInVocabulary(data[offset + j, :], wordList)
          bowVectorsTest[i, index] += 1
        offset += len(objects[objectNames[i]])

    numCorrect = 0
    for i in range(numObjs):
      output = bowClassifierPredict(bowVectorsTest[i, :], bowVectors)
      # if i==42:
      #   print
      #   print bowVectorsTest[i, :]
      #   print bowVectors[i, :]
      #   print "maxSenses=", maxSenses
      #   print -output
      #   print -output[42]
      predictLabel = np.argmin(output)
      outcome = predictLabel == i
      numCorrect += outcome
      classificationOutcome[i, maxSenses] = outcome
    accuracy = float(numCorrect) / numObjs
    accuracyList.append(accuracy)
    # print "maxSenses {} accuracy {}".format(maxSenses, accuracy)

  convergencePoint = np.zeros((numObjs, ))
  for i in range(numObjs):
    # if i==42:
    # print "obj ", i, "result: ", classificationOutcome[i, :]
    # print locateConvergencePoint(classificationOutcome[i, :])
    if np.max(classificationOutcome[i, :])>0:
      convergencePoint[i] = locateConvergencePoint(classificationOutcome[i, :])
      # convergencePoint[i] = np.where(classificationOutcome[i, :] == 1)[0][0]
    else:
      convergencePoint[i] = 11

  args.update({"accuracy": accuracyList})
  args.update({"numTouches": range(1, 11)})
  args.update({"convergencePoint": np.mean(convergencePoint)})
  args.update({"classificationOutcome": classificationOutcome})

  return args


def plotConvergenceByObject(results, objectRange, featureRange, numTrials,
                            linestyle='-'):
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

  # Plot each curve
  legendList = []
  colorList = ['r', 'b', 'g', 'm', 'c', 'k', 'y']

  for i in range(len(featureRange)):
    f = featureRange[i]
    print "features={} objectRange={} convergence={}".format(
      f,objectRange, convergence[f-1,objectRange])
    legendList.append('Unique features={}'.format(f))
    plt.plot(objectRange, convergence[f-1, objectRange],
             color=colorList[i], linestyle=linestyle)

  # format
  plt.legend(legendList, loc="lower right", prop={'size':10})
  plt.xlabel("Number of objects in training set")
  plt.xticks(range(0,max(objectRange)+1,10))
  plt.yticks(range(0,int(convergence.max())+2))
  plt.ylabel("Average number of touches")
  plt.title("Number of touches to recognize one object (single column)")




def plotConvergenceByColumn(results, columnRange, featureRange, numTrials, lineStype='-'):
  """
  Plots the convergence graph: iterations vs number of columns.
  Each curve shows the convergence for a given number of unique features.
  """
  ########################################################################
  #
  # Accumulate all the results per column in a convergence array.
  #
  # Convergence[f,c] = how long it took it to  converge with f unique features
  # and c columns.
  convergence = numpy.zeros((max(featureRange), max(columnRange) + 1))
  for r in results:
    if r["numColumns"] > max(columnRange):
      continue
    convergence[r["numFeatures"] - 1,
                r["numColumns"]] += r["convergencePoint"]
  convergence /= numTrials
  # For each column, print convergence as fct of number of unique features
  for c in range(1, max(columnRange) + 1):
    print c, convergence[:, c]
  # Print everything anyway for debugging
  print "Average convergence array=", convergence
  ########################################################################
  #
  # Create the plot. x-axis=
  plotPath = os.path.join("plots", "bow_convergence_by_column.pdf")
  # Plot each curve
  legendList = []
  colorList = ['r', 'b', 'g', 'm', 'c', 'k', 'y']
  for i in range(len(featureRange)):
    f = featureRange[i]
    print columnRange
    print convergence[f-1,columnRange]
    legendList.append('Unique features={}'.format(f))
    plt.plot(columnRange, convergence[f-1,columnRange],
             color=colorList[i], linestyle=lineStype)
  # format
  plt.legend(legendList, loc="upper right")
  plt.xlabel("Number of columns")
  plt.xticks(columnRange)
  plt.ylim([0, 4])
  plt.yticks(range(0,int(convergence.max())+1))
  plt.ylabel("Average number of touches")
  plt.title("Number of touches to recognize one object (multiple columns)")
    # save
  # plt.savefig(plotPath)
  # plt.close()



def run_bow_experiment_single_column(options):
  # Create the objects
  featureRange = [5, 10, 20, 30]
  pointRange = 1
  objectRange = [2, 10, 20, 30, 40, 50, 60, 80, 100]
  numLocations = [10]
  numPoints = 10
  numTrials = 10
  numColumns = [1]
  useLocation = options.useLocation
  args = []
  for c in reversed(numColumns):
    for o in reversed(objectRange):
      for l in numLocations:
        for f in featureRange:
          for t in range(numTrials):
            args.append(
              {"numObjects": o,
               "numLocations": l,
               "numFeatures": f,
               "numColumns": c,
               "trialNum": t,
               "pointRange": pointRange,
               "numPoints": numPoints,
               "useLocation": useLocation
               }

          )

  numWorkers = cpu_count()
  print "{} experiments to run, {} workers".format(len(args), numWorkers)
  print "useLocation: ", useLocation

  pool = Pool(processes=numWorkers)
  result = pool.map(run_bag_of_words_classifier, args)

  resultsName = "bag_of_words_useLocation_{}.pkl".format(useLocation)
  # Pickle results for later use
  with open(resultsName, "wb") as f:
    cPickle.dump(result, f)


  plt.figure()
  with open("object_convergence_results.pkl", "rb") as f:
    results = cPickle.load(f)
  plotConvergenceByObject(results, objectRange, featureRange, linestyle='-')

  with open(resultsName, "rb") as f:
    resultsBOW = cPickle.load(f)
  plotConvergenceByObject(resultsBOW, objectRange, featureRange, numTrials,
                          linestyle='--')
  plotPath = os.path.join("plots",
                          "convergence_by_object_random_location_bow.pdf")
  plt.savefig(plotPath)


def run_bow_experiment_multiple_columns():
  # Create the objects
  featureRange = [5, 10, 30]
  pointRange = 1
  objectRange = [100]
  numLocations = [10]
  numPoints = 10
  numTrials = 10
  columnRange = [1, 2, 3, 4, 5, 6, 7, 8]
  useLocation = 1
  args = []
  for c in reversed(columnRange):
    for o in reversed(objectRange):
      for l in numLocations:
        for f in featureRange:
          for t in range(numTrials):
            args.append(
              {"numObjects": o,
               "numLocations": l,
               "numFeatures": f,
               "numColumns": c,
               "trialNum": t,
               "pointRange": pointRange,
               "numPoints": numPoints,
               "useLocation": useLocation
               }
            )

  numWorkers = cpu_count()
  pool = Pool(processes=numWorkers)
  result = pool.map(run_bag_of_words_classifier, args)

  resultsName = "bag_of_words_multi_column_useLocation_{}.pkl".format(useLocation)
  # Pickle results for later use
  with open(resultsName, "wb") as f:
    cPickle.dump(result, f)

  columnRange = range(1, 9)
  featureRange = [5, 10, 30]
  objectRange = [100]
  networkType = ["MultipleL4L2Columns"]
  numTrials = 10

  runExperimentPool(
    numObjects=objectRange,
    numLocations=[10],
    numFeatures=featureRange,
    numColumns=columnRange,
    networkType=networkType,
    numPoints=10,
    nTrials=numTrials,
    numWorkers=cpu_count(),
    resultsName="column_convergence_results.pkl")

  with open("column_convergence_results.pkl", "rb") as f:
    results = cPickle.load(f)

  resultsName = "bag_of_words_multi_column_useLocation_{}.pkl".format(
    useLocation)
  with open(resultsName, "rb") as f:
    resultsBOW = cPickle.load(f)

  plt.figure()
  plotConvergenceByColumn(results, columnRange, featureRange, numTrials)
  plotConvergenceByColumn(resultsBOW, columnRange, featureRange, numTrials,
                          "--")
  plt.savefig('plots/ideal_observer_multiple_column.pdf')


def run_ideal_model_comparison():
  pointRange = 1
  numTrials = 4
  useLocation = options.useLocation
  args = []

  for t in range(numTrials):
    for useLocation in [0, 1]:
      args.append(
        {"numObjects": 100,
         "numLocations": 10,
         "numFeatures": 10,
         "numColumns": 1,
         "trialNum": t,
         "pointRange": pointRange,
         "numPoints": 10,
         "useLocation": useLocation
         }
      )

  numWorkers = cpu_count()

  print "{} experiments to run, {} workers".format(len(args), numWorkers)
  print "useLocation: ", useLocation

  # run_bag_of_words_classifier(args[0])
  pool = Pool(processes=numWorkers)
  result = pool.map(run_bag_of_words_classifier, args)
  resultsName = "ideal_mode_result.pkl"
  # Pickle results for later use
  with open(resultsName, "wb") as f:
    cPickle.dump(result, f)

  # run sensorimotor network
  numTrials = 10
  columnRange = [1]
  objectRange = [100]
  numAmbiguousLocationsRange = [0]
  runExperimentPool(
    numObjects=objectRange,
    numLocations=[10],
    numFeatures=[10],
    numColumns=columnRange,
    numPoints=10,
    nTrials=numTrials,
    numWorkers=cpu_count(),
    ambiguousLocationsRange=numAmbiguousLocationsRange,
    resultsName="single_column_convergence_results.pkl")

  # plot accuracy across sensations
  accuracyIdeal = 0
  accuracyBOF = 0
  for r in result:
    if r["useLocation"]:
      accuracyIdeal += np.array(r['accuracy'])
    else:
      accuracyBOF += np.array(r['accuracy'])

  accuracyIdeal /= len(result) / 2
  accuracyBOF /= len(result) / 2

  numTouches = len(accuracyIdeal)
  with open("single_column_convergence_results.pkl", "rb") as f:
    resultsModel = cPickle.load(f)
  accuracyModel = 0
  for r in resultsModel:
    accuracyModel += np.array(r['accuracy'])
  accuracyModel /= len(resultsModel)

  plt.figure()
  plt.plot(np.arange(numTouches)+1, accuracyIdeal, '-o', label='Ideal Observer')
  plt.plot(np.arange(numTouches) + 1, accuracyBOF, '-s', label='Bag Of Features')
  plt.plot(np.arange(numTouches)+1, accuracyModel, '-^', label='Model')
  plt.xlabel("Number of sensations")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.savefig('plots/compare_model_with_ideal_observer.pdf')


if __name__ == "__main__":
  (options, args) = _getArgs()

  # run_bow_experiment_single_column(options)

  run_bow_experiment_multiple_columns()

  # run_ideal_model_comparison()

