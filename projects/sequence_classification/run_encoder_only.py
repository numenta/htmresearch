# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
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
Run sequence classification experiment with simple encoder model

1. Encode each element with RDSE encoder
2. Calculate prediction using kNN based on average overalap distance
3. Search for the optimal encoder resolution
"""

import pickle
import time
import matplotlib.pyplot as plt
import multiprocessing

from util_functions import *

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
plt.ion()


def runEncoderOverDataset(encoder, dataset):
  activeColumnsData = []

  for i in range(dataset.shape[0]):
    activeColumnsTrace = []

    for element in dataset[i, :]:
      encoderOutput = encoder.encode(element)
      activeColumns = set(np.where(encoderOutput > 0)[0])
      activeColumnsTrace.append(activeColumns)

    activeColumnsData.append(activeColumnsTrace)
  return activeColumnsData



def calcualteEncoderModelWorker(taskQueue, resultQueue, *args):
  while True:
    nextTask = taskQueue.get()
    print "Next task is : ", nextTask
    if nextTask is None:
      break
    nBuckets = nextTask["nBuckets"]
    accuracyColumnOnly = calculateEncoderModelAccuracy(nBuckets, *args)
    resultQueue.put({nBuckets: accuracyColumnOnly})
    print "Column Only model, Resolution: {} Accuracy: {}".format(
      nBuckets, accuracyColumnOnly)
  return



def calculateEncoderModelAccuracy(nBuckets, numCols, w, trainData, trainLabel):
  maxValue = np.max(trainData)
  minValue = np.min(trainData)

  resolution = (maxValue - minValue) / nBuckets
  encoder = RandomDistributedScalarEncoder(resolution, w=w, n=numCols)

  activeColumnsTrain = runEncoderOverDataset(encoder, trainData)
  distMatColumnTrain = calculateDistanceMatTrain(activeColumnsTrain)
  meanAccuracy, outcomeColumn = calculateAccuracy(distMatColumnTrain,
                                                  trainLabel, trainLabel)
  accuracyColumnOnly = np.mean(outcomeColumn)
  return accuracyColumnOnly



def searchForOptimalEncoderResolution(nBucketList, trainData, trainLabel, numCols, w):

  numCPU = multiprocessing.cpu_count()
  numWorker = numCPU
  # Establish communication queues
  taskQueue = multiprocessing.JoinableQueue()
  resultQueue = multiprocessing.Queue()

  for nBuckets in nBucketList:
    taskQueue.put({"nBuckets": nBuckets})
  for _ in range(numWorker):
    taskQueue.put(None)
  jobs = []

  for i in range(numWorker):
    print "Start process ", i
    p = multiprocessing.Process(target=calcualteEncoderModelWorker,
                                args=(taskQueue, resultQueue, numCols, w, trainData, trainLabel))
    jobs.append(p)
    p.daemon = True
    p.start()

  while not taskQueue.empty():
    time.sleep(0.1)
  accuracyVsResolution = np.zeros((len(nBucketList,)))
  while not resultQueue.empty():
    exptResult = resultQueue.get()
    nBuckets = exptResult.keys()[0]
    accuracyVsResolution[nBucketList.index(nBuckets)] = exptResult[nBuckets]

  return accuracyVsResolution



if __name__ == "__main__":
  # datasetName = "SyntheticData"
  # dataSetList = listDataSets(datasetName)

  datasetName = 'UCR_TS_Archive_2015'
  dataSetList = listDataSets(datasetName)
  # dataSetList = ["synthetic_control"]


  for dataName in dataSetList:
    trainData, trainLabel, testData, testLabel = loadDataset(dataName, datasetName)
    numTest = len(testLabel)
    numTrain = len(trainLabel)
    sequenceLength = len(trainData[0])
    classList = np.unique(trainLabel)

    if max(numTrain, numTest) * sequenceLength > 600 * 600:
      print "skip this large dataset for now"
      continue

    print
    print "Processing {}".format(dataName)
    print "Train Sample # {}, Test Sample # {}".format(numTrain, numTest)
    print "Sequence Length {} Class # {}".format(sequenceLength, len(classList))


    try:
      searchResolution = pickle.load(
        open('results/optimalEncoderResolution/{}'.format(dataName), 'r'))
      continue
    except:
      print "Search encoder parameters for this dataset"

    EuclideanDistanceMat = calculateEuclideanDistanceMat(testData, trainData)

    outcomeEuclidean = []
    for i in range(testData.shape[0]):
      predictedClass = one_nearest_neighbor(trainData, trainLabel, testData[i,:])
      correct = 1 if predictedClass == testLabel[i] else 0
      outcomeEuclidean.append(correct)
      # print "{} out of {} done outcome: {}".format(i, testData.shape[0], correct)

    print
    print "Euclidean model accuracy: {}".format(np.mean(outcomeEuclidean))
    print
    accuracyEuclideanDist = np.mean(outcomeEuclidean)

    # # Use SDR overlap instead of Euclidean distance
    print "Running Encoder model"
    from nupic.encoders.random_distributed_scalar import RandomDistributedScalarEncoder
    maxValue = np.max(trainData)
    minValue = np.min(trainData)
    numCols = 2048
    w = 41

    try:
      searchResolution = pickle.load(
        open('results/optimalEncoderResolution/{}'.format(dataName), 'r'))
      optimalResolution = searchResolution['optimalResolution']
    except:
      nBucketList = range(20, 200, 10)
      accuracyVsResolution = searchForOptimalEncoderResolution(
        nBucketList, trainData, trainLabel, numCols, w)
      optNumBucket = nBucketList[np.argmax(np.array(accuracyVsResolution))]
      optimalResolution = (maxValue - minValue)/optNumBucket
      searchResolution = {
        'nBucketList': nBucketList,
        'accuracyVsResolution': accuracyVsResolution,
        'optimalResolution': optimalResolution
      }
      # save optimal resolution for future use
      outputFile = open('results/optimalEncoderResolution/{}'.format(dataName), 'w')
      pickle.dump(searchResolution, outputFile)
      outputFile.close()

    print "optimal bucket # {}".format((maxValue - minValue)/optimalResolution)

    encoder = RandomDistributedScalarEncoder(optimalResolution, w=w, n=numCols)
    print "encoding train data ..."
    activeColumnsTrain = runEncoderOverDataset(encoder, trainData)
    print "encoding test data ..."
    activeColumnsTest = runEncoderOverDataset(encoder, testData)
    print "calculate column distance matrix ..."
    distMatColumnTest = calculateDistanceMat(activeColumnsTest, activeColumnsTrain)
    meanAccuracy, outcomeColumn = calculateAccuracy(distMatColumnTest, trainLabel, testLabel)

    accuracyColumnOnly = np.mean(outcomeColumn)
    print
    print "Column Only model, Accuracy: {}".format(accuracyColumnOnly)

    expResults = {'accuracyEuclideanDist': accuracyEuclideanDist,
                  'accuracyColumnOnly': accuracyColumnOnly,
                  'EuclideanDistanceMat': EuclideanDistanceMat,
                  'distMatColumnTest': distMatColumnTest}
    outputFile = open('results/modelPerformance/{}_columnOnly'.format(dataName), 'w')
    pickle.dump(expResults, outputFile)
    outputFile.close()
