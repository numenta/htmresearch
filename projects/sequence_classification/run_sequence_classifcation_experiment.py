# ----------------------------------------------------------------------
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
import pickle
import time
import matplotlib.pyplot as plt
import multiprocessing
from util_functions import *
from nupic.encoders.random_distributed_scalar import RandomDistributedScalarEncoder

plt.ion()

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams.update({'figure.autolayout': True})


def rdse_encoder_nearest_neighbor(trainData, trainLabel, unknownSequence, encoder):

  overlapSum = np.zeros((trainData.shape[0],))
  for i in range(trainData.shape[0]):
    overlapI = np.zeros((len(unknownSequence),))
    for t in range(len(unknownSequence)):
      overlapI[t] = np.sum(np.logical_and(encoder.encode(unknownSequence[t]),
                                          encoder.encode(trainData[i, t])))
    overlapSum[i] = np.sum(overlapI)

  predictedClass = trainLabel[np.argmax(overlapSum)]
  return predictedClass



def constructDistanceMat(distMatColumn, distMatCell, trainLabel, wOpt, bOpt):
  numTest, numTrain = distMatColumn.shape
  classList = np.unique(trainLabel).tolist()
  distanceMat = np.zeros((numTest, numTrain))
  for classI in classList:
    classIidx = np.where(trainLabel == classI)[0]
    distanceMat[:, classIidx] = \
      (1 - wOpt[classI]) * distMatColumn[:, classIidx] + \
      wOpt[classI] * distMatCell[:, classIidx] + bOpt[classI]

  return distanceMat



def runTMOnSequence(tm, activeColumns, unionLength=1):
  numCells = tm.getCellsPerColumn() * tm.getColumnDimensions()[0]

  activeCellsTrace = []
  predictiveCellsTrace = []
  predictedActiveCellsTrace = []
  activeColumnTrace = []

  activationFrequency = np.zeros((numCells,))
  predictedActiveFrequency = np.zeros((numCells,))

  unionStepInBatch = 0
  unionBatchIdx = 0
  unionCells = set()
  unionCols = set()

  tm.reset()
  for t in range(len(activeColumns)):
    tm.compute(activeColumns[t], learn=False)
    activeCellsTrace.append(set(tm.getActiveCells()))
    predictiveCellsTrace.append(set(tm.getPredictiveCells()))
    if t == 0:
      predictedActiveCells = set()
    else:
      predictedActiveCells = activeCellsTrace[t].intersection(
        predictiveCellsTrace[t - 1])

    activationFrequency[tm.getActiveCells()] += 1
    predictedActiveFrequency[list(predictedActiveCells)] += 1

    unionCells = unionCells.union(predictedActiveCells)
    unionCols = unionCols.union(activeColumns[t])

    unionStepInBatch += 1
    if unionStepInBatch == unionLength:
      predictedActiveCellsTrace.append(unionCells)
      activeColumnTrace.append(unionCols)
      unionStepInBatch = 0
      unionBatchIdx += 1
      unionCells = set()
      unionCols = set()

  if unionStepInBatch > 0:
    predictedActiveCellsTrace.append(unionCells)
    activeColumnTrace.append(unionCols)

  activationFrequency = activationFrequency / np.sum(activationFrequency)
  predictedActiveFrequency = predictedActiveFrequency / np.sum(
    predictedActiveFrequency)
  return (activeColumnTrace,
          predictedActiveCellsTrace,
          activationFrequency,
          predictedActiveFrequency)



def runTMOverDatasetFast(tm, activeColumns, unionLength=1):
  """
  Run encoder -> tm network over dataset, save activeColumn and activeCells
  traces
  :param tm:
  :param encoder:
  :param dataset:
  :return:
  """
  numSequence = len(activeColumns)
  predictedActiveCellsUnionTrace = []
  activationFrequencyTrace = []
  predictedActiveFrequencyTrace = []
  activeColumnUnionTrace = []

  for i in range(numSequence):
    (activeColumnTrace,
     predictedActiveCellsTrace,
     activationFrequency,
     predictedActiveFrequency) = runTMOnSequence(tm, activeColumns[i], unionLength)

    predictedActiveCellsUnionTrace.append(predictedActiveCellsTrace)
    activeColumnUnionTrace.append(activeColumnTrace)
    activationFrequencyTrace.append(activationFrequency)
    predictedActiveFrequencyTrace.append(predictedActiveFrequency)
    # print "{} out of {} done ".format(i, numSequence)

  return (activeColumnUnionTrace,
          predictedActiveCellsUnionTrace,
          activationFrequencyTrace,
          predictedActiveFrequencyTrace)



def runEncoderOverDataset(encoder, dataset):
  activeColumnsData = []

  for i in range(dataset.shape[0]):
    activeColumnsTrace = []

    for element in dataset[i, :]:
      encoderOutput = encoder.encode(element)
      activeColumns = set(np.where(encoderOutput > 0)[0])
      activeColumnsTrace.append(activeColumns)

    activeColumnsData.append(activeColumnsTrace)
    # print "{} out of {} done ".format(i, dataset.shape[0])
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

  # Establish communication queues
  taskQueue = multiprocessing.JoinableQueue()
  resultQueue = multiprocessing.Queue()

  for nBuckets in nBucketList:
    taskQueue.put({"nBuckets": nBuckets})
  for _ in range(numCPU):
    taskQueue.put(None)
  jobs = []
  for i in range(numCPU):
    print "Start process ", i
    p = multiprocessing.Process(target=calcualteEncoderModelWorker,
                                args=(taskQueue, resultQueue, numCols, w, trainData, trainLabel))
    jobs.append(p)
    p.daemon = True
    p.start()
    # p.join()
  # taskQueue.join()
  while not taskQueue.empty():
    time.sleep(0.1)
  accuracyVsResolution = np.zeros((len(nBucketList,)))
  while not resultQueue.empty():
    exptResult = resultQueue.get()
    nBuckets = exptResult.keys()[0]
    accuracyVsResolution[nBucketList.index(nBuckets)] = exptResult[nBuckets]

  return accuracyVsResolution



if __name__ == "__main__":
  plt.close('all')
  datasetName = "SyntheticData"
  dataSetList = listDataSets(datasetName)

  # datasetName = 'UCR_TS_Archive_2015'
  # dataSetList = listDataSets(datasetName)
  # dataSetList = ["synthetic_control"]

  skipTMmodel = False

  for dataName in dataSetList:
    trainData, trainLabel, testData, testLabel = loadDataset(
      dataName, datasetName)
    numTest = len(testLabel)
    numTrain = len(trainLabel)
    sequenceLength = len(trainData[0])
    classList = np.unique(trainLabel).tolist()
    numClass = len(classList)

    print "Processing {}".format(dataName)
    print "Train Sample # {}, Test Sample # {}".format(numTrain, numTest)
    print "Sequence Length {} Class # {}".format(sequenceLength, len(classList))
    EuclideanDistanceMat = calculateEuclideanDistanceMat(testData, trainData)
    outcomeEuclidean = calculateEuclideanModelAccuracy(trainData, trainLabel,
                                                       testData, testLabel)
    accuracyEuclideanDist = np.mean(outcomeEuclidean)
    print
    print "Euclidean model accuracy: {}".format(accuracyEuclideanDist)
    print


    # # Use SDR overlap instead of Euclidean distance
    print "Running Encoder model"
    maxValue = np.max(trainData)
    minValue = np.min(trainData)
    numCols = 2048
    w = 41

    try:
      searchResolution = pickle.load(
        open('results/optimalEncoderResolution/{}'.format(dataName), 'r'))
      nBucketList = searchResolution['nBucketList']
      accuracyVsResolution = searchResolution['accuracyVsResolution']
      optNumBucket = nBucketList[smoothArgMax(np.array(accuracyVsResolution))]
      optimalResolution = (maxValue - minValue)/optNumBucket
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
    testAccuracyColumnOnly, outcomeColumn = calculateAccuracy(
      distMatColumnTest, trainLabel, testLabel)

    print
    print "Column Only model, Accuracy: {}".format(testAccuracyColumnOnly)

    expResults = {'accuracyEuclideanDist': accuracyEuclideanDist,
                  'accuracyColumnOnly': testAccuracyColumnOnly,
                  'EuclideanDistanceMat': EuclideanDistanceMat,
                  'distMatColumnTest': distMatColumnTest}
    outputFile = open('results/modelPerformance/{}_columnOnly'.format(dataName), 'w')
    pickle.dump(expResults, outputFile)
    outputFile.close()

    if skipTMmodel:
      continue

    # Train TM
    from nupic.bindings.algorithms import TemporalMemory as TemporalMemoryCPP
    tm = TemporalMemoryCPP(columnDimensions=(numCols, ),
                           cellsPerColumn=32,
                           permanenceIncrement=0.1,
                           permanenceDecrement=0.1,
                           predictedSegmentDecrement=0.01,
                           minThreshold=10,
                           activationThreshold=15,
                           maxNewSynapseCount=20)

    print
    print "Training TM on sequences ... "
    numRepeatsBatch = 1
    numRptsPerSequence = 1

    np.random.seed(10)
    for rpt in xrange(numRepeatsBatch):
      # randomize the order of training sequences
      randomIdx = np.random.permutation(range(numTrain))
      for i in range(numTrain):
        for _ in xrange(numRptsPerSequence):
          for t in range(sequenceLength):
            tm.compute(activeColumnsTrain[randomIdx[i]][t], learn=True)
          tm.reset()
        print "Rpt: {}, {} out of {} done ".format(rpt, i, trainData.shape[0])

    # run TM over training data
    unionLength = 20
    print "Running TM on Training Data with union window {}".format(unionLength)
    (activeColTrain,
     activeCellsTrain,
     activeFreqTrain,
     predActiveFreqTrain) = runTMOverDatasetFast(tm, activeColumnsTrain, unionLength)

    # construct two distance matrices using training data
    distMatColumnTrain = calculateDistanceMat(activeColTrain, activeColTrain)
    distMatCellTrain = calculateDistanceMat(activeCellsTrain, activeCellsTrain)
    distMatActiveFreqTrain = calculateDistanceMat(activeFreqTrain, activeFreqTrain)
    distMatPredActiveFreqTrain = calculateDistanceMat(predActiveFreqTrain, predActiveFreqTrain)

    maxColumnOverlap = np.max(distMatColumnTrain)
    maxCellOverlap = np.max(distMatCellTrain)
    distMatColumnTrain /= maxColumnOverlap
    distMatCellTrain /= maxCellOverlap
    # set diagonal line to zeros
    for i in range(trainData.shape[0]):
      distMatColumnTrain[i, i] = 0
      distMatCellTrain[i, i] = 0

    print "Running TM on Testing Data ... "
    (activeColTest,
     activeCellsTest,
     activeFreqTest,
     predActiveFreqTest) = runTMOverDatasetFast(tm, activeColumnsTest, unionLength)

    distMatColumnTest = calculateDistanceMat(activeColTest, activeColTrain)
    distMatCellTest = calculateDistanceMat(activeCellsTest, activeCellsTrain)
    distMatActiveFreqTest = calculateDistanceMat(activeFreqTest, activeFreqTrain)
    distMatPredActiveFreqTest = calculateDistanceMat(predActiveFreqTest, predActiveFreqTrain)
    distMatColumnTest /= maxColumnOverlap
    distMatCellTest /= maxCellOverlap

    expResultTM = {"distMatColumnTrain": distMatColumnTrain,
                   "distMatCellTrain": distMatCellTrain,
                   "distMatActiveFreqTrain": distMatActiveFreqTrain,
                   "distMatPredActiveFreqTrain": distMatPredActiveFreqTrain,
                   "distMatColumnTest": distMatColumnTest,
                   "distMatCellTest": distMatCellTest,
                   "distMatActiveFreqTest": distMatActiveFreqTest,
                   "distMatPredActiveFreqTest": distMatPredActiveFreqTest}

    pickle.dump(expResultTM, open('results/distanceMat/{}_union_{}'.format(
      dataName, unionLength), 'w'))

    activeFreqResult = {"activeFreqTrain": activeFreqTrain,
                        "activeFreqTest": activeFreqTest,
                        "predActiveFreqTrain": predActiveFreqTrain,
                        "predActiveFreqTest": predActiveFreqTest}
    pickle.dump(activeFreqResult, open('results/activeFreq/{}'.format(
      dataName), 'w'))

    # fit supervised model
    from htmresearch.algorithms.sdr_classifier_batch import classificationNetwork

    classIdxMapTrain = {}
    classIdxMapTest = {}
    for classIdx in classList:
      classIdxMapTrain[classIdx] = np.where(trainLabel == classIdx)[0]
      classIdxMapTest[classIdx] = np.where(testLabel == classIdx)[0]

    options = {"useColumnRepresentation": False,
               "useCellRepresentation": True}

    classifierInputTrain = prepareClassifierInput(
      distMatColumnTrain, distMatCellTrain, classIdxMapTrain, trainLabel, options)

    classifierInputTest = prepareClassifierInput(
      distMatColumnTest, distMatCellTest, classIdxMapTrain, trainLabel, options)

    numInputs = len(classifierInputTrain[0])

    regularizationLambda = {"lambdaL2": [1],
                            "wIndice": [np.array(range(0, numInputs*numClass))]}
    # regularizationLambda = None
    cl = classificationNetwork(numInputs, numClass, regularizationLambda)

    wInit = np.zeros((numInputs, numClass))
    for classIdx in range(numClass):
      wInit[classIdx, classIdx] = 1
    wInit = np.reshape(wInit, (numInputs*numClass, ))
    cl.optimize(classifierInputTrain, trainLabel, wInit)
    trainAccuracy = cl.accuracy(classifierInputTrain, trainLabel)
    testAccuracy = cl.accuracy(classifierInputTest, testLabel)
    print "Train accuracy: {} test accuracy: {}".format(trainAccuracy,
                                                        testAccuracy)


    # default to use column distance only
    wOpt = {}
    bOpt = {}
    for classI in classList:
      wOpt[classI] = 0
      bOpt[classI] = 0

    # estimate the optimal weight factors
    wList = np.linspace(0, 1.0, 101)
    accuracyVsWRpt = np.zeros((len(wList), ))
    for i in range(len(wList)):
      accuracyVsWRpt[i] = - costFuncSharedW(
        wList[i], wOpt, bOpt, distMatColumnTrain, distMatCellTrain, trainLabel, classList)

    bestWForClassI = wList[np.argmax(accuracyVsWRpt)]
    for classI in classList:
      wOpt[classI] = bestWForClassI
    combinedDistanceMat = constructDistanceMat(distMatColumnTest,
                                               distMatCellTest,
                                               trainLabel, wOpt, bOpt)

    # testing

    print "Column Only model, Accuracy: {}".format(testAccuracyColumnOnly)

    testAccuracyActiveFreq, outcomeFreq = calculateAccuracy(
      distMatActiveFreqTest, trainLabel, testLabel)
    print "Active Freq Dist Accuracy {}".format(testAccuracyActiveFreq)

    testAccuracyPredActiveFreq, outcomeFreq = calculateAccuracy(
      distMatPredActiveFreqTest, trainLabel, testLabel)
    print "Pred-Active Freq Dist Accuracy {}".format(testAccuracyPredActiveFreq)

    testAccuracyCellOnly, outcomeCellOnly = calculateAccuracy(
      distMatCellTest, trainLabel, testLabel)
    print "Cell Dist accuracy {}".format(testAccuracyCellOnly)

    testAccuracyCombined, outcomeTM = calculateAccuracy(
      combinedDistanceMat, trainLabel, testLabel)


    print "Mixed weight accuracy {}".format(testAccuracyCombined)

    distMatColumnTestSort = sortDistanceMat(distMatColumnTest, trainLabel, testLabel)
    distMatCellTestSort = sortDistanceMat(distMatCellTest, trainLabel,
                                          testLabel)
    distMatActiveFreqTestSort = sortDistanceMat(distMatActiveFreqTest, trainLabel,
                                          testLabel)
    distMatPredActiveFreqTestSort = sortDistanceMat(distMatPredActiveFreqTest,
                                                trainLabel,
                                                testLabel)

    EuclideanDistanceMatSort = sortDistanceMat(EuclideanDistanceMat, trainLabel,
                                          testLabel)
    combinedDistanceMatSort = sortDistanceMat(combinedDistanceMat, trainLabel,
                                          testLabel)

    vLineLocs, hLineLocs = calculateClassLines(trainLabel, testLabel, classList)
    fig, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(distMatColumnTestSort)
    addClassLines(ax[0, 0], vLineLocs, hLineLocs)
    ax[0, 0].set_title('Column Dist, {:2.2f}'.format(testAccuracyColumnOnly))

    ax[0, 1].imshow(-EuclideanDistanceMatSort)
    addClassLines(ax[0, 1], vLineLocs, hLineLocs)
    ax[0, 1].set_title('- Euclidean Dist {:2.2f}'.format(accuracyEuclideanDist))

    ax[1, 0].imshow(distMatCellTestSort)
    addClassLines(ax[1, 0], vLineLocs, hLineLocs)
    ax[1, 0].set_title('Cell Dist, {:2.2f}'.format(testAccuracyCellOnly))

    ax[1, 1].imshow(distMatActiveFreqTestSort)
    addClassLines(ax[1, 1], vLineLocs, hLineLocs)
    ax[1, 1].set_title('Active Freq, {:2.2f}'.format(testAccuracyActiveFreq))

    ax[1, 2].imshow(distMatPredActiveFreqTestSort)
    addClassLines(ax[1, 2], vLineLocs, hLineLocs)
    ax[1, 2].set_title('Pred-Active Freq, {:2.2f}'.format(testAccuracyPredActiveFreq))

    ax[0, 2].imshow(combinedDistanceMatSort)
    addClassLines(ax[0, 2], vLineLocs, hLineLocs)
    ax[0, 2].set_title('Combined Dist {:2.2f}'.format(testAccuracyCombined))
    plt.savefig('figures/{}_summary.pdf'.format(dataName))

    # accuracy per class
    accuracyTM = {}
    accuracyEuclidean = {}
    accuracyColumn = {}
    accuracyCellOnly = {}
    for classI in classList:
      idx = np.where(testLabel == classI)[0]
      accuracyTM[classI] = np.mean(np.array(outcomeTM)[idx])
      accuracyEuclidean[classI] = np.mean(np.array(outcomeEuclidean)[idx])
      accuracyColumn[classI] = np.mean(np.array(outcomeColumn)[idx])
      accuracyCellOnly[classI] = np.mean(np.array(outcomeCellOnly)[idx])

    fig, ax = plt.subplots()
    width=0.5
    ax.bar(0, accuracyEuclideanDist, width, color='c')
    ax.bar(1, testAccuracyColumnOnly, width, color='g')
    ax.bar(2, testAccuracyCellOnly, width, color='b')
    ax.bar(3, testAccuracyCombined, width, color='r')
    plt.xlim([0, 4])
    plt.legend([ 'Euclidean', 'Column', 'Cell', 'Weighted Dist'], loc=3)
    plt.savefig('figures/{}_performance_overall.pdf'.format(dataName))

    fig, ax = plt.subplots()
    width = 0.2
    classIdx = np.array(accuracyTM.keys())
    ax.bar(classIdx-width, accuracyTM.values(), width, color='r')
    ax.bar(classIdx, accuracyColumn.values(), width, color='g')
    ax.bar(classIdx+width, accuracyCellOnly.values(), width, color='b')
    ax.bar(classIdx+2*width, accuracyEuclidean.values(), width, color='c')
    plt.legend(['Weighted Dist', 'Column', 'Cell', 'Euclidean'], loc=3)
    plt.ylabel('Accuracy')
    plt.xlabel('Class #')
    plt.ylim([0, 1.05])
    plt.savefig('figures/{}_performance_by_class.pdf'.format(dataName))


    expResults = {'accuracyEuclideanDist': accuracyEuclideanDist,
                  'accuracyColumnOnly': testAccuracyColumnOnly,
                  'accuracyTM': testAccuracyCombined,
                  'weights': wOpt,
                  'offsets': bOpt,
                  'EuclideanDistanceMat': EuclideanDistanceMat,
                  'activeColumnOverlap': distMatColumnTrain,
                  'activeCellOverlap': distMatCellTrain}
    pickle.dump(expResults, open('results/modelPerformance/{}'.format(dataName), 'w'))

