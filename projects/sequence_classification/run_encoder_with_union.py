"""
Run sequence classification experiment with
Input -> RDSE encoder -> Union model
Search for the optimal union window
"""

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



def unionForOneSequence(activeColumns, unionLength=1):
  activeColumnTrace = []

  unionStepInBatch = 0
  unionBatchIdx = 0
  unionCols = set()
  for t in range(len(activeColumns)):
    unionCols = unionCols.union(activeColumns[t])

    unionStepInBatch += 1
    if unionStepInBatch == unionLength:
      activeColumnTrace.append(unionCols)
      unionStepInBatch = 0
      unionBatchIdx += 1
      unionCols = set()

  if unionStepInBatch > 0:
    activeColumnTrace.append(unionCols)

  return activeColumnTrace



def runUnionStep(activeColumns, unionLength=1):
  """
  Run encoder -> tm network over dataset, save activeColumn and activeCells
  traces
  :param tm:
  :param encoder:
  :param dataset:
  :return:
  """
  numSequence = len(activeColumns)
  activeColumnUnionTrace = []

  for i in range(numSequence):
    activeColumnTrace = unionForOneSequence(activeColumns[i], unionLength)
    activeColumnUnionTrace.append(activeColumnTrace)
    # print "{} out of {} done ".format(i, numSequence)
  return activeColumnUnionTrace



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

  while not taskQueue.empty():
    time.sleep(0.1)
  accuracyVsResolution = np.zeros((len(nBucketList,)))
  while not resultQueue.empty():
    exptResult = resultQueue.get()
    nBuckets = exptResult.keys()[0]
    accuracyVsResolution[nBucketList.index(nBuckets)] = exptResult[nBuckets]

  return accuracyVsResolution



def runDataSet(dataName, datasetName):
  trainData, trainLabel, testData, testLabel = loadDataset(dataName,
                                                           datasetName)
  numTest = len(testLabel)
  numTrain = len(trainLabel)
  sequenceLength = len(trainData[0])
  classList = np.unique(trainLabel).tolist()
  numClass = len(classList)

  print "Processing {}".format(dataName)
  print "Train Sample # {}, Test Sample # {}".format(numTrain, numTest)
  print "Sequence Length {} Class # {}".format(sequenceLength, len(classList))

  if (max(numTrain, numTest) * sequenceLength < 600 * 600):
    print "skip this small dataset for now"
    return

  try:
    unionLengthList = [1, 5, 10, 15, 20]
    for unionLength in unionLengthList:
      expResultTM = pickle.load(open('results/modelPerformance/{}_columnOnly_union_{}'.format(
        dataName, unionLength), 'r'))
    return
  except:
    print "run data set: ", dataName

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
    optimalResolution = (maxValue - minValue) / optNumBucket
  except:
    return
    nBucketList = range(20, 200, 10)
    accuracyVsResolution = searchForOptimalEncoderResolution(
      nBucketList, trainData, trainLabel, numCols, w)
    optNumBucket = nBucketList[np.argmax(np.array(accuracyVsResolution))]
    optimalResolution = (maxValue - minValue) / optNumBucket
    searchResolution = {
      'nBucketList': nBucketList,
      'accuracyVsResolution': accuracyVsResolution,
      'optimalResolution': optimalResolution
    }
    # save optimal resolution for future use
    outputFile = open('results/optimalEncoderResolution/{}'.format(dataName),
                      'w')
    pickle.dump(searchResolution, outputFile)
    outputFile.close()
  print "optimal bucket # {}".format((maxValue - minValue) / optimalResolution)

  encoder = RandomDistributedScalarEncoder(optimalResolution, w=w, n=numCols)
  print "encoding train data ..."
  activeColumnsTrain = runEncoderOverDataset(encoder, trainData)
  print "encoding test data ..."
  activeColumnsTest = runEncoderOverDataset(encoder, testData)
  print "calculate column distance matrix ..."

  # run encoder -> union model, search for the optimal union window
  unionLengthList = [1, 5, 10, 15, 20]
  for unionLength in unionLengthList:
    activeColumnUnionTrain = runUnionStep(activeColumnsTrain, unionLength)
    activeColumnUnionTest = runUnionStep(activeColumnsTest, unionLength)

    distMatColumnTrain = calculateDistanceMatTrain(activeColumnUnionTrain)
    distMatColumnTest = calculateDistanceMat(activeColumnUnionTest,
                                             activeColumnUnionTrain)

    trainAccuracyColumnOnly, outcomeColumn = calculateAccuracy(distMatColumnTest,
                                                              trainLabel,
                                                              testLabel)

    testAccuracyColumnOnly, outcomeColumn = calculateAccuracy(distMatColumnTest,
                                                              trainLabel,
                                                              testLabel)

    expResults = {'distMatColumnTrain': distMatColumnTrain,
                  'distMatColumnTest': distMatColumnTest,
                  'trainAccuracyColumnOnly': trainAccuracyColumnOnly,
                  'testAccuracyColumnOnly': testAccuracyColumnOnly}
    outputFile = open('results/distanceMat/{}_columnOnly_union_{}'.format(
      dataName, unionLength), 'w')
    pickle.dump(expResults, outputFile)
    outputFile.close()



def runDataSetWorker(taskQueue, datasetName):
  while True:
    nextTask = taskQueue.get()
    print "Next task is : ", nextTask
    if nextTask is None:
      break
    dataName = nextTask["dataName"]
    runDataSet(dataName, datasetName)
  return



if __name__ == "__main__":
  datasetName = "SyntheticData"
  dataSetList = listDataSets(datasetName)

  datasetName = 'UCR_TS_Archive_2015'
  dataSetList = listDataSets(datasetName)
  # dataSetList = ["synthetic_control"]

  numCPU = multiprocessing.cpu_count()
  numTask = 8
  # Establish communication queues
  taskQueue = multiprocessing.JoinableQueue()

  for dataName in dataSetList:
    taskQueue.put({"dataName": dataName,
                   "datasetName": datasetName})
  for _ in range(numTask):
    taskQueue.put(None)
  jobs = []
  for i in range(numTask):
    print "Start process ", i
    p = multiprocessing.Process(target=runDataSetWorker, args=(taskQueue, datasetName))
    jobs.append(p)
    p.daemon = True
    p.start()

  while not taskQueue.empty():
    time.sleep(5)
