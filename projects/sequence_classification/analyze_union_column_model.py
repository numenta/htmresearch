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
Analyze experiment results from the RDSE->Union model
One needs to run the script "run_encoder_with_union.py" first to get the
experiment results (distance matrices)
"""
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from util_functions import *

plt.ion()

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams.update({'figure.autolayout': True})



def runTMOverDatasetFast(tm, activeColumns, unionLength=0):
  """
  Run encoder -> tm network over dataset, save activeColumn and activeCells
  traces
  :param tm:
  :param encoder:
  :param dataset:
  :return:
  """

  sequenceLength = len(activeColumns[0])
  numSequence = len(activeColumns)
  numCells = tm.getCellsPerColumn() * tm.getColumnDimensions()[0]
  numSteps = sequenceLength / unionLength
  if np.mod(sequenceLength, unionLength) > 0:
    numSteps += 1

  predictedActiveCellsUnionTrace = []
  activationFrequencyTrace = []
  predictedActiveFrequencyTrace = []

  for i in range(numSequence):
    activeCellsTrace = []
    predictiveCellsTrace = []
    predictedActiveCellsTrace = []

    unionStep = 0
    # unionCells = np.zeros((numCells, ))
    unionBatchIdx = 0
    unionCells = set()
    activationFrequency = np.zeros((numCells, ))
    predictedActiveFrequency = np.zeros((numCells,))
    for t in range(sequenceLength):
      tm.compute(activeColumns[i][t], learn=False)
      activeCellsTrace.append(set(tm.getActiveCells()))
      predictiveCellsTrace.append(set(tm.getPredictiveCells()))
      if t == 0:
        predictedActiveCells = set()
      else:
        # predictedActiveCells = activeCellsTrace[t]
        predictedActiveCells = activeCellsTrace[t].intersection(predictiveCellsTrace[t-1])

      activationFrequency[tm.getActiveCells()] += 1
      predictedActiveFrequency[list(predictedActiveCells)] += 1
      unionCells = unionCells.union(predictedActiveCells)
      # unionCells[list(predictedActiveCells)] += 1
      unionStep += 1
      if unionStep == unionLength:
        predictedActiveCellsTrace.append(unionCells)
        # predictedActiveCellsUnionTrace[i, unionBatchIdx, :] = unionCells
        # unionCells = np.zeros((numCells,))
        unionStep = 0
        unionBatchIdx += 1
        unionCells = set()
    if unionStep > 0:
      predictedActiveCellsTrace.append(unionCells)
      # predictedActiveCellsUnionTrace[i, unionBatchIdx, :] = unionCells

    activationFrequency = activationFrequency/np.linalg.norm(activationFrequency)
    predictedActiveFrequency = predictedActiveFrequency / np.linalg.norm(predictedActiveFrequency)

    predictedActiveCellsUnionTrace.append(predictedActiveCellsTrace)
    activationFrequencyTrace.append(activationFrequency)
    predictedActiveFrequencyTrace.append(predictedActiveFrequency)
    print "{} out of {} done ".format(i, numSequence)

  return (predictedActiveCellsUnionTrace,
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




if __name__ == "__main__":
  plt.close('all')
  datasetName = "SyntheticData"
  dataSetList = listDataSets(datasetName)

  datasetName = 'UCR_TS_Archive_2015'
  dataSetList = listDataSets(datasetName)
  # dataSetList = ["synthetic_control"]


  accuracyAll = []
  dataSetNameList = []
  for dataName in dataSetList:
    plt.close('all')
    trainData, trainLabel, testData, testLabel = loadDataset(dataName, datasetName)
    numTest = len(testLabel)
    numTrain = len(trainLabel)
    sequenceLength = len(trainData[0])
    classList = np.unique(trainLabel).tolist()
    numClass = len(classList)

    # if numTrain <= 30:
    #   continue
    # print
    print "Processing {}".format(dataName)
    print "Train Sample # {}, Test Sample # {}".format(numTrain, numTest)
    print "Sequence Length {} Class # {}".format(sequenceLength, len(classList))

    # if max(numTest, numTrain) * sequenceLength > 600 * 600:
    #   print "skip this large dataset for now"
    #   continue

    try:
      unionLengthList = [1, 5, 10, 15, 20]
      for unionLength in unionLengthList:
        expResultTM = pickle.load(open('results/distanceMat/{}_columnOnly_union_{}'.format(
          dataName, unionLength), 'r'))
    except:
      continue

    EuclideanDistanceMat = calculateEuclideanDistanceMat(testData, trainData)
    outcomeEuclidean = calculateEuclideanModelAccuracy(trainData, trainLabel, testData, testLabel)
    accuracyEuclideanDist = np.mean(outcomeEuclidean)
    print "Euclidean model accuracy: {}".format(accuracyEuclideanDist)

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
      continue

    expResultTM = pickle.load(open('results/distanceMat/{}_columnOnly_union_{}'.format(
      dataName, 1), 'r'))

    distMatColumnTest = expResultTM['distMatColumnTest']
    distMatColumnTrain = expResultTM['distMatColumnTrain']
    # fit supervised model
    testAccuracyListVsUnionLength = []
    trainAccuracyListVsUnionLength = []
    unionLengthList = [1, 5, 10, 15, 20]
    for unionLength in unionLengthList:
      expResultTM = pickle.load(open('results/distanceMat/{}_columnOnly_union_{}'.format(
        dataName, unionLength), 'r'))

      distMatColumnUnionTest = expResultTM['distMatColumnTest']
      distMatColumnUnionTrain = expResultTM['distMatColumnTrain']


      options = {"useColumnRepresentation": True,
                 "useCellRepresentation": True}


      # calculate accuracy
      trainAccuracyColumnUnion, outcomeColumn = calculateAccuracy(
        distMatColumnUnionTrain, trainLabel, trainLabel)
      testAccuracyColumnUnion, outcomeColumn = calculateAccuracy(
        distMatColumnUnionTest, trainLabel, testLabel)

      trainAccuracyColumnOnly, outcomeColumn = calculateAccuracy(
        distMatColumnTrain, trainLabel, trainLabel)
      testAccuracyColumnOnly, outcomeColumn = calculateAccuracy(
        distMatColumnTest, trainLabel, testLabel)
      print "Column Only model, Accuracy: {}".format(testAccuracyColumnOnly)
      print "Column wt Union model, Accuracy: {}".format(testAccuracyColumnUnion)

      accuracyListTrain = np.array([accuracyEuclideanDist,
                                    trainAccuracyColumnOnly,
                                    trainAccuracyColumnUnion])

      accuracyListTest = np.array([accuracyEuclideanDist,
                                   testAccuracyColumnOnly,
                                   testAccuracyColumnUnion])

      testAccuracyListVsUnionLength.append(accuracyListTest)
      trainAccuracyListVsUnionLength.append(accuracyListTest)

    trainAccuracyListVsUnionLength = np.array(trainAccuracyListVsUnionLength)
    testAccuracyListVsUnionLength = np.array(testAccuracyListVsUnionLength)

    numModel = testAccuracyListVsUnionLength.shape[1]
    bestAccuracy = np.zeros((numModel, ))

    for i in range(numModel):
      idx = np.argmax(trainAccuracyListVsUnionLength[:, i])
      bestAccuracy[i] = testAccuracyListVsUnionLength[idx, i]

    bestAccuracy[1] = testAccuracyListVsUnionLength[0, 1]
    accuracyAll.append(bestAccuracy)
    dataSetNameList.append(dataName)
    continue


  accuracyAll = np.array(accuracyAll)


  # fig, ax = plt.subplots(1, 2)
  # (T, p) = scipy.stats.wilcoxon(accuracyAll[:, 1], accuracyAll[:, 6])
  # ax[0].plot(accuracyAll[:, 1]*100, accuracyAll[:, 6]*100, 'ko')
  # ax[0].plot([0, 105], [0, 105], 'k--')
  # ax[0].set_xlim([0, 105])
  # ax[0].set_ylim([0, 105])
  # ax[0].set_xlabel('1-NN Accuracy (%)')
  # ax[0].set_ylabel('classifier Accuracy (%)')
  # ax[0].set_aspect('equal')
  # ax[0].set_title('n={} p={}'.format(len(accuracyAll), p))

  fig, ax = plt.subplots(2, 2)

  ax[0, 0].plot(accuracyAll[:, 0] * 100, accuracyAll[:, 1] * 100, 'ko')
  ax[0, 0].plot([0, 105], [0, 105], 'k--')
  ax[0, 0].set_xlim([0, 105])
  ax[0, 0].set_ylim([0, 105])
  ax[0, 0].set_ylabel('column representation (%)')
  ax[0, 0].set_xlabel('Euclidean distance (%)')
  ax[0, 0].set_aspect('equal')
  improv = np.mean((accuracyAll[:, 1] - accuracyAll[:, 0]) / accuracyAll[:, 0])
  ax[0, 0].set_title('n={} improv={:3f}'.format(len(accuracyAll), improv))

  ax[0, 1].plot(accuracyAll[:, 0] * 100, accuracyAll[:, 2] * 100, 'ko')
  ax[0, 1].plot([0, 105], [0, 105], 'k--')
  ax[0, 1].set_xlim([0, 105])
  ax[0, 1].set_ylim([0, 105])
  ax[0, 1].set_xlabel('Euclidean distance (%)')
  ax[0, 1].set_ylabel('column wt union (%)')
  ax[0, 1].set_aspect('equal')
  improv = np.mean((accuracyAll[:, 2] - accuracyAll[:, 0]) / accuracyAll[:, 0])
  ax[0, 1].set_title('n={} improv={:3f}'.format(len(accuracyAll), improv))
  plt.savefig("figures/rdse_union_model_performance.pdf")



