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

import csv
import math
import operator
from optparse import OptionParser
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

plt.ion()



def euclideanDistance(instance1, instance2, considerDimensions):
  """
  Calculate Euclidean Distance between two samples
  Example use:
  data1 = [2, 2, 2, 'class_a']
  data2 = [4, 4, 4, 'class_b']
  distance = euclideanDistance(data1, data2, 3)

  :param instance1: list of attributes
  :param instance2: list of attributes
  :param considerDimensions: a list of dimensions to consider
  :return: float euclidean distance between data1 & 2
  """
  distance = 0
  for x in considerDimensions:
    distance += pow((instance1[x] - instance2[x]), 2)
  return math.sqrt(distance)



def getNeighbors(trainingSet, testInstance, k, considerDimensions=None):
  """
  collect the k most similar instances in the trainingSet for a given test
  instance
  :param trainingSet: A list of data instances
  :param testInstance: a single data instance
  :param k: number of neighbors
  :param considerDimensions: a list of dimensions to consider
  :return: neighbors: a list of neighbor instance
  """
  if considerDimensions is None:
    considerDimensions = len(testInstance) - 1

  neighborList = []
  for x in range(len(trainingSet)):
    dist = euclideanDistance(testInstance, trainingSet[x], considerDimensions)
    neighborList.append((trainingSet[x], dist))
  neighborList.sort(key=operator.itemgetter(1))

  neighbors = []
  distances = []
  for x in range(k):
    neighbors.append(neighborList[x][0])
    distances.append(neighborList[x][1])

  return neighbors



def getResponse(neighbors, weights=None):
  """
  Calculated weighted response based on a list of nearest neighbors
  :param neighbors: a list of neighbors, each entry is a data instance
  :param weights: a numpy array of the same length as the neighbors
  :return: weightedAvg: weighted average response
  """
  neighborResponse = []
  for x in range(len(neighbors)):
    neighborResponse.append(neighbors[x][-1])

  neighborResponse = np.array(neighborResponse).astype('float')
  if weights is None:
    weightedAvg = np.mean(neighborResponse)
  else:
    weightedAvg = np.sum(weights * neighborResponse)

  return weightedAvg



def readDataSet(dataSet):
  filePath = 'data/' + dataSet + '.csv'

  if dataSet == 'nyc_taxi':
    df = pd.read_csv(filePath, header=0, skiprows=[1, 2],
                     names=['time', 'data', 'timeofday', 'dayofweek'])
    sequence = df['data']
    dayofweek = df['dayofweek']
    timeofday = df['timeofday']
    sequence5stepsAgo = np.roll(np.array(sequence), 5)

    seq = []
    for i in xrange(len(sequence)):
      seq.append(
        [timeofday[i], dayofweek[i], sequence5stepsAgo[i], sequence[i]])

  else:
    raise (' unrecognized dataset type ')

  return seq



def _getArgs():
  parser = OptionParser(usage="%prog PARAMS_DIR OUTPUT_DIR [options]"
                              "\n\nCompare TM performance with trivial predictor using "
                              "model outputs in prediction directory "
                              "and outputting results to result directory.")
  parser.add_option("-d",
                    "--dataSet",
                    type=str,
                    default='nyc_taxi',
                    dest="dataSet",
                    help="DataSet Name, choose from sine, SantaFe_A, MackeyGlass")

  parser.add_option("-n",
                    "--trainingDataSize",
                    type=int,
                    default=6000,
                    dest="trainingDataSize",
                    help="size of training dataset")

  (options, remainder) = parser.parse_args()
  print options

  return options, remainder



def saveResultToFile(dataSet, predictedInput, algorithmName):
  inputFileName = 'data/' + dataSet + '.csv'
  inputFile = open(inputFileName, "rb")

  csvReader = csv.reader(inputFile)

  # skip header rows
  csvReader.next()
  csvReader.next()
  csvReader.next()

  outputFileName = './prediction/' + dataSet + '_' + algorithmName + '_pred.csv'
  outputFile = open(outputFileName, "w")
  csvWriter = csv.writer(outputFile)
  csvWriter.writerow(
    ['timestamp', 'data', 'prediction-' + str(predictionStep) + 'step'])
  csvWriter.writerow(['datetime', 'float', 'float'])
  csvWriter.writerow(['', '', ''])

  for i in xrange(len(sequence)):
    row = csvReader.next()
    csvWriter.writerow([row[0], row[1], predictedInput[i]])

  inputFile.close()
  outputFile.close()



def normalizeSequence(sequence, considerDimensions=None):
  """
  normalize sequence by subtracting the mean and
  :param sequence: a list of data samples
  :param considerDimensions: a list of dimensions to consider
  :return: normalized sequence
  """
  seq = np.array(sequence).astype('float64')
  nSampleDim = seq.shape[1]

  if considerDimensions is None:
    considerDimensions = range(nSampleDim)

  for dim in considerDimensions:
    seq[:, dim] = (seq[:, dim] - np.mean(seq[:, dim])) / np.std(seq[:, dim])

  sequence = seq.tolist()
  return sequence



if __name__ == "__main__":

  (_options, _args) = _getArgs()
  dataSet = _options.dataSet
  numTrain = _options.trainingDataSize

  print "run knn on ", dataSet

  sequence = readDataSet(dataSet)

  # predict 5 steps ahead
  predictionStep = 5
  nFeature = 2
  k = 10

  sequence = normalizeSequence(sequence, considerDimensions=[0, 1, 2])

  targetInput = np.zeros((len(sequence),))
  predictedInput = np.zeros((len(sequence),))

  for i in xrange(numTrain, len(sequence) - predictionStep):
    testInstance = sequence[i + predictionStep]
    targetInput[i] = testInstance[-1]

    # select data points at that shares same timeOfDay and dayOfWeek
    neighbors = getNeighbors(sequence[i - numTrain:i], testInstance, k, [0, 1, 2])
    predictedInput[i] = getResponse(neighbors)

    print "step %d, target input: %d  predicted Input: %d " % (
      i, targetInput[i], predictedInput[i])

  saveResultToFile(dataSet, predictedInput, 'plainKNN')

  plt.figure()
  plt.plot(targetInput)
  plt.plot(predictedInput)
  plt.xlim([12800, 13500])
  plt.ylim([0, 30000])
