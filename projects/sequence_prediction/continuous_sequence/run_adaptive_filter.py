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

import adaptfilt
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

plt.ion()


def readDataSet(dataSet):
  filePath = 'data/' + dataSet + '.csv'

  if dataSet == 'nyc_taxi':
    df = pd.read_csv(filePath, header=0, skiprows=[1, 2],
                     names=['time', 'data', 'timeofday', 'dayofweek'])
    sequence = df['data']
  elif dataSet == 'sine':
    df = pd.read_csv(filePath, header=0, skiprows=[1, 2],
                     names=['time', 'data'])
    sequence = df['data']

  else:
    raise (' unrecognized dataset type ')

  return np.array(sequence)



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



def normalizeSequence(sequence):
  """
  normalize sequence by subtracting the mean and
  :param sequence: a list of data samples
  :param considerDimensions: a list of dimensions to consider
  :return: normalized sequence
  """
  seq = np.array(sequence).astype('float64')

  meanSeq = np.mean(seq)
  stdSeq = np.std(seq)
  seq = (seq - np.mean(seq)) / np.std(seq)

  sequence = seq.tolist()
  return sequence, meanSeq, stdSeq



if __name__ == "__main__":

  (_options, _args) = _getArgs()
  dataSet = _options.dataSet
  numTrain = _options.trainingDataSize

  print "run adaptive filter on ", dataSet

  sequence = readDataSet(dataSet)

  # predict 5 steps ahead
  predictionStep = 5

  sequence, meanSeq, stdSeq = normalizeSequence(sequence)

  targetInput = np.zeros((len(sequence),))
  predictedInput = np.zeros((len(sequence),))

  numTrain = 6000
  filterLength = 10
  for i in xrange(numTrain, len(sequence) - predictionStep):
    y, e, w = adaptfilt.lms(sequence[(i-numTrain):(i-predictionStep+1)],
                            sequence[(i-numTrain+predictionStep):(i+1)],
                            M=filterLength, step=0.01)

    # use the resulting filter coefficeints to make prediction
    target = np.convolve(sequence[(i-filterLength):(i+1)], w)
    predictedInput[i] = target[filterLength]
    targetInput[i] = sequence[i + predictionStep]

    print "record {} value {} predicted {}".format(i, targetInput[i], predictedInput[i])

  predictedInput = (predictedInput * stdSeq) + meanSeq
  targetInput = (targetInput * stdSeq) + meanSeq
  saveResultToFile(dataSet, predictedInput, 'adaptiveFilter')

  from plot import computeAltMAPE, computeNRMSE
  MAPE = computeAltMAPE(predictedInput, targetInput, startFrom=6000)
  NRMSE = computeNRMSE(predictedInput, targetInput, startFrom=6000)
  print "MAPE {}".format(MAPE)
  print "NRMSE {}".format(NRMSE)
  #
  # plt.figure()
  # plt.plot(targetInput)
  # plt.plot(predictedInput)
  # plt.xlim([12800, 13500])
  # plt.ylim([0, 30000])
