# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2015, Numenta, Inc.  Unless you have an agreement
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
from optparse import OptionParser

from matplotlib import pyplot as plt
import numpy as np

from pybrain.datasets import SequentialDataSet
from pybrain.tools.shortcuts import buildNetwork

from pybrain.structure import LinearLayer
from pybrain.structure import RecurrentNetwork
from pybrain.structure import FullConnection
from pybrain.structure.modules import LSTMLayer
from pybrain.supervised import RPropMinusTrainer

from swarm_runner import SwarmRunner
from scipy import random

import pandas as pd
from errorMetrics import *

from nupic.encoders.scalar import ScalarEncoder

plt.ion()

def initializeLSTMnet(nDimInput, nDimOutput, nLSTMcells=10):
  # Build LSTM network with nDim input units, nLSTMcells hidden units (LSTM cells) and nDim output cells
  net = buildNetwork(nDimInput, nLSTMcells, nDimOutput,
                     hiddenclass=LSTMLayer, bias=True, outputbias=True, recurrent=True)
  return net


def readDataSet(dataSet):
  filePath = 'data/'+dataSet+'.csv'
  # df = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['time', 'data'])
  # sequence = df['data']

  if dataSet=='nyc_taxi':
    df = pd.read_csv(filePath, header=0, skiprows=[1,2], names=['time', 'data', 'timeofday', 'dayofweek'])
    sequence = df['data']
    dayofweek = df['dayofweek']
    timeofday = df['timeofday']

    seq = pd.DataFrame(np.array(pd.concat([sequence, timeofday, dayofweek], axis=1)),
                        columns=['data', 'timeofday', 'dayofweek'])
  elif dataSet=='sine':
    df = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['time', 'data'])
    sequence = df['data']
    seq = pd.DataFrame(np.array(sequence), columns=['data'])
  else:
    raise(' unrecognized dataset type ')

  return seq


def getPyBrainDataSet(sequence, nTrain, predictionStep=1, useTimeOfDay=True, useDayOfWeek=True):
  print "generate a pybrain dataset of sequences"
  print "the training data contains ", str(nTrain-predictionStep), "records"

  inDim = 1 + int(useTimeOfDay) + int(useDayOfWeek)
  ds = SequentialDataSet(inDim, 1)
  if useTimeOfDay:
    print "include time of day as input field"
  if useDayOfWeek:
    print "include day of week as input field"

  for i in xrange(nTrain-predictionStep):
    if useTimeOfDay and useDayOfWeek:
      sample = np.array([sequence['data'][i], sequence['timeofday'][i], sequence['dayofweek'][i]])
    elif useTimeOfDay:
      sample = np.array([sequence['data'][i], sequence['timeofday'][i]])
    elif useDayOfWeek:
      sample = np.array([sequence['data'][i], sequence['dayofweek'][i]])
    else:
      sample = np.array([sequence['data'][i]])

    ds.addSample(sample, sequence['data'][i+predictionStep])
  return ds


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

  # parser.add_option("-n",
  #                   "--predictionstep",
  #                   type=int,
  #                   default=1,
  #                   dest="predictionstep",
  #                   help="number of steps ahead to be predicted")

  parser.add_option("-r",
                    "--repeatNumber",
                    type=int,
                    default=30,
                    dest="repeatNumber",
                    help="number of training epoches")

  (options, remainder) = parser.parse_args()
  print options

  return options, remainder


if __name__ == "__main__":

  (_options, _args) = _getArgs()
  dataSet = _options.dataSet

  rptNum = _options.repeatNumber

  print "run LSTM on ", dataSet
  SWARM_CONFIG = SwarmRunner.importSwarmDescription(dataSet)
  predictedField = SWARM_CONFIG['inferenceArgs']['predictedField']
  nTrain = SWARM_CONFIG["streamDef"]['streams'][0]['last_record']
  predictionStep = SWARM_CONFIG['inferenceArgs']['predictionSteps'][0]

  useTimeOfDay = True
  useDayOfWeek = True

  nTrain = 5000

  # prepare dataset as pyBrain sequential dataset
  sequence = readDataSet(dataSet)

  # standardize data by subtracting mean and dividing by std
  meanSeq = np.mean(sequence['data'])
  stdSeq = np.std(sequence['data'])
  sequence['data'] = (sequence['data'] - meanSeq)/stdSeq

  meanTimeOfDay = np.mean(sequence['timeofday'])
  stdTimeOfDay = np.std(sequence['timeofday'])
  sequence['timeofday'] = (sequence['timeofday'] - meanTimeOfDay)/stdTimeOfDay

  meanDayOfWeek = np.mean(sequence['dayofweek'])
  stdDayOfWeek = np.std(sequence['dayofweek'])
  sequence['dayofweek'] = (sequence['dayofweek'] - meanDayOfWeek)/stdDayOfWeek

  ds = getPyBrainDataSet(sequence, nTrain, predictionStep, useTimeOfDay, useDayOfWeek)

  print "train LSTM with "+str(rptNum)+" epochs"
  random.seed(6)
  net = initializeLSTMnet(nDimInput=len(ds.getSample()[0]), nDimOutput=1, nLSTMcells=20)

  trainer = RPropMinusTrainer(net, dataset=ds, verbose=True)
  error = []
  for rpt in xrange(rptNum):
    err = trainer.train()
    error.append(err)

  print "test LSTM"
  net.reset()

  predictedInput = np.zeros((len(sequence),))
  targetInput = np.zeros((len(sequence),))
  trueData = np.zeros((len(sequence),))
  for i in xrange(1, len(sequence)-predictionStep):
    if useTimeOfDay and useDayOfWeek:
      sample = np.array([sequence['data'][i], sequence['timeofday'][i], sequence['dayofweek'][i]])
    elif useTimeOfDay:
      sample = np.array([sequence['data'][i], sequence['timeofday'][i]])
    elif useDayOfWeek:
      sample = np.array([sequence['data'][i], sequence['dayofweek'][i]])
    else:
      sample = np.array([sequence['data'][i]])

    netActivation = net.activate(sample)
    predictedInput[i] = netActivation
    targetInput[i] = sequence['data'][i+predictionStep]
    trueData[i] = sequence['data'][i]
    # print " target input: ", targetInput[i], " predicted Input: ", predictedInput[i]

  predictedInput = (predictedInput * stdSeq) + meanSeq
  targetInput = (targetInput * stdSeq) + meanSeq
  trueData = (trueData * stdSeq) + meanSeq

  nrmse_train = NRMSE(targetInput[:nTrain], predictedInput[:nTrain])
  nrmse_test = NRMSE(targetInput[nTrain:-predictionStep], predictedInput[nTrain:-predictionStep])

  print "NRMSE, Train %f, Test %f" %(nrmse_train, nrmse_test)

  (window_center, nrmse_slide) = NRMSE_sliding(targetInput, predictedInput, 480)

  plt.close('all')
  plt.figure(1)
  plt.plot(targetInput[nTrain:], color='black')
  plt.plot(predictedInput[nTrain:], color='red')
  plt.title('LSTM, useTimeOfDay='+str(useTimeOfDay)+dataSet+' test NRMSE = '+str(nrmse_test))
  plt.xlim([0, 500])
  plt.xlabel('Time')
  plt.ylabel('Prediction')


  plt.figure(2)
  plt.plot(targetInput, color='black')
  plt.plot(predictedInput, color='red')
  plt.title('LSTM, useTimeOfDay='+str(useTimeOfDay)+dataSet+' test NRMSE = '+str(nrmse_test))
  NT = len(trueData)
  plt.xlim([NT-500, NT-predictionStep])
  plt.ylim([0, 40000])
  # plt.xlim([0, 500])
  plt.xlabel('Time')
  plt.ylabel('Prediction')


  fig = plt.figure(3)
  plt.subplot(2, 1, 1)
  plt.plot(window_center, nrmse_slide)
  plt.xlabel(' Time ')
  plt.ylabel(' NRMSE')

  plt.subplot(2, 1, 2)
  plt.plot(targetInput[:], color='black')
  plt.plot(predictedInput[:], color='red')
  plt.xlabel('Time')
  plt.ylabel('Prediction')

  # import plotly.plotly as py
  # plot_url = py.plot_mpl(fig)

  #
  filePath = 'data/'+dataSet+'.csv'
  inputFile = open(filePath, "rb")

  csvReader = csv.reader(inputFile)
  # skip header rows
  csvReader.next()
  csvReader.next()
  csvReader.next()

  outputFileName = './prediction/'+dataSet+'_lstm_pred_useTimeOfDay_'+str(useTimeOfDay)+\
                   '_useDayOfWeek'+str(useDayOfWeek)+'.csv'
  outputFile = open(outputFileName,"w")
  csvWriter = csv.writer(outputFile)
  csvWriter.writerow(['timestamp', predictedField, 'prediction-'+str(predictionStep)+'step'])
  csvWriter.writerow(['datetime', 'float', 'float'])
  csvWriter.writerow(['', '', ''])

  for i in xrange(len(sequence)):
    row = csvReader.next()
    csvWriter.writerow([row[0], row[1], predictedInput[i]])

  inputFile.close()
  outputFile.close()

