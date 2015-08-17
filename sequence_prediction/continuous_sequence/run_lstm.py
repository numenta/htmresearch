#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
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


import pandas as pd

plt.ion()

def initializeLSTMnet(nDim, nLSTMcells=10):
  # Build LSTM network with nDim input units, nLSTMcells hidden units (LSTM cells) and nDim output cells
  net = buildNetwork(nDim, nLSTMcells, nDim,
                     hiddenclass=LSTMLayer, bias=True, outputbias=True, recurrent=True)

  return net

def readDataSet(dataSet):
  filePath = 'data/'+dataSet+'.csv'
  df = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['time', 'data'])
  sequence = df['data']

  return sequence

def getPyBrainDataSet(sequence, nTrain, predictionStep=1):
  print "generate a pybrain dataset of sequences"
  print "the training data contains ", str(nTrain-predictionStep), "records"
  ds = SequentialDataSet(1, 1)
  for i in xrange(nTrain-predictionStep):
    ds.addSample(sequence[i], sequence[i+predictionStep])
  return ds


def NRMSE(data, pred):
  return np.sqrt(np.nanmean(np.square(pred-data)))/np.sqrt(np.nanmean( np.square(data-np.nanmean(data))))


def _getArgs():
  parser = OptionParser(usage="%prog PARAMS_DIR OUTPUT_DIR [options]"
                              "\n\nCompare TM performance with trivial predictor using "
                              "model outputs in prediction directory "
                              "and outputting results to result directory.")
  parser.add_option("-d",
                    "--dataSet",
                    type=str,
                    default='sine',
                    dest="dataSet",
                    help="DataSet Name, choose from sine, SantaFe_A, MackeyGlass")

  parser.add_option("-n",
                    "--predictionstep",
                    type=int,
                    default=1,
                    dest="predictionstep",
                    help="number of steps ahead to be predicted")

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
  predictionStep = _options.predictionstep
  rptNum = _options.repeatNumber

  print "run LSTM on ", dataSet
  SWARM_CONFIG = SwarmRunner.importSwarmDescription(dataSet)
  predictedField = SWARM_CONFIG['inferenceArgs']['predictedField']
  nTrain = SWARM_CONFIG["streamDef"]['streams'][0]['last_record']

  # prepare dataset as pyBrain sequential dataset
  sequence = readDataSet(dataSet)

  # standardize data by subtracting mean and dividing by std
  meanSeq = np.mean(sequence)
  stdSeq = np.std(sequence)
  sequence = (sequence - meanSeq)/stdSeq

  ds = getPyBrainDataSet(sequence, nTrain, predictionStep)

  print "train LSTM with "+str(rptNum)+" repeats"

  net = initializeLSTMnet(1, nLSTMcells=20)

  trainer = RPropMinusTrainer(net, dataset=ds, verbose=True)
  trainer.trainEpochs(rptNum)

  print "test LSTM"
  net.reset()

  predictedInput = np.zeros((len(sequence),))
  targetInput = np.zeros((len(sequence),))
  for i in xrange(len(sequence)-predictionStep):
    netActivation = net.activate(sequence[i])
    predictedInput[i] = (netActivation)
    targetInput[i] = sequence[i+predictionStep]
    print " target input: ", targetInput[i], " predicted Input: ", predictedInput[i]

  predictedInput = (predictedInput * stdSeq) + meanSeq
  targetInput = (targetInput * stdSeq) + meanSeq

  # predictedInput = predictedInput[nTrain:]
  # targetInput = targetInput[nTrain:]

  nrmse = NRMSE(targetInput[nTrain:-predictionStep], predictedInput[nTrain:-predictionStep])

  plt.close('all')
  plt.figure(1)
  plt.plot(targetInput[nTrain:], color='black')
  plt.plot(predictedInput[nTrain:], color='red')
  plt.title(dataSet+' NRMSE = '+str(nrmse))
  plt.xlim([100, 200])
  plt.xlabel('Time')
  plt.ylabel('Prediction')


  filePath = 'data/'+dataSet+'.csv'
  inputFile = open(filePath, "rb")

  csvReader = csv.reader(inputFile)
  # skip header rows
  csvReader.next()
  csvReader.next()
  csvReader.next()

  outputFileName = './prediction/'+dataSet+'_lstm_pred.csv'
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

