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
from pybrain.structure.modules import SigmoidLayer

from pybrain.supervised import RPropMinusTrainer

from swarm_runner import SwarmRunner


import pandas as pd
from errorMetrics import *

from nupic.encoders.scalar import ScalarEncoder


plt.ion()

def initializeLSTMnet(nDimInput, nDimOutput, nLSTMcells=10):
  # Build LSTM network with nDim input units, nLSTMcells hidden units (LSTM cells) and nDim output cells
  net = buildNetwork(nDimInput, nLSTMcells, nDimOutput,
                     hiddenclass=LSTMLayer, bias=True, outclass=SigmoidLayer, recurrent=True)
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


def getPyBrainDataSetScalarEncoder(sequence, nTrain, encoder, predictionStep=1, useTimeOfDay=True, useDayOfWeek=True):
  """
  Use scalar encoder for the data
  :param sequence:
  :param nTrain:
  :param predictionStep:
  :param useTimeOfDay:
  :param useDayOfWeek:
  :return:
  """
  print "generate a pybrain dataset of sequences"
  print "the training data contains ", str(nTrain-predictionStep), "records"

  inDim = encoder.n + int(useTimeOfDay) + int(useDayOfWeek)
  outDim = encoder.n
  ds = SequentialDataSet(inDim, outDim)
  if useTimeOfDay:
    print "include time of day as input field"
  if useDayOfWeek:
    print "include day of week as input field"

  for i in xrange(nTrain-predictionStep):

    dataSDR = encoder.encode(sequence['data'][i])

    if useTimeOfDay and useDayOfWeek:
      sample = np.concatenate((dataSDR, [sequence['timeofday'][i]], [sequence['dayofweek'][i]]), axis=0)
    elif useTimeOfDay:
      sample = np.concatenate((dataSDR, [sequence['timeofday'][i]]), axis=0)
    elif useDayOfWeek:
      sample = np.concatenate((dataSDR, [sequence['dayofweek'][i]]), axis=0)
    else:
      sample = dataSDR

    ds.addSample(sample, encoder.encode(sequence['data'][i+predictionStep]))

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
                    default=20,
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
  useDayOfWeek = False

  nTrain = 5000

  # prepare dataset as pyBrain sequential dataset
  sequence = readDataSet(dataSet)
  encoder = ScalarEncoder(w=1, minval=0, maxval=40000, n=15, forced=True)

  meanTimeOfDay = np.mean(sequence['timeofday'])
  stdTimeOfDay = np.std(sequence['timeofday'])
  sequence['timeofday'] = (sequence['timeofday'] - meanTimeOfDay)/stdTimeOfDay

  ds = getPyBrainDataSetScalarEncoder(sequence, nTrain, encoder, predictionStep, useTimeOfDay, useDayOfWeek)

  print "train LSTM with "+str(rptNum)+" repeats"

  net = initializeLSTMnet(nDimInput=len(ds.getSample()[0]), nDimOutput=encoder.n, nLSTMcells=20)

  trainer = RPropMinusTrainer(net, dataset=ds, verbose=True)
  error = []
  for rpt in xrange(rptNum):
    err = trainer.train()
    error.append(err)

  print "test LSTM"
  net.reset()

  predictedDistribution = np.zeros((len(sequence), encoder.n))
  targetDistribution = np.zeros((len(sequence), encoder.n))
  targetInput = np.zeros((len(sequence),))
  trueData = np.zeros((len(sequence),))
  for i in xrange(len(sequence)-predictionStep):
    dataSDR = encoder.encode(sequence['data'][i])

    if useTimeOfDay and useDayOfWeek:
      sample = np.concatenate((dataSDR, [sequence['timeofday'][i]], [sequence['dayofweek'][i]]), axis=0)
    elif useTimeOfDay:
      sample = np.concatenate((dataSDR, [sequence['timeofday'][i]]), axis=0)
    elif useDayOfWeek:
      sample = np.concatenate((dataSDR, [sequence['dayofweek'][i]]), axis=0)
    else:
      sample = dataSDR

    netActivation = net.activate(sample)
    predictedDistribution[i, :] = netActivation/sum(netActivation)
    targetDistribution[i, :] = encoder.encode(sequence['data'][i+predictionStep])
    trueData[i] = sequence['data'][i]
    targetInput[i] = sequence['data'][i+predictionStep]
    # print " target input: ", targetDistribution[i], " predicted Input: ", predictedInput[i]

  # calculate negative log-likelihood
  Likelihood = np.multiply(predictedDistribution, targetDistribution)
  Likelihood = np.sum(Likelihood, axis=1)

  minProb = 0.00001
  Likelihood[np.where(Likelihood<minProb)[0]] = minProb
  negLL = -np.log(Likelihood)

  negLLtest = np.mean(negLL[nTrain:])
  print "LSTM, negLL Train %f Test %f" % (np.mean(negLL[:nTrain]), np.mean(negLL[nTrain:]))

  plt.close('all')
  fig = plt.figure(1)
  NT = len(trueData)
  plt.imshow(np.transpose(predictedDistribution), extent=(0, NT, encoder.minval, encoder.maxval),
             interpolation='nearest', aspect='auto', origin='lower', cmap='Reds')

  plt.plot(targetInput, color='black')
  plt.title('LSTM, useTimeOfDay='+str(useTimeOfDay)+' '+dataSet+' test neg LL = '+str(negLLtest))
  plt.xlim([NT-500, NT-predictionStep])
  plt.xlabel('Time')
  plt.ylabel('Prediction')

  # import plotly.plotly as py
  # plot_url = py.plot_mpl(fig)
