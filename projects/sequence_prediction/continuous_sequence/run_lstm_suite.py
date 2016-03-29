#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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



from expsuite import PyExperimentSuite

from pybrain.datasets import SequentialDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer
from pybrain.supervised import RPropMinusTrainer
from pybrain.structure.modules import SigmoidLayer

from nupic.encoders.scalar import ScalarEncoder as NupicScalarEncoder

import pandas as pd
import numpy as np
from scipy import random

def readDataSet(dataSet, noise=0):
  """
  :param dataSet: dataset name
  :param noise: amount of noise added to the dataset
  :return:
  """
  filePath = 'data/'+dataSet+'.csv'

  if dataSet == 'nyc_taxi' or dataSet == 'nyc_taxi_perturb' or dataSet == 'nyc_taxi_perturb_baseline':
    seq = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['time', 'data', 'timeofday', 'dayofweek'])
    seq['time'] = pd.to_datetime(seq['time'])

    if noise > 0:
      for i in xrange(len(seq)):
        value = seq['data'][i]
        noiseValue = np.random.normal(scale=(value * noise))
        value += noiseValue
        value = max(0, value)
        value = min(40000, value)
        seq['data'][i] = value
  # elif dataSet == 'sine':
  #   df = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['time', 'data'])
  #   sequence = df['data']
  #   seq = pd.DataFrame(np.array(sequence), columns=['data'])
  else:
    raise(' unrecognized dataset type ')

  return seq


class Encoder(object):

  def __init__(self):
    pass

  def encode(self, symbol):
    pass


  def random(self):
    pass


  def classify(self, encoding, num=1):
    pass



class PassThroughEncoder(Encoder):

  def encode(self, symbol):
    return symbol


class ScalarBucketEncoder(Encoder):

  def __init__(self):
    self.encoder = NupicScalarEncoder(w=1, minval=0, maxval=40000, n=22, forced=True)

  def encode(self, symbol):
    encoding = self.encoder.encode(symbol)
    return encoding

class Dataset(object):

  def generateSequence(self):
    pass

  def reconstructSequence(self, data):
    pass



class NYCTaxiDataset(Dataset):

  def __init__(self, dataset='nyc_taxi'):
    self.sequence_name = dataset


  def normalizeSequence(self):
    # standardize data by subtracting mean and dividing by std
    self.meanSeq = np.mean(self.sequence['data'])
    self.stdSeq = np.std(self.sequence['data'])
    self.sequence.loc[:, 'normalizedData'] = \
      pd.Series((self.sequence['data'] - self.meanSeq)/self.stdSeq, index=self.sequence.index)

    self.meanTimeOfDay = np.mean(self.sequence['timeofday'])
    self.stdTimeOfDay = np.std(self.sequence['timeofday'])
    self.sequence.loc[:, 'normalizedTimeofday'] = \
      pd.Series((self.sequence['timeofday'] - self.meanTimeOfDay)/self.stdTimeOfDay, index=self.sequence.index)

    self.meanDayOfWeek = np.mean(self.sequence['dayofweek'])
    self.stdDayOfWeek = np.std(self.sequence['dayofweek'])
    self.sequence.loc[:, 'normalizedDayofweek'] = \
      pd.Series((self.sequence['dayofweek'] - self.meanDayOfWeek)/self.stdDayOfWeek, index=self.sequence.index)


  def generateSequence(self, perturbed=False, prediction_nstep=5,
                       output_encoding=None, noise=0):
    if perturbed:
      self.sequence = readDataSet(self.sequence_name+'_perturb', noise)
    else:
      print "read dataset", self.sequence_name
      self.sequence = readDataSet(self.sequence_name, noise)

    self.normalizeSequence()
      #
      # # create a new daily profile
      # dailyTime = np.sort(self.sequence['timeofday'].unique())
      # dailyHour = dailyTime/60
      # profile = np.ones((len(dailyTime),))
      # # decrease 7am-11am traffic by 20%
      # profile[np.where(np.all([dailyHour >= 7.0, dailyHour < 11.0], axis=0))[0]] = 0.8
      # # increase 21:00 - 24:00 traffic by 20%
      # profile[np.where(np.all([dailyHour >= 21.0, dailyHour <= 23.0], axis=0))[0]] = 1.2
      # dailyProfile = {}
      # for i in range(len(dailyTime)):
      #   dailyProfile[dailyTime[i]] = profile[i]
      #
      # # apply the new daily pattern to weekday traffic
      # old_data = self.sequence['data']
      # new_data = np.zeros(old_data.shape)
      # for i in xrange(len(old_data)):
      #   if self.sequence['dayofweek'][i] < 5:
      #     new_data[i] = old_data[i] * dailyProfile[self.sequence['timeofday'][i]]
      #   else:
      #     new_data[i] = old_data[i]
      #
      # self.sequence['data'] = new_data
      # self.meanSeq = np.mean(self.sequence['data'])
      # self.stdSeq = np.std(self.sequence['data'])
      # self.sequence.loc[:, 'normalizedData'] = \
      #   pd.Series((self.sequence['data'] - self.meanSeq)/self.stdSeq, index=self.sequence.index)

    networkInput = self.sequence[['normalizedData',
                              'normalizedTimeofday',
                              'normalizedDayofweek']].values.tolist()

    if output_encoding == None:
      targetPrediction = self.sequence['normalizedData'].values.tolist()
    elif output_encoding == 'likelihood':
      targetPrediction = self.sequence['data'].values.tolist()
    else:
      raise Exception("unrecognized output encoding type")

    trueData = self.sequence['data'].values.tolist()

    return (networkInput[:-prediction_nstep],
            targetPrediction[+prediction_nstep:],
            trueData[+prediction_nstep:])


  def reconstructSequence(self, data):
    return data * self.stdSeq + self.meanSeq


class Suite(PyExperimentSuite):

  def reset(self, params, repetition):
    print params

    self.nDimInput = 3
    self.inputEncoder = PassThroughEncoder()

    if params['output_encoding'] == None:
      self.outputEncoder = PassThroughEncoder()
      self.nDimOutput = 1
    elif params['output_encoding'] == 'likelihood':
      self.outputEncoder = ScalarBucketEncoder()
      self.nDimOutput = self.outputEncoder.encoder.n

    if (params['dataset'] == 'nyc_taxi' or
            params['dataset'] == 'nyc_taxi_perturb_baseline'):
      self.dataset = NYCTaxiDataset(params['dataset'])
    else:
      raise Exception("Dataset not found")

    self.testCounter = 0
    self.resets = []
    self.iteration = 0

    # initialize LSTM network
    random.seed(6)
    if params['output_encoding'] == None:
      self.net = buildNetwork(self.nDimInput, params['num_cells'], self.nDimOutput,
                         hiddenclass=LSTMLayer, bias=True, outputbias=True, recurrent=True)
    elif params['output_encoding'] == 'likelihood':
      self.net = buildNetwork(self.nDimInput, params['num_cells'], self.nDimOutput,
                         hiddenclass=LSTMLayer, bias=True, outclass=SigmoidLayer, recurrent=True)

    (self.networkInput, self.targetPrediction, self.trueData) = \
      self.dataset.generateSequence(
      prediction_nstep=params['prediction_nstep'],
      output_encoding=params['output_encoding'],
      noise=params['noise'])


  def window(self, data, params):
    start = max(0, self.iteration - params['learning_window'])
    return data[start:self.iteration]


  def train(self, params, verbose=False):

    if params['create_network_before_training']:
      if verbose:
        print 'create lstm network'

      random.seed(6)
      if params['output_encoding'] == None:
        self.net = buildNetwork(self.nDimInput, params['num_cells'], self.nDimOutput,
                           hiddenclass=LSTMLayer, bias=True, outputbias=True, recurrent=True)
      elif params['output_encoding'] == 'likelihood':
        self.net = buildNetwork(self.nDimInput, params['num_cells'], self.nDimOutput,
                           hiddenclass=LSTMLayer, bias=True, outclass=SigmoidLayer, recurrent=True)

    self.net.reset()

    ds = SequentialDataSet(self.nDimInput, self.nDimOutput)
    trainer = RPropMinusTrainer(self.net, dataset=ds, verbose=verbose)

    networkInput = self.window(self.networkInput, params)
    targetPrediction = self.window(self.targetPrediction, params)

    # prepare a training data-set using the history
    for i in xrange(len(networkInput)):
      ds.addSample(self.inputEncoder.encode(networkInput[i]),
                   self.outputEncoder.encode(targetPrediction[i]))
    if verbose:
      print " train LSTM on ", len(ds), " records for ", params['num_epochs'], " epochs "

    if len(networkInput) > 1:
      trainer.trainEpochs(params['num_epochs'])

    # run through the training dataset to get the lstm network state right
    self.net.reset()
    for i in xrange(len(networkInput)):
      self.net.activate(ds.getSample(i)[0])


  def iterate(self, params, repetition, iteration, verbose=True):
    self.iteration = iteration

    if self.iteration >= len(self.networkInput):
      return None

    train = False
    if iteration > params['compute_after']:
      if iteration == params['train_at_iteration']:
        train = True

      if params['train_every_month']:
        train = (self.dataset.sequence['time'][iteration].is_month_start and
                  self.dataset.sequence['time'][iteration].hour == 0 and
                  self.dataset.sequence['time'][iteration].minute == 0)

      if params['train_every_week']:
        train = (self.dataset.sequence['time'][iteration].dayofweek==0 and
                  self.dataset.sequence['time'][iteration].hour == 0 and
                  self.dataset.sequence['time'][iteration].minute == 0)

    if verbose:
      print
      print "iteration: ", iteration, " time: ", self.dataset.sequence['time'][iteration]

    if train:
      if verbose:
        print " train at", iteration, " time: ", self.dataset.sequence['time'][iteration]
      self.train(params, verbose)

    if train:
      # reset test counter after training
      self.testCounter = params['test_for']

    if self.testCounter == 0:
      return None
    else:
      self.testCounter -= 1

    symbol = self.networkInput[iteration]
    output = self.net.activate(self.inputEncoder.encode(symbol))

    if params['output_encoding'] == None:
      predictions = self.dataset.reconstructSequence(output[0])
    elif params['output_encoding'] == 'likelihood':
      predictions = list(output/sum(output))
    else:
      predictions = None

    if verbose:
      print " test at :", iteration,

    if iteration == params['perturb_after']:
      if verbose:
        print " perturb data and introduce new patterns"

      (newNetworkInput, newTargetPrediction, newTrueData) = \
        self.dataset.generateSequence(perturbed=True,
                                      prediction_nstep=params['prediction_nstep'],
                                      output_encoding=params['output_encoding'],
                                      noise=params['noise'])

      self.networkInput[iteration+1:] = newNetworkInput[iteration+1:]
      self.targetPrediction[iteration+1:] = newTargetPrediction[iteration+1:]
      self.trueData[iteration+1:] = newTrueData[iteration+1:]

    return {"current": self.networkInput[iteration],
            "reset": None,
            "train": train,
            "predictions": predictions,
            "truth": self.trueData[iteration]}



if __name__ == '__main__':
  suite = Suite()
  suite.start()
