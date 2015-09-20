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

def readDataSet(dataSet):
  filePath = 'data/'+dataSet+'.csv'

  if dataSet == 'nyc_taxi':
    df = pd.read_csv(filePath, header=0, skiprows=[1,2], names=['time', 'data', 'timeofday', 'dayofweek'])
    sequence = df['data']
    dayofweek = df['dayofweek']
    timeofday = df['timeofday']

    seq = pd.DataFrame(np.array(pd.concat([sequence, timeofday, dayofweek], axis=1)),
                        columns=['data', 'timeofday', 'dayofweek'])
  elif dataSet == 'sine':
    df = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['time', 'data'])
    sequence = df['data']
    seq = pd.DataFrame(np.array(sequence), columns=['data'])
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

  def __init__(self):
    self.sequence = readDataSet('nyc_taxi')
    self.normalizeSequence()

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

  def generateSequence(self, perturbed=False, startFrom=0, prediction_nstep=5):
    if perturbed:
      # create a new daily profile
      dailyTime = np.sort(self.sequence['timeofday'].unique())
      dailyHour = dailyTime/60
      profile = np.ones((len(dailyTime),))
      # decrease 7am-11am traffic by 20%
      profile[np.where(np.all([dailyHour >= 7.0, dailyHour < 11.0], axis=0))[0]] = 0.8
      # increase 21:00 - 24:00 traffic by 20%
      profile[np.where(np.all([dailyHour >= 21.0, dailyHour <= 23.0], axis=0))[0]] = 1.2
      dailyProfile = {}
      for i in range(len(dailyTime)):
        dailyProfile[dailyTime[i]] = profile[i]

      # apply the new daily pattern to weekday traffic
      old_data = self.sequence['data']
      new_data = np.zeros(old_data.shape)
      for i in xrange(len(old_data)):
        if self.sequence['dayofweek'][i] < 5:
          new_data[i] = old_data[i] * dailyProfile[self.sequence['timeofday'][i]]
        else:
          new_data[i] = old_data[i]

      self.sequence['data'] = new_data
      self.meanSeq = np.mean(self.sequence['data'])
      self.stdSeq = np.std(self.sequence['data'])
      self.sequence.loc[:, 'normalizedData'] = \
        pd.Series((self.sequence['data'] - self.meanSeq)/self.stdSeq, index=self.sequence.index)

    sequence = self.sequence[['normalizedData',
                              'normalizedTimeofday',
                              'normalizedDayofweek']].values.tolist()

    targetInput = self.sequence['data'].values.tolist()

    return (sequence[startFrom:-prediction_nstep], targetInput[startFrom+prediction_nstep:])

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

    if params['dataset'] == 'nyc_taxi':
      self.dataset = NYCTaxiDataset()
    else:
      raise Exception("Dataset not found")

    self.testCounter = 0
    self.net = None
    self.history = []
    self.resets = []

    (self.currentSequence, self.targetPrediction) = self.dataset.generateSequence()


  def window(self, data, params):
    start = max(0, len(data) - params['learning_window'] -1)
    return data[start:]


  def train(self, params, verbose=False):

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

    history = self.window(self.history, params)
    resets = self.window(self.resets, params)
    targetPrediction = self.window(self.targetPrediction, params)

    # prepare a training dataset using the history
    for i in xrange(len(history)):
      if not resets[i-1]:
        ds.addSample(self.inputEncoder.encode(history[i]),
                     self.outputEncoder.encode(targetPrediction[i]))
      if resets[i]:
        ds.newSequence()

    # print ds.getSample(0)
    # print ds.getSample(1)
    # print ds.getSample(1000)
    # print " training data size", ds.getLength(), " len(history) ", len(history), " self.history ", len(self.history)
    # print ds

    if len(history) > 1:
      trainer.trainEpochs(params['num_epochs'])

    # run through the training dataset to get the lstm network state right
    self.net.reset()
    for i in xrange(len(history)):
      output = self.net.activate(ds.getSample(i)[0])

      if resets[i]:
        self.net.reset()


  def iterate(self, params, repetition, iteration, verbose=False):

    if len(self.currentSequence) == 0:
      return None

    self.history.append(self.currentSequence.pop(0))
    self.targetPrediction.append(self.targetPrediction.pop(0))

    if verbose:
      print "iteration: ", iteration, ' history length', len(self.history), ' last ele: ', self.history[-1]

    resetFlag = (len(self.currentSequence) == 0 and
                 params['separate_sequences_with'] == 'reset')
    self.resets.append(resetFlag)


    if iteration < params['compute_after']:
      return None

    train = (not params['compute_test_mode'] or
             iteration % params['train_every'] == 0 or
             iteration == params['train_at_iteration'])

    if train:
      self.train(params, verbose)

    if train:
      # reset test counter after training
      self.testCounter = params['test_for']

    if self.testCounter == 0:
      return None
    else:
      self.testCounter -= 1

    if self.resets[-1]:
      self.net.reset()

    symbol = self.history[-(params['prediction_nstep']+1)]
    output = self.net.activate(self.inputEncoder.encode(symbol))

    if params['output_encoding'] == None:
      predictions = self.dataset.reconstructSequence(output[0])
    elif params['output_encoding'] == 'likelihood':
      predictions = list(output/sum(output))
    else:
      predictions = None

    truth = None if (self.resets[-1]) else self.targetPrediction[-1]

    if iteration == params['perturb_after']:
      (self.currentSequence, self.targetPrediction) = \
        self.dataset.generateSequence(perturbed=True, startFrom=iteration, prediction_nstep=params['prediction_nstep'])

    return {"current": self.history[-1],
            "reset": self.resets[-1],
            "train": train,
            "predictions": predictions,
            "truth": truth}



if __name__ == '__main__':
  suite = Suite()
  suite.start()
