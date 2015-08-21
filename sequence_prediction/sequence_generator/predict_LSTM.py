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

import operator
import random
import time

from matplotlib import pyplot as plt
import numpy as np

from pybrain.datasets import SequentialDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer
from pybrain.supervised import RPropMinusTrainer

from predict import generateSequences
from plot import plotAccuracy



NUM_PREDICTIONS = 1



def num2vec(activeBits, nDim):
  sample = np.zeros((1,nDim))
  sample[0][activeBits] = 1
  return sample

def seq2vec(sequence, nDim):
  nSample = len(sequence)
  seq_vec = np.zeros((nSample, nDim))
  for i in xrange(nSample):
    seq_vec[i][sequence[i]] = 1
  return seq_vec


def initializeLSTMnet(nDim, nLSTMcells=10):
  # Build LSTM network with nDim input units, nLSTMcells hidden units (LSTM cells) and nDim output cells
  net = buildNetwork(nDim, nLSTMcells, nDim,
                     hiddenclass=LSTMLayer, bias=True, outputbias=False, recurrent=True)
  return net



if __name__ == "__main__":
  sequences = generateSequences(NUM_PREDICTIONS)
  nDim = max([len(sequence) for sequence in sequences]) + 2  # TODO: Why 2?

  from pylab import rcParams
  rcParams.update({'figure.autolayout': True})
  rcParams.update({'figure.facecolor': 'white'})
  rcParams.update({'ytick.labelsize': 8})

  # Batch training mode
  print "generate a dataset of sequences"
  ds = SequentialDataSet(nDim, nDim)
  for i in xrange(len(sequences)):
    sequence = sequences[i]
    print sequence
    seq_vec = seq2vec(sequence, nDim)
    for j in xrange(len(sequence)-1):
      ds.addSample(seq_vec[j], seq_vec[j+1])
    ds.newSequence()

  rptPerSeqList = [1, 2, 5, 10, 20, 50, 100, 150, 175]
  accuracyList = []
  for rptNum in rptPerSeqList:
    # train LSTM
    net = initializeLSTMnet(nDim, nLSTMcells=20)
    net.reset()
    trainer = RPropMinusTrainer(net, dataset=ds)
    trainer.trainEpochs(rptNum)

    print "test LSTM"
    # test LSTM
    correct = []
    for i in xrange(len(sequences)):
      sequence = sequences[i]
      print sequence
      net.reset()
      predictedInput = []
      for j in xrange(len(sequence)-1):
        sample = num2vec(sequence[j], nDim)
        netActivation = net.activate(sample)
        predictedInput.append(np.argmax(netActivation))
        print " actual input: ", sequence[j+1], " predicted Input: ", predictedInput[j]

      correct.append(predictedInput[-1] == sequence[-1])

    accuracyList.append(sum(correct)/float(len(correct)))
    print "Accuracy: ", accuracyList[-1]

  plt.semilogx(np.array(rptPerSeqList), np.array(accuracyList), '-*')
  plt.xlabel(' Repeat of entire batch')
  plt.ylabel(' Accuracy ')

  plt.show()

  # online mode (does not work well)
  # net = initializeLSTMnet(nDim, nLSTMcells=20)
  # accuracyList = []
  # for seq in xrange(5000):
  #   sequence = random.choice(sequences)
  #   print sequence
  #   seq_vec = seq2vec(sequence, nDim)
  #   ds = SequentialDataSet(nDim, nDim)
  #   for j in xrange(len(sequence)-1):
  #     ds.addSample(seq_vec[j], seq_vec[j+1])
  #
  #   # test LSTM
  #   net.reset()
  #   predictedInput = []
  #   for i in xrange(len(sequence)-1):
  #     sample = num2vec(sequence[i], nDim)
  #     netActivation = net.activate(sample)
  #     predictedInput.append(np.argmax(netActivation))
  #     print " predicted Input: ", predictedInput[i], " actual input: ", sequence[i+1]
  #
  #   accuracyList.append(predictedInput[-1] == sequence[-1])
  #
  #   # train LSTM
  #   net.reset()
  #   trainer = RPropMinusTrainer(net, dataset=ds)
  #   trainer.trainEpochs(1)
  #
  #   # test LSTM on the whole dataset
  #   # correct = []
  #   # for i in xrange(len(sequences)):
  #   #   sequence = sequences[i]
  #   #   print sequence
  #   #   net.reset()
  #   #   predictedInput = []
  #   #   for j in xrange(len(sequence)-1):
  #   #     sample = num2vec(sequence[j], nDim)
  #   #     netActivation = net.activate(sample)
  #   #     predictedInput.append(np.argmax(netActivation))
  #   #     print " actual input: ", sequence[j+1], " predicted Input: ", predictedInput[j]
  #   #
  #   #   correct.append(predictedInput[-1] == sequence[-1])
  #   # accuracyList.append(sum(correct)/float(len(correct)))
  #
  #   if seq % 100 == 0:
  #       rcParams.update({'figure.figsize': (12, 6)})
  #       plt.figure(1)
  #       plt.clf()
  #       plotAccuracy(accuracyList)
  #       plt.draw()

