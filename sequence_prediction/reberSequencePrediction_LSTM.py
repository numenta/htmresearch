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

from __future__ import print_function
from reberGrammar import *

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.ion()

from pybrain.datasets import SequentialDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer
from pybrain.supervised import RPropMinusTrainer

from sys import stdout


maxLength = 20
numTrainSequence = 50
numTestSequence = 50
rptPerSeq = 5
categoryList = ['B', 'T', 'S', 'X', 'P', 'V', 'E']

def initializeLSTMnet():
  # Build LSTM network with 7 input units, 8 hidden units (LSTM cells) and 7 output cells
  net = buildNetwork(7, 8, 7,
                     hiddenclass=LSTMLayer, bias=True, outputbias=False, recurrent=True)
  return net


def getReberDS(maxLength, display = 0):
  """
  @param maxLength (int): maximum length of the sequence
  """
  [in_seq, out_seq] = generateSequencesVector(maxLength)

  target = out_seq
  last_target = target[-1]
  last_target[np.argmax(out_seq[-1])] = 1
  target[-1] = last_target

  ds = SequentialDataSet(7, 7)
  i = 0
  for sample, next_sample in zip(in_seq, target):
    ds.addSample(sample, next_sample)
    if display:
      print("     sample: %s" % sample)
      print("     target: %s" % next_sample)
      print("next sample: %s" % out_seq[i])
      print()
    i += 1

  return (ds, in_seq, out_seq)


def getMaxActivation(activation):
  activation_round = np.zeros(activation.shape)
  activation_round[np.argmax(activation)] = 1
  return sequenceToWord([activation_round])


def getElementFromVec(out_seq_next):
  elements = []
  non_zero_indx = np.where(out_seq_next)[0]
  for i in xrange(len(non_zero_indx)):
    elements.append(categoryList[non_zero_indx[i]])

  return elements


def trainLSTMnet(net, numTrainSequence, seedSeq=1):
  np.random.seed(seedSeq)
  for _ in xrange(numTrainSequence):
    (ds, in_seq, out_seq) = getReberDS(maxLength)
    print("train seq", _, sequenceToWord(in_seq))
    trainer = RPropMinusTrainer(net, dataset=ds)
    trainer.trainEpochs(rptPerSeq)

  return net
  # print("final error =", train_errors[-1])


def testLSTMnet(net, numTestSequence, seedSeq=2):
  np.random.seed(seedSeq)
  outcomeAll = []

  numOutcome = 0
  numPred = 0
  numMiss = 0
  numFP = 0
  numStep = 0
  for _ in xrange(numTestSequence):
    net.reset()
    (ds, in_seq, out_seq) = getReberDS(maxLength)
    print("test seq", _, sequenceToWord(in_seq))

    for i in xrange(len(in_seq)-1):
      sample = in_seq[i]
      target = out_seq[i]

      currentInput = sequenceToWord([sample])
      netActivation = net.activate(sample)
      predictNextInput = getMaxActivation(netActivation)

      possibleNextInput = getElementFromVec(target)

      outcome = checkPrediction(possibleNextInput, predictNextInput)
      outcomeAll.append(outcome)

      prediction = getMatchingElements(netActivation, .5)
      (missN, fpN) = checkPrediction2(possibleNextInput, prediction)

      numPred += len(prediction)
      numOutcome += len(target)
      numMiss += missN
      numFP += fpN
      numStep += 1

      print("step: ", i, "current input", currentInput,
            " possible next elements: ", possibleNextInput, " prediction: ", prediction,
            " outcome: ", outcome, "Miss: ", missN, "FP: ", fpN)

  correctRate = sum(outcomeAll)/float(len(outcomeAll))
  missRate = float(numMiss)/float(numStep * 7)
  fpRate = float(numFP)/float(numStep * 7)
  errRate = float(numMiss + numFP)/float(numStep * 7)

  print("Correct Rate (Best Prediction): ", correctRate)
  print("Error Rate: ", errRate)
  print("Miss Rate: ", missRate)
  print("False Positive Rate: ", fpRate)
  return correctRate, missRate, fpRate


def runSingleExperiment(numTrainSequence, train_seed=1, test_seed=2):
  net = initializeLSTMnet()
  net = trainLSTMnet(net, numTrainSequence, seedSeq=train_seed)
  (correctRate, missRate, fpRate) = testLSTMnet(net, numTestSequence, seedSeq=test_seed)


def runExperiment():
  """
  Experiment 1: Calculate error rate as a function of training sequence numbers
  :return:
  """
  trainSeqN = [5, 10, 20, 50, 100, 200]
  rptPerCondition = 5
  correctRateAll = np.zeros((len(trainSeqN), rptPerCondition))
  missRateAll = np.zeros((len(trainSeqN), rptPerCondition))
  fpRateAll = np.zeros((len(trainSeqN), rptPerCondition))
  for i in xrange(len(trainSeqN)):
    for rpt in xrange(rptPerCondition):
      train_seed = 1
      numTrainSequence = trainSeqN[i]
      net = initializeLSTMnet()
      net = trainLSTMnet(net, numTrainSequence, seedSeq=train_seed)
      (correctRate, missRate, fpRate) = testLSTMnet(net, numTestSequence, seedSeq=train_seed+rpt)
      correctRateAll[i, rpt] = correctRate
      missRateAll[i, rpt] = missRate
      fpRateAll[i, rpt] = fpRate

  np.savez('result/reberSequenceLSTM.npz',
           correctRateAll=correctRateAll, missRateAll=missRateAll,
           fpRateAll=fpRateAll, trainSeqN=trainSeqN)

  plt.figure()
  plt.subplot(2,2,1)
  plt.semilogx(trainSeqN, 100*np.mean(correctRateAll,1),'-*')
  plt.xlabel(' Training Sequence Number')
  plt.ylabel(' Hit Rate - Best Match (%)')
  plt.subplot(2,2,2)
  plt.semilogx(trainSeqN, 100*np.mean(missRateAll,1),'-*')
  plt.xlabel(' Training Sequence Number')
  plt.ylabel(' Miss Rate (%)')
  plt.subplot(2,2,3)
  plt.semilogx(trainSeqN, 100*np.mean(fpRateAll,1),'-*')
  plt.xlabel(' Training Sequence Number')
  plt.ylabel(' False Positive Rate (%)')
  plt.savefig('result/ReberSequence_LSTMperformance.pdf')


if __name__ == "__main__":
  runExperiment()

  # uncomment to run a single experiment
  # runSingleExperiment(20)