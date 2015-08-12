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

from reberGrammar import *
from HMM import HMM

import matplotlib.pyplot as plt
import numpy as np

from copy import copy
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
plt.ion()

from sys import stdout


MAXLENGTH = 20
numTestSequence = 50
numStates = 2
numCats = len(chars)
charsToInts = dict(zip(chars, range(numCats)))


def initializeHMM(numStates=numStates, possibleObservations=numCats):

  print "Initializing HMM..."
  hmm = HMM(numStates=numStates, numCats=possibleObservations)
  hmm.pi = np.random.rand(numStates)
  hmm.pi /= sum(hmm.pi)

  hmm.A = np.random.rand(numStates, numStates)
  A_row_sums = hmm.A.sum(axis=1)
  hmm.A /= A_row_sums[:, np.newaxis]

  hmm.B = np.random.rand(numStates, numCats)
  B_row_sums = hmm.B.sum(axis=1)
  hmm.B /= B_row_sums[:, np.newaxis]

  print "Initial HMM stats"
  print "A: ",
  print hmm.A
  print "B: ",
  print hmm.B
  print "pi: ",
  print hmm.pi

  return hmm


def learnHMM(numTrainSequence, numStates=numStates):

  hmm = initializeHMM(numStates)

  for i in range(numTrainSequence):
    sample, _ = generateSequences(MAXLENGTH)
    sampleInts = np.array([charsToInts[c] for c in sample])
    hmm.train(sampleInts)

  print "HMM stats"
  print "A: ",
  print hmm.A
  print "B: ",
  print hmm.B
  print "pi: ",
  print hmm.pi

  return hmm


def testHMM(hmm, numTestSequence=numTestSequence):

  outcomeAll = []

  numOutcome = 0.0
  numPred = 0.0
  numMiss = 0.0
  numFP = 0.0
  numStep = 0.0

  for _ in range(numTestSequence):
    sample, target = generateSequences(MAXLENGTH)
    hmm.reset()
    for i in range(len(sample)):
      current_input = charsToInts[sample[i]]
      possible_next_inputs = set(np.array([charsToInts[c] for c in target[i]]))
      predicted_next_inputs = hmm.predict_next_inputs(current_input)

      # fraction of predicted inputs that were in possible outputs
      numPreds = 1.0*len(predicted_next_inputs)
      if numPreds > 0:
        outcome = len(predicted_next_inputs & possible_next_inputs) / numPreds
      else:
        outcome = 0
      outcomeAll.append(outcome)

      # missN is number of possible outcomes not predicted
      missN = len(possible_next_inputs - predicted_next_inputs)

      #fpN is number of predicted outcomes not possible
      fpN = len(predicted_next_inputs - possible_next_inputs)

      numPred += numPreds
      numOutcome += len(target)
      numMiss += missN
      numFP += fpN
      numStep += 1

  correctRate = sum(outcomeAll)/float(len(outcomeAll))
  missRate = float(numMiss)/float(numStep * 7)
  fpRate = float(numFP)/float(numStep * 7)
  errRate = float(numMiss + numFP)/float(numStep * 7)

  print("Correct Rate (Best Prediction): ", correctRate)
  print("Error Rate: ", errRate)
  print("Miss Rate: ", missRate)
  print("False Positive Rate: ", fpRate)
  return correctRate, missRate, fpRate

def runSingleExperiment(numTrainSequence, numTestSequence=numTestSequence):
  hmm = learnHMM(numTrainSequence)
  return testHMM(hmm, numTestSequence)


def runExperiment():
  """
  Experiment 1: Calculate error rate as a function of training sequence numbers
  :return:
  """
  trainSeqN = [5, 10, 20, 50, 100, 200]
  rptPerCondition = 20
  correctRateAll = np.zeros((len(trainSeqN), rptPerCondition))
  missRateAll = np.zeros((len(trainSeqN), rptPerCondition))
  fpRateAll = np.zeros((len(trainSeqN), rptPerCondition))
  for i in xrange(len(trainSeqN)):
    for rpt in xrange(rptPerCondition):
      numTrainSequence = trainSeqN[i]
      correctRate, missRate, fpRate = runSingleExperiment(numTrainSequence=numTrainSequence)
      correctRateAll[i, rpt] = correctRate
      missRateAll[i, rpt] = missRate
      fpRateAll[i, rpt] = fpRate

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
  plt.savefig('result/ReberSequence_HMMperformance.pdf')
  plt.show()

if __name__ == "__main__":
  runExperiment()