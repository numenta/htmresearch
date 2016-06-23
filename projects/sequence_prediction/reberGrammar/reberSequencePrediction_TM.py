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

from __future__ import print_function
from htmresearch.support.reberGrammar import *
from nupic.encoders import SDRCategoryEncoder
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.ion()
rcParams.update({'figure.autolayout': True})

from nupic.research.monitor_mixin.monitor_mixin_base import MonitorMixinBase

from htmresearch.algorithms.extended_temporal_memory import (
     ExtendedTemporalMemory)

from nupic.research.monitor_mixin.temporal_memory_monitor_mixin import (
    TemporalMemoryMonitorMixin)

class MonitoredFastExtendedTemporalMemory(TemporalMemoryMonitorMixin,
                                         ExtendedTemporalMemory):
  pass


maxLength = 20
numTrainSequence = 50
numTestSequence = 50
rptPerSeq = 5

n = 2048
w = 40
enc = SDRCategoryEncoder(n, w, categoryList)

sdr_dict = dict()
for i in xrange(len(categoryList)):
  sdr = enc.encode(categoryList[i])
  activeCols = set(np.where(sdr)[0])
  sdr_dict[categoryList[i]] = activeCols

def initializeTM():
  DEFAULT_TEMPORAL_MEMORY_PARAMS = {"columnDimensions": (2048,),
                                    "cellsPerColumn": 32,
                                    "activationThreshold": 15,
                                    "initialPermanence": 0.41,
                                    "connectedPermanence": 0.5,
                                    "minThreshold": 10,
                                    "maxNewSynapseCount": 20,
                                    "permanenceIncrement": 0.10,
                                    "permanenceDecrement": 0.02,
                                    "seed": 42}


  params = dict(DEFAULT_TEMPORAL_MEMORY_PARAMS)
  # params.update(tmOverrides or {})
  # params["seed"] = seed
  tm = MonitoredFastExtendedTemporalMemory(mmName="TM", **params)
  return tm


def calculateOverlapWithSDRdict(activeCols, sdr_dict):
  overlaps = []
  for i in xrange(len(categoryList)):
    overlaps.append(len(sdr_dict[categoryList[i]] & activeCols))
  return overlaps


def findMaxOverlap(overlaps):
  maxOverlapLoc = np.argmax(overlaps)
  return categoryList[maxOverlapLoc]


def columnForCells(tm, cells):
  columns = []
  for i in xrange(len(cells)):
    columns.append(tm.columnForCell(cells[i]))
  return set(columns)


def feedSequenceToTM(tm, in_seq, tmLearn):
  for i in xrange(len(in_seq)):
    sequenceLabel = in_seq[i]
    sdr_in = sdr_dict[sequenceLabel]
    tm.compute(sdr_in,
                learn=tmLearn,
                sequenceLabel=sequenceLabel)


def trainTMonReberSeq(tm, numTrainSequence, seedSeq=1):

  np.random.seed(seedSeq)

  tmLearn = 1
  for _ in xrange(numTrainSequence):
    [in_seq, out_seq] = generateSequences(maxLength)
    print("train seq", _, in_seq)
    for _ in xrange(rptPerSeq):
      tm.reset()
      feedSequenceToTM(tm, in_seq, tmLearn)
  return tm


def testTMOnReberSeq(tm, numTestSequence, seedSeq=2):
  np.random.seed(seedSeq)
  tmLearn = 0
  outcomeAll = []

  numStep = 0
  numOutcome = 0
  numPred = 0
  numMiss = 0
  numFP = 0
  for _ in xrange(numTestSequence):
    tm.reset()
    tm.mmClearHistory()

    [in_seq, out_seq] = generateSequences(maxLength)
    print("test seq", _, in_seq)

    feedSequenceToTM(tm, in_seq, tmLearn)

    activeColsTrace = tm._mmTraces['activeColumns'].data
    predictionTrace = tm._mmTraces['predictiveCells'].data
    for i in xrange(len(activeColsTrace)-1):
      # overlap = calculateOverlapWithSDRdict(activeColsTrace[i], sdr_dict)
      # activation = findMaxOverlap(overlap)

      predictedColumns = columnForCells(tm, list(predictionTrace[i]))
      overlap = calculateOverlapWithSDRdict(predictedColumns, sdr_dict)
      prediction = findMaxOverlap(overlap)
      outcome = checkPrediction(out_seq[i], prediction)
      outcomeAll.append(outcome)

      prediction = getMatchingElements(overlap, 20)
      (missN, fpN) = checkPrediction2(out_seq[i], prediction)

      numPred += len(prediction)
      numOutcome += len(out_seq[i])
      numMiss += missN
      numFP += fpN
      numStep += 1

      print("step: ", i, "current input", in_seq[i],
            " possible next elements: ", out_seq[i], " prediction: ", prediction,
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
  tm = initializeTM()
  trainTMonReberSeq(tm, numTrainSequence, seedSeq=train_seed)
  (correctRate, missRate, fpRate) = testTMOnReberSeq(tm, numTestSequence, seedSeq=test_seed)


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
      tm = initializeTM()
      train_seed = 1
      numTrainSequence = trainSeqN[i]
      trainTMonReberSeq(tm, numTrainSequence, seedSeq=train_seed)
      (correctRate, missRate, fpRate) = testTMOnReberSeq(tm, numTestSequence, seedSeq=train_seed+rpt)
      correctRateAll[i, rpt] = correctRate
      missRateAll[i, rpt] = missRate
      fpRateAll[i, rpt] = fpRate

  np.savez('result/reberSequenceTM.npz',
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
  plt.savefig('result/ReberSequence_TMperformance.pdf')


if __name__ == "__main__":
  runExperiment()

  # uncomment to run a single experiment
  # runSingleExperiment(5)
