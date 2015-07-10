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

import numpy as np
import random

from collections import defaultdict
from matplotlib import pyplot as plt

from nupic.data.generators.pattern_machine import PatternMachine
from nupic.data.generators.sequence_machine import SequenceMachine
from sensorimotor.general_temporal_memory import GeneralTemporalMemory
from nupic.research.monitor_mixin.temporal_memory_monitor_mixin import (
    TemporalMemoryMonitorMixin)



class MonitoredGeneralTemporalMemory(TemporalMemoryMonitorMixin,
                   GeneralTemporalMemory):
  pass



DEFAULT_TEMPORAL_MEMORY_PARAMS = {"columnDimensions": (2048,),
                                  "cellsPerColumn": 20,
                                  "activationThreshold": 20,
                                  "initialPermanence": 0.5,
                                  "connectedPermanence": 0.6,
                                  "minThreshold": 20,
                                  "maxNewSynapseCount": 30,
                                  "permanenceIncrement": 0.10,
                                  "permanenceDecrement": 0.02,
                                  "seed": 42,
                                  "learnOnOneCell": False}

params = dict(DEFAULT_TEMPORAL_MEMORY_PARAMS)
tmNoFeedback = MonitoredGeneralTemporalMemory(mmName="TM1", **params)
tmFeedback = MonitoredGeneralTemporalMemory(mmName="TM2", **params)
feedback_n = 200
trials = 30


def generateSequences(patternDimensionality, patternCardinality, sequenceLength,
                      sequenceCount):
  patternAlphabetSize = sequenceLength * sequenceCount
  patternMachine = PatternMachine(patternDimensionality, patternCardinality,
                                  patternAlphabetSize)
  sequenceMachine = SequenceMachine(patternMachine)
  numbers = sequenceMachine.generateNumbers(sequenceCount, sequenceLength)
  generatedSequences = sequenceMachine.generateFromNumbers(numbers)

  return generatedSequences


def serializePattern(patternSet):
  return ','.join(sorted([str(x) for x in patternSet]))


def getAlphabet(sequences):
  alphabetOfPatterns = []
  count = 0
  alphabet = {}

  for sensorPattern in sequences:
    if sensorPattern is not None:
      ser = serializePattern(sensorPattern)
      if ser not in alphabet:
        alphabet[ser] = chr(count + ord('A'))
        count += 1

  return alphabet

def labelPattern(pattern, alphabet):
  if pattern is None:
    return None
  ser = serializePattern(pattern)
  if ser in alphabet:
    return alphabet[ser]
  return '?'

def labelSequences(sequences, alphabet=None):
  if alphabet is None:
    alphabet = getAlphabet(sequences)

  labels = []

  for sensorPattern in sequences:
    label = labelPattern(sensorPattern, alphabet)
    if label is not None:
      labels.append(label)

  return labels

def shiftingFeedback(starting_feedback, n, percent_shift=0.02):

  feedback_seq = []
  feedback = starting_feedback

  for _ in range(n):
    feedback = set([x for x in feedback])
    p = int(percent_shift*len(feedback))
    toRemove = set(random.sample(feedback, p))
    toAdd = set([random.randint(0, 2048) for _ in range(p)])
    feedback = (feedback - toRemove) | toAdd
    feedback_seq.append(feedback)

  return feedback_seq

def train(tm, sequences, feedback_seq=None, trials=trials,
          feedback_buffer=10, clearhistory=True):

  for i in range(trials):
    for j, sensorPattern in enumerate(sequences):
      if sensorPattern is None:
        tm.reset()
      else:
        if i<feedback_buffer:
          feedback = set([random.randint(0, 2048) for _ in range(feedback_n)])
        elif feedback_seq is not None:
          feedback = feedback_seq[j]
        else:
          feedback = set()
        tm.compute(sensorPattern, activeApicalCells=feedback,
                   learn=True, sequenceLabel=None)

    if clearhistory:
      tm.mmClearHistory()


def run(tm, mutate_times, sequences, alphabet, feedback_seq=None, mutation=0):

  allLabels = []

  for j, sensorPattern in enumerate(sequences):
    print "Pattern ", j
    if sensorPattern is None:
      tm.reset()
    else:
      if j in mutate_times:
        if mutation:
          continue
        else:
          sensorPattern = set([random.randint(0, 2047) for _ in sensorPattern])

      if feedback_seq is not None:
        feedback = feedback_seq[j]
        print "feedback: ", feedback
      else:
        feedback = set()

      tm.compute(sensorPattern, activeApicalCells=feedback,
                 learn=True, sequenceLabel=None)

      allLabels.append(labelPattern(sensorPattern, alphabet))


  ys = [len(x) for x in tm.mmGetTraceUnpredictedActiveColumns().data]

  return ys, allLabels


def plotResults(ys1, ys2, allLabels, title=None):

  fig, ax = plt.subplots()

  index = np.arange(len(ys1))

  bar_width = 0.35

  opacity = 0.4

  rects1 = plt.bar(index, ys1, bar_width,
                   alpha=opacity,
                   color='b',
                   label='No Feedback')

  rects2 = plt.bar(index+bar_width, ys2, bar_width,
                   alpha=opacity,
                   color='r',
                   label='Feedback')

  plt.xlabel('Sequence input over time')
  plt.ylabel('# Bursting Columns')
  if title:
    plt.title(title)
  plt.xticks(index + bar_width, allLabels)
  plt.legend(loc='lower right')

  plt.tight_layout()
  plt.show()


def plotPredictionAccuracy(tm1, tm2, allLabels, title=None):
  total_preds1 = [len(x) for x in tm1.mmGetTracePredictiveCells().data]
  accurate_preds1 = [len(x) for x in tm1.mmGetTracePredictedActiveCells().data]
  total_preds2 = [len(x) for x in tm2.mmGetTracePredictiveCells().data]
  accurate_preds2 = [len(x) for x in tm2.mmGetTracePredictedActiveCells().data]

  xs = range(1, len(accurate_preds1))
  ys1 = []
  ys2 = []

  for i in xs:
    if total_preds1[i-1]:
      y1 = (1.0*accurate_preds1[i])/(total_preds1[i-1])
    else:
      y1 = 0.0
    if total_preds2[i-1]:
      y2 = (1.0*accurate_preds2[i])/(total_preds2[i-1])
    else:
      y2 = 0.0
    ys1.append(y1)
    ys2.append(y2)

  index = np.arange(len(ys1))

  bar_width = 0.35
  opacity = 0.4

  rects1 = plt.bar(index, ys1, bar_width,
                   alpha=opacity,
                   color='b',
                   label='No Feedback')

  rects2 = plt.bar(index+bar_width, ys2, bar_width,
                   alpha=opacity,
                   color='r',
                   label='Feedback')

  plt.xlabel('Sequence input over time')
  plt.ylabel('Percent Accuracy')
  if title:
    plt.title(title)
  plt.xticks(index + bar_width, allLabels)
  plt.legend(loc='lower right')

def experiment1(aorb='a', mutate_times = [3]):

  if aorb =='a':
    mutation = 0
  elif aorb =='b':
    mutation = 1
  else:
    raise ValueError

  sequences = generateSequences(2048, 20, 5, 1)
  alphabet = getAlphabet(sequences)

  fixed_feedback = set([random.randint(0, 2048) for _ in range(feedback_n)])
  feedback_seq = shiftingFeedback(fixed_feedback, len(sequences))

  train(tmNoFeedback, sequences)
  train(tmFeedback, sequences, feedback_seq)

  ys1, allLabels = run(tmNoFeedback, mutate_times, sequences, alphabet, mutation=mutation)
  ys2, _ = run(tmFeedback, mutate_times, sequences, alphabet, feedback_seq, mutation)

  plotPredictionAccuracy(tmNoFeedback, tmFeedback, allLabels)
  plotResults(ys1, ys2, allLabels)


def experiment2(aorb='a'):
  sequences1 = generateSequences(2048, 20, 5, 1)

  sequences2 = [x for x in sequences1]
  sequences2[0] = set([random.randint(0, 2047) for _ in sequences1[0]])
  sequences2[-2] = set([random.randint(0, 2047) for _ in sequences1[-2]])

  fixed_feedback1 = set([random.randint(0, 2048) for _ in range(feedback_n)])
  fixed_feedback2 = set([random.randint(0, 2048) for _ in range(feedback_n)])
  feedback_seq1 = shiftingFeedback(fixed_feedback1, len(sequences1))
  feedback_seq2 = shiftingFeedback(fixed_feedback2, len(sequences2))

  sequences = sequences1+sequences2
  feedback_seq = feedback_seq1+feedback_seq2

  alphabet = getAlphabet(sequences)

  train(tmNoFeedback, sequences)
  train(tmFeedback, sequences, feedback_seq)

  partial_sequences1 = sequences1[1:]

  if aorb == 'a':
    testFeedback = feedback_seq1
    title = 'UP is in "ABCDE" state'
  elif aorb == 'b':
    testFeedback = feedback_seq2
    title = 'UP is in "XBCDF" state'
  else:
    raise ValueError

  ys1, allLabels = run(tmNoFeedback, defaultdict(list), partial_sequences1,
                       alphabet)

  ys2, _ = run(tmFeedback, defaultdict(list), partial_sequences1, alphabet,
               feedback_seq=testFeedback)

  print allLabels

  plotPredictionAccuracy(tmNoFeedback, tmFeedback, allLabels, title)
  plotResults(ys1, ys2, allLabels, title)

def experiment3():
  sequences1 = generateSequences(2048, 20, 5, 1)

  sequences2 = [x for x in sequences1]
  sequences2[-2] = sequences1[1]

  fixed_feedback = set([random.randint(0, 2048) for _ in range(feedback_n)])
  feedback_seq = shiftingFeedback(fixed_feedback, len(sequences1))

  alphabet = getAlphabet(sequences1)

  train(tmNoFeedback, sequences1)
  train(tmFeedback, sequences1, feedback_seq)

  ys1, allLabels = run(tmNoFeedback, defaultdict(list), sequences2,
                       alphabet)

  ys2, _ = run(tmFeedback, defaultdict(list), sequences2, alphabet,
               feedback_seq=feedback_seq)

  title = 'Feedback is in "ABCDE" state'

  plotPredictionAccuracy(tmNoFeedback, tmFeedback, allLabels, title)
  plotResults(ys1, ys2, allLabels, title)


if __name__ == "__main__":
  experiment1('a')
  #experiment2('b')
  #experiment3()t3()