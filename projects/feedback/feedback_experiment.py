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

import numpy as np
import random

from collections import defaultdict
from matplotlib import pyplot as plt

from nupic.bindings.math import GetNTAReal
from nupic.data.generators.pattern_machine import PatternMachine
from nupic.data.generators.sequence_machine import SequenceMachine
from htmresearch.algorithms.general_temporal_memory import GeneralTemporalMemory
from nupic.research.monitor_mixin.temporal_memory_monitor_mixin import (
    TemporalMemoryMonitorMixin)
from htmresearch.algorithms.union_temporal_pooler import UnionTemporalPooler
from nupic.research.monitor_mixin.monitor_mixin_base import MonitorMixinBase



class MonitoredGeneralTemporalMemory(TemporalMemoryMonitorMixin,
                   GeneralTemporalMemory):
  pass



realDType = GetNTAReal()


DEFAULT_TEMPORAL_MEMORY_PARAMS = {"columnDimensions": (2048,),
                                  "cellsPerColumn": 20,
                                  "activationThreshold": 20,
                                  "initialPermanence": 0.5,
                                  "connectedPermanence": 0.6,
                                  "minThreshold": 20,
                                  "maxNewSynapseCount": 40,
                                  "permanenceIncrement": 0.10,
                                  "permanenceDecrement": 0.02,
                                  "predictedSegmentDecrement": 0.08,
                                  "seed": 42,
                                  "learnOnOneCell": False}


DEFAULT_UNION_POOLER_PARAMS = {# Spatial Pooler Params
                                 # inputDimensions set to TM cell count
                                 # potentialRadius set to TM cell count
                                 "columnDimensions": [2048],
                                 "numActiveColumnsPerInhArea": 40,
                                 "stimulusThreshold": 0,
                                 "synPermInactiveDec": 0.01,
                                 "synPermActiveInc": 0.1,
                                 "synPermConnected": 0.1,
                                 "potentialPct": 0.5,
                                 "globalInhibition": True,
                                 "localAreaDensity": -1,
                                 "minPctOverlapDutyCycle": 0.001,
                                 "minPctActiveDutyCycle": 0.001,
                                 "dutyCyclePeriod": 1000,
                                 "maxBoost": 10.0,
                                 "seed": 42,
                                 "spVerbosity": 0,
                                 "wrapAround": True,

                                 # Union Temporal Pooler Params
                                 "activeOverlapWeight": 1.0,
                                 "predictedActiveOverlapWeight": 10.0,
                                 "maxUnionActivity": 0.20,
                                 "exciteFunctionType": 'Fixed',
                                 "decayFunctionType": 'NoDecay'}

trials = 30
feedback_n = 400

tmNoFeedback = None
tmFeedback = None
up = None

def setup(upSet=False):
  """
  Setup experiment, create two identical TM models, named as tmNoFeedback and tmFeedback
  Create a union temporal pooler if upSet is True
  """
  global tmNoFeedback, tmFeedback, up

  print "Initializing temporal memory..."
  params = dict(DEFAULT_TEMPORAL_MEMORY_PARAMS)
  tmNoFeedback = MonitoredGeneralTemporalMemory(mmName="TM1", **params)
  tmFeedback = MonitoredGeneralTemporalMemory(mmName="TM2", **params)
  feedback_n = 400
  trials = 30
  print "Done."

  if upSet:
    print "Initializing union pooler..."
    params = dict(DEFAULT_UNION_POOLER_PARAMS)
    params["inputDimensions"] = [tmFeedback.numberOfCells()]
    params["potentialRadius"] = tmFeedback.numberOfCells()
    up = UnionTemporalPooler(**params)
    print "Done."



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

def shiftingFeedback(starting_feedback, n, percent_shift=0.2):

  feedback_seq = []
  feedback = starting_feedback

  for _ in range(n):
    feedback = set([x for x in feedback])
    p = int(percent_shift*len(feedback))
    toRemove = set(random.sample(feedback, p))
    toAdd = set([random.randint(0, 2047) for _ in range(p)])
    feedback = (feedback - toRemove) | toAdd
    feedback_seq.append(feedback)

  return feedback_seq

def getUnionTemporalPoolerInput(tm):
  """
  Gets the Union Temporal Pooler input from the Temporal Memory
  """
  activeCells = np.zeros(tm.numberOfCells()).astype(realDType)
  activeCells[list(tm.activeCellsIndices())] = 1

  predictedActiveCells = np.zeros(tm.numberOfCells()).astype(realDType)
  predictedActiveCells[list(tm.predictedActiveCellsIndices())] = 1

  burstingColumns = np.zeros(tm.numberOfColumns()).astype(realDType)
  burstingColumns[list(tm.unpredictedActiveColumns)] = 1

  return activeCells, predictedActiveCells, burstingColumns

def trainUP(tm, sequences, up=None, trials=trials, clearhistory=True, verbose=0):

  for i in range(trials):
    for j, sensorPattern in enumerate(sequences):
      if sensorPattern is None:
        tm.reset()
        if up is not None:
          up.reset()
      else:
        if up is None:
          feedback = set()
        else:
          feedback = set(np.nonzero(up.getUnionSDR())[0])
        tm.compute(sensorPattern, activeApicalCells=feedback,
                   learn=True, sequenceLabel=None)
        if up is not None:
          activeCells, predActiveCells, burstingCols, = getUnionTemporalPoolerInput(tm)
          up.compute(activeCells, predActiveCells, learn=False)

    if clearhistory:
      if i == trials-1:
        if verbose > 0:
          print " TM metrics after training"
          print MonitorMixinBase.mmPrettyPrintMetrics(tm.mmGetDefaultMetrics())

        if verbose > 1:
          print " TM traces after training"
          print MonitorMixinBase.mmPrettyPrintTraces(tm.mmGetDefaultTraces(verbosity=True),
                                                     breakOnResets=tm.mmGetTraceResets())

      tm.mmClearHistory()

def runUP(tm, mutate_times, sequences, alphabet, up=None, mutation=0, verbose=0):

  allLabels = []

  for j, sensorPattern in enumerate(sequences):
    if sensorPattern is None:
      tm.reset()
    else:
      if j in mutate_times:
        if mutation:
          continue
        else:
          sensorPattern = set([random.randint(0, 2047) for _ in sensorPattern])

      if up is None:
        feedback = set()
      else:
        feedback = set(np.nonzero(up.getUnionSDR())[0])

      tm.compute(sensorPattern, activeApicalCells=feedback,
                 learn=True, sequenceLabel=None)

      if up is not None:
        activeCells, predActiveCells, burstingCols, = getUnionTemporalPoolerInput(tm)
        up.compute(activeCells, predActiveCells, learn=False)

      allLabels.append(labelPattern(sensorPattern, alphabet))

  ys = [len(x) for x in tm.mmGetTraceUnpredictedActiveColumns().data]

  if verbose > 0:
    print " TM metrics on test sequence"
    print MonitorMixinBase.mmPrettyPrintMetrics(tm.mmGetDefaultMetrics())

  if verbose > 1:
    print MonitorMixinBase.mmPrettyPrintTraces(tm.mmGetDefaultTraces(verbosity=True),
                                               breakOnResets=tm.mmGetTraceResets())

  return ys, allLabels

def train(tm, sequences, feedback_seq=None, trials=trials,
          feedback_buffer=10, clearhistory=True, verbose=0):

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
      if i == trials-1:
        if verbose > 0:
          print " TM metrics after training"
          print MonitorMixinBase.mmPrettyPrintMetrics(tm.mmGetDefaultMetrics())

        if verbose > 1:
          print " TM traces after training"
          print MonitorMixinBase.mmPrettyPrintTraces(tm.mmGetDefaultTraces(verbosity=True),
                                                     breakOnResets=tm.mmGetTraceResets())

      tm.mmClearHistory()


def run(tm, mutate_times, sequences, alphabet, feedback_seq=None, mutation=0, verbose=0):

  allLabels = []
  tm.reset()
  for j, sensorPattern in enumerate(sequences):
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
      else:
        feedback = set()

      tm.compute(sensorPattern, activeApicalCells=feedback,
                 learn=True, sequenceLabel=None)

      allLabels.append(labelPattern(sensorPattern, alphabet))

  ys = [len(x) for x in tm.mmGetTraceUnpredictedActiveColumns().data]

  if verbose > 0:
    print " TM metrics on test sequence"
    print MonitorMixinBase.mmPrettyPrintMetrics(tm.mmGetDefaultMetrics())

  if verbose > 1:
    print MonitorMixinBase.mmPrettyPrintTraces(tm.mmGetDefaultTraces(verbosity=True),
                                               breakOnResets=tm.mmGetTraceResets())

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

def confusionMatrixStats(tm, timestep):
  predAct = len(tm.mmGetTracePredictedActiveCells().data[timestep])
  predInact = len(tm.mmGetTracePredictedInactiveCells().data[timestep])
  unpredAct = len(tm.mmGetTraceUnpredictedActiveColumns().data[timestep])

  pred_or_act = tm.mmGetTraceUnpredictedActiveColumns().data[timestep] | (
                tm.mmGetTracePredictedActiveColumns().data[timestep] | (
                tm.mmGetTracePredictedInactiveColumns().data[timestep]))

  unpredInact = tm.numberOfColumns()-len(pred_or_act)

  return predAct, unpredAct, predInact, unpredInact

def printConfusionMatrix(mat):
  print "\t predicted neurons \t unpredicted columns"
  print "active: \t ", mat[0], '\t\t\t', mat[1]
  print "inactive: \t ", mat[2], '\t\t\t', mat[3]


def experiment1(aorb='a', upSet=False):
  """
  Test TM responses to miss or incorrect elements in a sequence.
  For example, instead of ABCDE, we get AB?DE (if aorb=='b') or ABDE (if aorb=='a')

  TM should have the following behavior
  (1) Burst at "D", while at the same time make prediction for E
  (2) Correctly predict "E"

  The behavior does not depend on feedback in this case
  """

  setup(upSet)

  sequences = generateSequences(2048, 40, 5, 1)
  alphabet = getAlphabet(sequences)

  if aorb == 'a':
    # miss an element in the sequence
    print "train TM on ABCDE"
    print "test TM on missing elements in a sequence, e.g., instead of ABCDE, the sequence is ABDE"
    mutation = 0
    mutate_times = []
    test_timestep = len(sequences)-2
  elif aorb == 'b':
    # an element is replaced with noise
    print "train TM on ABCDE"
    print "test TM on missing elements in a sequence, e.g., instead of ABCDE, the sequence is AB?DE"
    mutation = 1
    mutate_times = [len(sequences) - 3]
    test_timestep = len(sequences)-1
  else:
    raise ValueError


  fixed_feedback = set([random.randint(0, 2047) for _ in range(feedback_n)])
  feedback_seq = shiftingFeedback(fixed_feedback, len(sequences), percent_shift=0)

  # No feedback
  train(tmNoFeedback, sequences, verbose=0)
  ys1, allLabels = run(tmNoFeedback, mutate_times, sequences, alphabet,
                       mutation=mutation, verbose=0)

  # Feedback
  if up is None:
    train(tmFeedback, sequences, feedback_seq, verbose=0)
    ys2, _ = run(tmFeedback, mutate_times, sequences, alphabet, feedback_seq,
                 mutation, verbose=0)
  else:
    trainUP(tmFeedback, sequences, up)
    ys2, _ = runUP(tmFeedback, mutate_times, sequences, alphabet, up, mutation)


  print "Considering timestep "+str(test_timestep)
  print "No feedback confusion matrix: "
  printConfusionMatrix(confusionMatrixStats(tmNoFeedback, test_timestep))
  print
  print "Feedback confusion matrix: "
  printConfusionMatrix(confusionMatrixStats(tmFeedback, test_timestep))
  print


def experiment2(aorb='a', upSet=False):
  """
  Disambiguation experiment.
  Experiment setup:
    Train TM on sequences ABCDE and XBCDF with/without feedback
    Test TM on partial ambiguous sequences BCDE and BCDF
      Without feedback
      With correct feedback
      With incorrect feedback

  Desired result:
    With correct feedback, TM should make correct prediction after D
    With incorrect feedback, TM should not make correct prediction after D
    Without feedback, TM should predict both E & F
  """
  setup(upSet)

  # Create two sequences with format "ABCDE" and "XBCDF"
  sequences1 = generateSequences(2048, 40, 5, 1)

  sequences2 = [x for x in sequences1]
  sequences2[0] = set([random.randint(0, 2047) for _ in sequences1[0]])
  sequences2[-2] = set([random.randint(0, 2047) for _ in sequences1[-2]])

  fixed_feedback1 = set([random.randint(0, 2047) for _ in range(feedback_n)])
  fixed_feedback2 = set([random.randint(0, 2047) for _ in range(feedback_n)])
  feedback_seq1 = shiftingFeedback(fixed_feedback1, len(sequences1))
  feedback_seq2 = shiftingFeedback(fixed_feedback2, len(sequences2))

  sequences = sequences1 + sequences2
  feedback_seq = feedback_seq1 + feedback_seq2
  alphabet = getAlphabet(sequences)
  partial_sequences1 = sequences1[1:]
  partial_sequences2 = sequences2[1:]

  print "train TM on sequences ABCDE and XBCDF"
  test_sequence = partial_sequences1
  print "test TM on sequence BCDE, evaluate responses to E"

  if aorb == 'a':
    testFeedback = feedback_seq1
    print 'Feedback is in "ABCDE" state'
    print 'Desired outcome: '
    print '\t many extra prediction without feedback (>0 predicted inactive cell)'
    print '\t no extra prediction with feedback (~0 predicted inactive cell)'
  elif aorb == 'b':
    testFeedback = feedback_seq2
    print 'Feedback is in "XBCDF" state'
    print 'Desired outcome: '
    print '\t many extra prediction without feedback (>0 predicted inactive cell)'
    print '\t unexpected input with feedback (>0 predicted inactive cell, ~0 predicted active cell)'
  else:
    raise ValueError

  # No feedback
  train(tmNoFeedback, sequences, verbose=0)

  ys1, allLabels = run(tmNoFeedback, defaultdict(list), test_sequence,
                       alphabet, verbose=0)

  # Feedback
  if up is None:
    train(tmFeedback, sequences, feedback_seq, verbose= 0)
    ys2, _ = run(tmFeedback, defaultdict(list), test_sequence, alphabet,
                 feedback_seq=testFeedback, verbose=0)
  else:
    trainUP(tmFeedback, sequences, up)
    ys2, _ = runUP(tmFeedback, defaultdict(list), test_sequence, alphabet, up)

  timestep = len(test_sequence)-2
  print "Considering timestep "+str(timestep)
  print "No feedback confusion matrix: "
  printConfusionMatrix(confusionMatrixStats(tmNoFeedback, timestep))
  print
  print "Feedback confusion matrix: "
  printConfusionMatrix(confusionMatrixStats(tmFeedback, timestep))
  print


def experiment3(upSet=False):
  """
  Retain context with noisy sequences.
  Experiment setup:
    Train TM on sequences ABCDE and XBCDF with/without feedback
    Test TM on sequence with temporal variations A?CDE, A?CDF, ACDE, ACDF
      Without feedback
      With correct feedback
      With incorrect feedback

  """
  print " experiment: test TM with/without feedback on sequences with temporal variations"
  setup(upSet)

  # Create two sequences with format "ABCDE" and "XBCDF"
  sequences1 = generateSequences(2048, 40, 5, 1)

  sequences2 = [x for x in sequences1]
  sequences2[0] = set([random.randint(0, 2047) for _ in sequences1[0]])
  sequences2[-2] = set([random.randint(0, 2047) for _ in sequences1[-2]])

  fixed_feedback1 = set([random.randint(0, 2047) for _ in range(feedback_n)])
  fixed_feedback2 = set([random.randint(0, 2047) for _ in range(feedback_n)])
  feedback_seq1 = shiftingFeedback(fixed_feedback1, len(sequences1))
  feedback_seq2 = shiftingFeedback(fixed_feedback2, len(sequences2))

  sequences = sequences1 + sequences2
  feedback_seq = feedback_seq1 + feedback_seq2
  alphabet = getAlphabet(sequences)

  for test_case in xrange(4):
    # train TM with/without feedback
    print " training TM with ABCDE and XBCDF"
    # No feedback
    train(tmNoFeedback, sequences, verbose=0)
    # Feedback
    if up is None:
      train(tmFeedback, sequences, feedback_seq, verbose=0)
    else:
      trainUP(tmFeedback, sequences, up)

    testFeedback = feedback_seq1
    if test_case == 0:
      print "run on A?CDE, test for E element"
      print "desired result: "
      print " without feedback: "
      print "\t extra predictions (both E&F predicted, ~40 predicted active cells, ~40 predicted inactive cells)"
      print " with feedback: "
      print "\t no extra prediction (only E is predicted due to disambiguation at D, 0 predicted inactive cells)"
      test_sequence = sequences1
      test_sequence[1] = set([random.randint(0, 2047) for _ in sequences1[0]])
    elif test_case == 1:
      print "run on A?CDF, test for F element"
      print "desired result: "
      print " without feedback: "
      print "\t both E and F are predicted (~40 predicted active cells, ~40 predicted inactive cells)"
      print " with feedback: "
      print "\t F is unpredicted, E is predicted but inactive (clear context due to feedback) " \
            "(~40 predicted inactive cells, ~0 predicted active cells)"
      test_sequence = sequences1
      test_sequence[1] = set([random.randint(0, 2047) for _ in sequences1[0]])
      test_sequence[-2] = sequences2[-2]
    elif test_case == 2:
      print "run on ACDE, test for E element"
      print "desired result: "
      print " without feedback: "
      print "\t extra predictions (both E&F predicted, ~40 predicted active cells, ~40 predicted inactive cells)"
      print " with feedback: "
      print "\t no extra prediction (only E is predicted due to disambiguation at D, 0 predicted inactive cells)"
      test_sequence = [sequences1[0]] + sequences1[2:]
    elif test_case == 3:
      print "run on ACDF, test for F element"
      print "desired result: "
      print " without feedback: "
      print "\t both E and F are predicted (~40 predicted active cells, ~40 predicted inactive cells)"
      print " with feedback: "
      print "\t F is unpredicted, E is predicted but inactive (clear context due to feedback) " \
            "(~40 predicted inactive cells, ~0 predicted active cells)"
      test_sequence = [sequences1[0]] + sequences1[2:]
      test_sequence[-2] = sequences2[-2]

    ys1, allLabels = run(tmNoFeedback, defaultdict(list), test_sequence, alphabet)
    # Feedback
    if up is None:
      ys2, _ = run(tmFeedback, defaultdict(list), test_sequence, alphabet,
                   feedback_seq=testFeedback)
    else:
      ys2, _ = runUP(tmFeedback, defaultdict(list), test_sequence, alphabet, up)

    timestep = len(test_sequence)-2
    print "Considering timestep "+str(timestep)
    print "No feedback confusion matrix: "
    printConfusionMatrix(confusionMatrixStats(tmNoFeedback, timestep))
    print
    print "Feedback confusion matrix: "
    printConfusionMatrix(confusionMatrixStats(tmFeedback, timestep))
    print

def experiment4(aorb='a', upSet=False):
  setup(upSet)

  if aorb == 'a':
    sequence_len = 20
  elif aorb == 'b':
    sequence_len = 26
  elif aorb == 'c':
    return capacityExperiment(up)
  else:
    raise ValueError

  sequences1 = generateSequences(2048, 40, sequence_len, 1)

  sequences2 = [x for x in sequences1]
  sequences2[-2] = sequences1[1]

  fixed_feedback = set([random.randint(0, 2047) for _ in range(feedback_n)])
  feedback_seq = shiftingFeedback(fixed_feedback, len(sequences1))

  alphabet = getAlphabet(sequences1)

  # No feedback
  train(tmNoFeedback, sequences1)
  ys1, allLabels = run(tmNoFeedback, defaultdict(list), sequences2, alphabet)

  # Feedback
  if up is None:
    train(tmFeedback, sequences1, feedback_seq)
    ys2, _ = run(tmFeedback, defaultdict(list), sequences2, alphabet,
                 feedback_seq=feedback_seq)
  else:
    trainUP(tmFeedback, sequences1, up)
    ys2, _ = runUP(tmFeedback, defaultdict(list), sequences2, alphabet, up)

  timestep = sequence_len-1
  print "Considering timestep "+str(timestep)
  print "No feedback confusion matrix: "
  print confusionMatrixStats(tmNoFeedback, timestep)
  print
  print "Feedback confusion matrix: "
  print confusionMatrixStats(tmFeedback, timestep)
  print

# def capacityExperiment(up=None):

#   w = 40
#   trials_per_params = 5
#   seq_lengths = range(5, 41, 5)
#   ys = []

#   for seq_length in seq_lengths:
#     avg_y = 0.0
#     for _ in range(trials_per_params):
#       sequences1 = generateSequences(2048, w, seq_length, 1)

#       sequences2 = [x for x in sequences1]
#       sequences2[-2] = set([random.randint(0, 2047) for _ in range(w)])

#       fixed_feedback = set([random.randint(0, 2047) for _ in range(feedback_n)])
#       feedback_seq = shiftingFeedback(fixed_feedback, seq_length)

#       alphabet = getAlphabet(sequences1)

#       if up is None:
#         train(tmFeedback, sequences1, feedback_seq)
#         bursting, _ = run(tmFeedback, defaultdict(list), sequences2, alphabet,
#                           feedback_seq=feedback_seq)
#       else:
#         trainUP(tmFeedback, sequences1, up)
#         bursting, _ = runUP(tmFeedback, defaultdict(list), sequences2,
#                             alphabet, up)

#       y =  w-bursting[-1]
#       avg_y += y
#     avg_y /= trials_per_params
#     ys.append(avg_y)

#   print ys

#   plt.title('Sequence Length Capacity of Predictive Feedback Mechanism')
#   plt.xlabel('Sequence length')
#   plt.ylabel('Incorrectly active cells with random input')
#   plt.plot(seq_lengths, ys)
#   _, ymax = plt.ylim()
#   plt.ylim(0, ymax)
#   plt.show()

if __name__ == "__main__":

  print "Experiment 2a, disambiguation experiment, correct feedback"
  experiment2('a', upSet=False)
  print

  print "Experiment 2b, disambiguation experiment, incorrect feedback"
  experiment2('b', upSet=False)
  print

  print "Experiment 3a"
  experiment3(upSet=False)
  print
