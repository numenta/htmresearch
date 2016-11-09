# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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

"""
This file runs a number of experiments testing the effectiveness of feedback
when noisy inputs.
"""

import os
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy

from nupic.data.generators.pattern_machine import PatternMachine
from nupic.data.generators.sequence_machine import SequenceMachine
from htmresearch.frameworks.layers.feedback_experiment import FeedbackExperiment


def convertSequenceMachineSequence(generatedSequences):
  """
  Convert a sequence from the SequenceMachine into a list of sequences, such
  that each sequence is a list of set of SDRs.
  """
  sequenceList = []
  currentSequence = []
  for s in generatedSequences:
    if s is None:
      sequenceList.append(currentSequence)
      currentSequence = []
    else:
      currentSequence.append(s)

  return sequenceList


def generateSequences(n=2048, w=40, sequenceLength=5, sequenceCount=2,
                      sharedRange=None):
  """
  Generate high order sequences using SequenceMachine
  """
  # Lots of room for noise sdrs
  patternAlphabetSize = 10*(sequenceLength * sequenceCount)
  patternMachine = PatternMachine(n, w, patternAlphabetSize)
  sequenceMachine = SequenceMachine(patternMachine)
  numbers = sequenceMachine.generateNumbers(sequenceCount, sequenceLength,
                                            sharedRange=sharedRange )
  generatedSequences = sequenceMachine.generateFromNumbers(numbers)

  return sequenceMachine, generatedSequences, numbers


def addSpatialNoise(sequenceMachine, generatedSequences, amount):
  """
  Add spatial noise to sequences.
  """
  noisySequences = sequenceMachine.addSpatialNoise(generatedSequences, amount)
  return noisySequences


def addTemporalNoise(sequenceMachine, sequences, pos,
                     spatialNoise = 0.5,
                     noiseType='skip'):
  """
  For each sequence, add temporal noise at position 'pos'. Possible types of
  noise:
    'skip'   : skip element pos
    'swap'   : swap sdr at position pos with sdr at position pos+1
    'insert' : insert a new random sdr at position pos
    'pollute': add a lot of noise to sdr at position pos
  """
  patternMachine = sequenceMachine.patternMachine
  newSequences = []
  for s in sequences:
    newSequence = []
    for p,sdr in enumerate(s):
      if noiseType == 'skip':
        if p == pos:
          pass
        else:
          newSequence.append(sdr)
      elif noiseType == 'pollute':
        if p == pos:
          newsdr = patternMachine.addNoise(sdr, spatialNoise)
          newSequence.append(newsdr)
        else:
          newSequence.append(sdr)
      elif noiseType == 'swap':
        if p == pos:
          newSequence.append(s[pos+1])
        if p == pos+1:
          newSequence.append(s[pos-1])
        else:
          newSequence.append(sdr)
      elif noiseType == 'insert':
        if p == pos:
          # Insert new SDR which swaps out all the bits
          newsdr = patternMachine.addNoise(sdr, 1.0)
          newSequence.append(newsdr)
        newSequence.append(sdr)
      else:
        raise Exception("Unknown noise type: "+noiseType)
    newSequences.append(newSequence)

  return newSequences


def printSequences(sequences):
  for i,s in enumerate(sequences):
    print i,":",s
    print


def runInference(exp, sequences, enableFeedback=True):
  """
  Run inference on this set of sequences and compute error
  """
  if enableFeedback:
    print "Feedback enabled: ",
  else:
    print "Feedback disabled: ",

  error = 0
  for i,sequence in enumerate(sequences):
    (totalActiveCells,totalPredictedActiveCells,
     avgActiveCells,avgPredictedActiveCells) = exp.infer(
      sequence, sequenceNumber=i, enableFeedback=enableFeedback)
    error += avgActiveCells
  error /= len(sequences)
  print "Average error = ",error
  return error


def runExperiment(args):
  """
  Run experiment.  What did you think this does?

  args is a dict representing the parameters. We do it this way to support
  multiprocessing. args contains one or more of the following keys:

  @param noiseLevel  (float) Noise level to add to the locations and features
                             during inference. Default: 0.0
  @param numSequences (int)  The number of sequences.
                             Default: 10
  @param sequenceLen  (int)  The length of each sequence
                             Default: 10
  @param numColumns  (int)   The total number of cortical columns in network.
                             Default: 1
  @param trialNum     (int)  Trial number, for reporting

  The method returns the args dict updated with two additional keys:
    convergencePoint (int)   The average number of iterations it took
                             to converge across all objects
    objects          (pairs) The list of objects we trained on
  """
  numSequences = args.get("numSequences", 10)
  sequenceLen = args.get("sequenceLen", 10)
  numColumns = args.get("numColumns", 1)
  noiseLevel = args.get("noiseLevel", 0.0)  # TODO: implement this?
  trialNum = args.get("trialNum", 42)

  # Create the objects
  sequenceMachine, generatedSequences, numbers = generateSequences(
    sequenceLength=sequenceLen, sequenceCount=numSequences,
    sharedRange=(3,26))
  sequences = convertSequenceMachineSequence(generatedSequences)

  # Setup experiment and train the network on sequences
  name = "feedback_S%03d_SL%03d_C%03d_T%03d" % (
    numSequences, sequenceLen, numColumns, trialNum
  )

  # Use previously trained network if requested
  exp = FeedbackExperiment(
    name,
    numCorticalColumns=numColumns,
    numLearningPasses=60,
    seed=trialNum
  )

  exp.learnSequences(sequences)

  # Run various inference experiments

  # Run without any noise
  standardError = runInference(exp, sequences)

  # Run without spatial noise for all patterns
  # print "\n\nAdding spatial noise, noiseLevel=", noiseLevel
  # noisySequences = convertSequenceMachineSequence(addSpatialNoise(sequenceMachine, generatedSequences, noiseLevel))
  # runInference(exp, noisySequences, enableFeedback=True)
  # runInference(exp, noisySequences, enableFeedback=False)

  # Successively delete elements from each sequence
  print "\n\nAdding skip temporal noise..."
  noisySequences = deepcopy(sequences)
  skipErrors = numpy.zeros((2,15))
  skipErrors[0,0] = standardError
  skipErrors[1,0] = standardError
  for t in range(14):
    print "\n\nAdding temporal skip noise, level=",t+1
    noisySequences = addTemporalNoise(sequenceMachine, noisySequences, 4+t, noiseType='skip')
    skipErrors[0,t+1] = runInference(exp, noisySequences, enableFeedback=True)
    skipErrors[1,t+1] = runInference(exp, noisySequences, enableFeedback=False)

  # Successively swap elements from each sequence
  print "\n\nAdding swap temporal noise..."
  noisySequences = deepcopy(sequences)
  swapErrors = numpy.zeros((2,11))
  swapErrors[0,0] = standardError
  swapErrors[1,0] = standardError
  for t in range(10):
    print "\n\nAdding temporal swap noise, level=",t+1
    noisySequences = addTemporalNoise(sequenceMachine, noisySequences, 4+2*t, noiseType='swap')
    swapErrors[0,t+1] = runInference(exp, noisySequences, enableFeedback=True)
    swapErrors[1,t+1] = runInference(exp, noisySequences, enableFeedback=False)

  # Successively insert elements from each sequence
  print "\n\nAdding insert temporal noise..."
  noisySequences = deepcopy(sequences)
  insertErrors = numpy.zeros((2,11))
  swapErrors[0,0] = standardError
  swapErrors[1,0] = standardError
  for t in range(10):
    print "\n\nAdding insert noise, level=",t+1
    noisySequences = addTemporalNoise(sequenceMachine, noisySequences,
                                      4+t*2, noiseType='insert')
    insertErrors[0,t+1] = runInference(exp, noisySequences, enableFeedback=True)
    insertErrors[1,t+1] = runInference(exp, noisySequences, enableFeedback=False)

  # Add spatial noise to some subset of the sequences
  print "\n\nAdding spatial pollution...."
  noisySequences = deepcopy(sequences)
  pollutionErrors = numpy.zeros((2,11))
  for pos in range(4,10):
    noisySequences = addTemporalNoise(sequenceMachine, sequences, pos,
                                      spatialNoise=0.2,
                                      noiseType='pollute')
  pollutionErrors[0,pos-4] = runInference(exp, noisySequences, enableFeedback=True)
  pollutionErrors[1,pos-4] = runInference(exp, noisySequences, enableFeedback=False)

  # Can't pickle experiment so can't return it. However this is very useful
  # for debugging when running in a single thread.
  args.update({"experiment": exp})
  args.update({"swapErrors": swapErrors})
  return args


def plotConvergenceStats(convergence, columnRange, featureRange):
  """
  Plots the convergence graph

  Convergence[f,c] = how long it took it to converge with f unique features
  and c columns.

  Features: the list of features we want to plot
  """
  plt.figure()
  plotPath = os.path.join("plots", "convergence_1.png")

  # Plot each curve
  colorList = {3: 'r', 5: 'b', 7: 'g', 11: 'k'}
  markerList = {3: 'o', 5: 'D', 7: '*', 11: 'x'}
  for f in featureRange:
    print columnRange
    print convergence[f-1,columnRange]
    plt.plot(columnRange, convergence[f-1,columnRange],
             color=colorList[f],
             marker=markerList[f])

  # format
  plt.legend(['Unique features=3', 'Unique features=5',
              'Unique features=7', 'Unique features=11'], loc="upper right")
  plt.xlabel("Columns")
  plt.xticks(columnRange)
  plt.ylabel("Number of sensations")
  plt.title("Convergence")

    # save
  plt.savefig(plotPath)
  plt.close()


if __name__ == "__main__":

  # This is how you run a specific experiment in single process mode. Useful
  # for debugging, profiling, etc.
  results = runExperiment(
                {
                  "numSequences": 10,
                  "sequenceLen": 30,
                  "numColumns": 1,
                  "trialNum": 0,
                  "noiseLevel": 0.6,
                  "profile": False,
                }
  )
  exp = results['experiment']
  for v in results['swapErrors'][0]:
    print v
  print results['swapErrors'][1]