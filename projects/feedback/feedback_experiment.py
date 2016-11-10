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
from copy import deepcopy
import numpy
import cPickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

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
                      sharedRange=None, seed=42):
  """
  Generate high order sequences using SequenceMachine
  """
  # Lots of room for noise sdrs
  patternAlphabetSize = 10*(sequenceLength * sequenceCount)
  patternMachine = PatternMachine(n, w, patternAlphabetSize, seed)
  sequenceMachine = SequenceMachine(patternMachine, seed)
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
  activityTraces = []
  for i,sequence in enumerate(sequences):
    (avgActiveCells, avgPredictedActiveCells, activityTrace) = exp.infer(
      sequence, sequenceNumber=i, enableFeedback=enableFeedback)
    error += avgActiveCells
    activityTraces.append(activityTrace)
  error /= len(sequences)
  print "Average error = ",error
  return error, activityTraces


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
  trialNum = args.get("trialNum", 42)

  # Create the sequences
  sequenceMachine, generatedSequences, numbers = generateSequences(
    sequenceLength=sequenceLen, sequenceCount=numSequences,
    sharedRange=(3,20), seed=trialNum)
  sequences = convertSequenceMachineSequence(generatedSequences)

  # Setup experiment and train the network on sequences
  exp = FeedbackExperiment(
    "feedback_experiment",
    numCorticalColumns=numColumns,
    numLearningPasses=60,
    seed=trialNum
  )
  exp.learnSequences(sequences)

  # Run various inference experiments

  # Run without any noise
  standardError, _ = runInference(exp, sequences)

  # Run with spatial noise for all patterns
  # print "\n\nAdding spatial noise, noiseLevel=", noiseLevel
  # noisySequences = convertSequenceMachineSequence(addSpatialNoise(sequenceMachine, generatedSequences, noiseLevel))
  # runInference(exp, noisySequences, enableFeedback=True)
  # runInference(exp, noisySequences, enableFeedback=False)

  # Successively delete elements from each sequence
  # print "\n\nAdding skip temporal noise..."
  # noisySequences = deepcopy(sequences)
  # skipErrors = numpy.zeros((2,15))
  # skipErrors[0,0] = standardError
  # skipErrors[1,0] = standardError
  # for t in range(14):
  #   print "\n\nAdding temporal skip noise, level=",t+1
  #   noisySequences = addTemporalNoise(sequenceMachine, noisySequences, 4+t, noiseType='skip')
  #   skipErrors[0,t+1], _ = runInference(exp, noisySequences, enableFeedback=True)
  #   skipErrors[1,t+1], _ = runInference(exp, noisySequences, enableFeedback=False)

  # Successively swap elements from each sequence
  noisySequences = deepcopy(sequences)
  swapErrors = numpy.zeros((2,11))
  swapErrors[0,0] = standardError
  swapErrors[1,0] = standardError
  for t in range(1):
    print "Adding temporal swap noise, level=",t+1
    noisySequences = addTemporalNoise(sequenceMachine, noisySequences, 4+2*t, noiseType='swap')
    swapErrors[0,t+1], _ = runInference(exp, noisySequences, enableFeedback=True)
    swapErrors[1,t+1], _ = runInference(exp, noisySequences, enableFeedback=False)

  # # Successively insert elements from each sequence
  # print "\n\nAdding insert temporal noise..."
  # noisySequences = deepcopy(sequences)
  # insertErrors = numpy.zeros((2,30))
  # swapErrors[0,0] = standardError
  # swapErrors[1,0] = standardError
  # for t in [10]:
  #   print "\n\nAdding insert noise, level=",t+1
  #   noisySequences = addTemporalNoise(sequenceMachine, noisySequences,
  #                                     t, noiseType='insert')
  #   insertErrors[0,t+1], activityFeedback = runInference(exp, noisySequences, enableFeedback=True)
  #   insertErrors[1,t+1], activityNoFeedback = runInference(exp, noisySequences, enableFeedback=False)

  # Add spatial noise to some subset of the sequences
  # print "\n\nAdding spatial pollution...."
  # noisySequences = deepcopy(sequences)
  # pollutionErrors = numpy.zeros((2,11))
  # for pos in range(5,10):
  #   noisySequences = addTemporalNoise(sequenceMachine, noisySequences, pos,
  #                                     spatialNoise=0.3,
  #                                     noiseType='pollute')
  # pollutionErrors[0,pos-4], _ = runInference(exp, noisySequences, enableFeedback=True)
  # pollutionErrors[1,pos-4], _ = runInference(exp, noisySequences, enableFeedback=False)

  # Can't pickle experiment so can't return it. However this is very useful
  # for debugging when running in a single thread.
  args.update({"experiment": exp})
  args.update({"swapErrors": swapErrors,
               # "activityFeedback": activityFeedback,
               # "activityNoFeedback": activityNoFeedback,
               })
  return args


def plotErrorsvsNoise(errors):
  """
  Plots errors vs noise

  errors[0] = error with feedback enabled
  errors[1] = error with feedback disabled

  """
  plt.figure()
  plotPath = os.path.join("error_vs_noise.pdf")

  # Plot each curve
  colorList = ['r', 'b', 'g', 'm', 'c', 'k', 'y']
  # colorList = {3: 'r', 5: 'b', 7: 'g', 11: 'k'}
  # markerList = {3: 'o', 5: 'D', 7: '*', 11: 'x'}
  noiseRange = range(0,len(errors[0]))
  for f in [0,1]:
    print errors[f]
    plt.plot(noiseRange, errors[f,noiseRange], color=colorList[f])

  # format
  plt.legend(['Feedback enabled', 'Feedback disabled'], loc="lower right")
  plt.xlabel("Noise")
  # plt.xticks(columnRange)
  plt.ylabel("Prediction error")
  plt.title("Prediction error vs noise")

    # save
  plt.savefig(plotPath)
  plt.close()


def plotActivity(activityFeedback, activityNoFeedback):
  """
  Plots activity trace
  """
  a = numpy.zeros(len(activityFeedback[0]))
  an = numpy.zeros(len(activityFeedback[0]))
  for i in range(len(activityFeedback)):
    a = a + activityFeedback[i]
    an = an + activityNoFeedback[i]
  a = (a - min(a)) / (max(a) - min(a))
  an = (an - min(an)) / (max(an) - min(an))

  plt.figure()
  plotPath = os.path.join("activityTrace.pdf")

  # Plot each curve
  colorList = ['r', 'b', 'g', 'm', 'c', 'k', 'y']
  position = range(0,len(a))
  plt.plot(position, a, color=colorList[0])
  plt.plot(position, an, color=colorList[1])

  # format
  plt.legend(['Feedback enabled', 'Feedback disabled'], loc="upper right")
  plt.xlabel("Time step")
  plt.yticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
  plt.ylabel("Prediction error")
  plt.title("Error with noise injected during a sequence")

  plt.savefig(plotPath)
  plt.close()


def computeErrorvsComplexity():
  """
  Train networks using a varying number of sequences. For each network inject
  noise into the sequence and compute inference error with and without feedback.
  """
  results = numpy.zeros((2,15))
  errors = []
  for numSequences in range(2,32,2):
    print "numSequences=",numSequences
    for trial in range(5):
      result = runExperiment(
                    {
                      "numSequences": numSequences,
                      "sequenceLen": 30,
                      "noiseLevel": 0.6,
                      "profile": False,
                      "trialNum": trial
                    }
      )
      err = result['swapErrors']
      results[0,(numSequences-2)/2] += err[0,1] - err[0,0]  # w feedback
      results[1,(numSequences-2)/2] += err[1,1] - err[1,0]  # w/o feedback
      errors.append(err)

  results = results / results.max()
  print "Results after normalization: ",results

  return results,errors


def plotErrorvsComplexity(errors):
  plt.figure()
  plotPath = os.path.join("error_vs_complexity.pdf")

  # Plot each curve
  colorList = ['r', 'b', 'g', 'm', 'c', 'k', 'y']
  numSequenceRange = range(2,32,2)
  print "range=",numSequenceRange
  for f in [0,1]:
    print errors[f]
    plt.plot(numSequenceRange, errors[f,:], color=colorList[f])

  # format
  plt.legend(['Feedback enabled', 'Feedback disabled'], loc="center right")
  plt.xlabel("Number of sequences learned by model")
  plt.xticks(numSequenceRange)
  plt.ylabel("Prediction error")
  plt.title("Error inferring noisy sequences with varying model complexity")

    # save
  plt.savefig(plotPath)
  plt.close()



if __name__ == "__main__":

  # Train a single network and test on a number of different noise situations
  if False:
    results = runExperiment(
                  {
                    "numSequences": 2,
                    "sequenceLen": 30,
                    "noiseLevel": 0.6,
                    "profile": False,
                  }
    )
    exp = results['experiment']
    for v in results['swapErrors'][0]:
      print v
    print results['swapErrors'][1]
    err = results['swapErrors']
    errScaled = (err - err[0,0]) / 457
    plotErrorsvsNoise(errScaled)
    activityFeedback = results["activityFeedback"]
    activityNoFeedback = results["activityNoFeedback"]
    plotActivity(activityFeedback, activityNoFeedback)

  # Train a sequence of models, each with a different number of sequences
  else:
    results,errors = computeErrorvsComplexity()
    with open("results.pkl","wb") as f:
      cPickle.dump(results,f)

    plotErrorvsComplexity(results)
