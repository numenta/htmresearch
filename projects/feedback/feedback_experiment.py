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

  @param noiseLevel  (int)   Noise level to add during inference. This an
                             integer corresponding to how many times temporal
                             noise is added to the sequence.
                             Default: 1
  @param noiseStart  (int)   The position in the sequence at which noise starts
                             Default: 4
  @param numSequences (int)  The number of sequences.
                             Default: 10
  @param sequenceLen  (int)  The length of each sequence
                             Default: 30
  @param noiseType    (str)  One of the noise types for addTemporalNoise()
                             Default: 'swap'
  @param seed         (int)  Random seed for network and for sequences
                             Default: 42
  @param L4Overrides  (dict) Parameters to override default L4 settings.
                             Default: {}

  The method returns the args dict updated with additional keys:
    experiment      (object) The instance of FeedbackExperiment we used.
  """
  numSequences = args.get("numSequences", 10)
  sequenceLen = args.get("sequenceLen", 30)
  noiseLevel = args.get("noiseLevel", 1)
  noiseType = args.get("noiseType", "swap")
  noiseStart = args.get("noiseStart", 4)
  L4Overrides = args.get("L4Overrides", {})
  seed = args.get("seed", 42)

  # Create the sequences and arrays
  sequenceMachine, generatedSequences, numbers = generateSequences(
    sequenceLength=sequenceLen, sequenceCount=numSequences,
    sharedRange=(3,int(0.8*sequenceLen)),  # Need min of 3
    seed=seed)
  sequences = convertSequenceMachineSequence(generatedSequences)
  noisySequences = deepcopy(sequences)

  # inferenceErrors[0, ...] - average error with feedback
  # inferenceErrors[1, ...] - average error without feedback
  # inferenceErrors[i, j]   - average error with noiseLevel = j
  inferenceErrors = numpy.zeros((2,noiseLevel+1))

  # Setup experiment and train the network on sequences
  exp = FeedbackExperiment(
    "feedback_experiment",
    numLearningPasses=2*sequenceLen,    # To handle high order sequences
    seed=seed,
    L4Overrides=L4Overrides,
  )
  exp.learnSequences(sequences)

  # Run various inference experiments

  # Run without any noise. This becomes our baseline error
  standardError, _ = runInference(exp, sequences)
  inferenceErrors[0,0] = standardError
  inferenceErrors[1,0] = standardError

  # Successively delete elements from each sequence
  for t in range(noiseLevel):
    print "\n\nnoiseType=",noiseType, "level=",t+1
    if noiseType == 'skip':
        noisySequences = addTemporalNoise(sequenceMachine, noisySequences,
                                          noiseStart+t, noiseType=noiseType)
    elif noiseType == 'swap':
      noisySequences = addTemporalNoise(sequenceMachine, noisySequences,
                                        noiseStart+2*t, noiseType=noiseType)
    elif noiseType == 'insert':
      noisySequences = addTemporalNoise(sequenceMachine, noisySequences,
                                        noiseStart+t*2, noiseType=noiseType)
    elif noiseType == 'pollute':
      inferenceErrors = numpy.zeros((2,11))
      noisySequences = addTemporalNoise(sequenceMachine, noisySequences,
                                        noiseStart+t,
                                        spatialNoise=0.3, noiseType=noiseType)
    inferenceErrors[0,t+1], activityFeedback = runInference(exp, noisySequences, enableFeedback=True)
    inferenceErrors[1,t+1], activityNoFeedback = runInference(exp, noisySequences, enableFeedback=False)


  # Return our various structures
  args.update({"experiment": exp})
  args.update({"inferenceErrors": inferenceErrors,
               "activityFeedback": activityFeedback,
               "activityNoFeedback": activityNoFeedback,
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
  plt.plot(position[1:], a[1:], color=colorList[0])
  plt.plot(position[1:], an[1:], color=colorList[1])

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
    for seed in range(5):
      result = runExperiment(
                    {
                      "numSequences": numSequences,
                      "sequenceLen": 30,
                      "noiseLevel": 10,
                      "profile": False,
                      "seed": seed
                    }
      )
      err = result['inferenceErrors']
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

  # This script produces three charts. Set the appropriate if-statements
  # to True to generate them.

  # Train a single network and show error for a single example noisy sequence
  if True:
    results = runExperiment(
                  {
                    "numSequences": 2,
                    "sequenceLen": 30,
                    "noiseLevel": 2,
                    "noiseType": 'skip',
                    "noiseStart": 10,
                    "profile": False,
                    "L4Overrides": {"cellsPerColumn": 4}
                  }
    )
    exp = results['experiment']
    activityFeedback = results["activityFeedback"]
    activityNoFeedback = results["activityNoFeedback"]
    plotActivity(activityFeedback, activityNoFeedback)

  # Train a single network and plot error vs amount of noise
  if False:
    results = runExperiment(
                  {
                    "numSequences": 10,
                    "sequenceLen": 30,
                    "noiseLevel": 10,
                    "noiseType": 'swap',
                    "profile": False,
                  }
    )
    exp = results['experiment']
    for v in results['inferenceErrors'][0]:
      print v
    print results['inferenceErrors'][1]
    err = results['inferenceErrors']
    errScaled = (err - err[0,0]) / 457
    plotErrorsvsNoise(errScaled)
    activityFeedback = results["activityFeedback"]
    activityNoFeedback = results["activityNoFeedback"]
    plotActivity(activityFeedback, activityNoFeedback)

  # Train a bunch of models, each with a different number of sequences.
  # Plot the prediction error vs model complexity with noisy sequences.
  # Note that this plot takes about 70-80 minutes to generate.
  if False:
    results,errors = computeErrorvsComplexity()
    with open("results.pkl","wb") as f:
      cPickle.dump(results,f)

    plotErrorvsComplexity(results)
