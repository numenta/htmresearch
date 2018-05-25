# ----------------------------------------------------------------------
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
with noisy inputs.
"""

import os
from copy import deepcopy
from multiprocessing import Pool, cpu_count
import numpy
import cPickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

from nupic.data.generators.pattern_machine import PatternMachine
from nupic.data.generators.sequence_machine import SequenceMachine
import feedback_experiment
from feedback_experiment import FeedbackExperiment

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


def sparsenRange(sequenceMachine, sequences, startRange, endRange, probaZero):
  """
  """
  patternMachine = sequenceMachine.patternMachine
  newSequences = []
  for (numseq, s) in enumerate(sequences):
    newSequence = []
    for p,sdr in enumerate(s):
        if p < endRange and p >= startRange:
          newsdr = numpy.array(list(sdr))
          keep = numpy.random.rand(len(newsdr)) > probaZero
          newsdr = newsdr[keep==True]
          newSequence.append(set(newsdr))
        else:
          newSequence.append(sdr)
    newSequences.append(newSequence)

  return newSequences


def crossSequences(sequenceMachine, sequences, pos):
  """
  """
  patternMachine = sequenceMachine.patternMachine
  newSequences = []
  for (numseq, s) in enumerate(sequences):
    newSequence = []
    for p,sdr in enumerate(s):
        if p >= pos:
          newSequence.append(sequences[(numseq +1) % len(sequences)][p])
        else:
          newSequence.append(sdr)
    newSequences.append(newSequence)

  return newSequences



def addTemporalNoise(sequenceMachine, sequences, noiseStart, noiseEnd, noiseProba):
  """
  """
  patternMachine = sequenceMachine.patternMachine
  newSequences = []
  for (numseq, s) in enumerate(sequences):
    newSequence = []
    for p,sdr in enumerate(s):
        if p >= noiseStart and p < noiseEnd:
          newsdr = patternMachine.addNoise(sdr, noiseProba)
          newSequence.append(newsdr)
        else:
          newSequence.append(sdr)
    newSequences.append(newSequence)

  return newSequences


def addPerturbation(sequenceMachine, sequences, noiseType, pos, number=1):
  """
  """
  patternMachine = sequenceMachine.patternMachine
  newSequences = []
  for (numseq, s) in enumerate(sequences):
    newSequence = []
    for p,sdr in enumerate(s):
        if noiseType == "swap":
            if p==pos:
                newSequence.append(s[p+1])
            elif p==pos+1:
                newSequence.append(s[p-1])
            else:
                newSequence.append(sdr)
        elif p >= pos and p < pos+number:
          if noiseType == "skip":
            pass
          elif noiseType == "replace":
            newsdr = patternMachine.addNoise(sdr, 1.0)
            newSequence.append(newsdr)
          elif noiseType == "repeat":
            newSequence.append(s[p-1])
          else:
            raise("Unrecognized Noise Type!")
        else:
          newSequence.append(sdr)
    newSequences.append(newSequence)

  return newSequences

def runInference(exp, sequences, enableFeedback=True):
  """
  Run inference on this set of sequences and compute error
  """
  if enableFeedback:
    print "Feedback enabled: "
  else:
    print "Feedback disabled: "

  error = 0
  activityTraces = []
  responses = []
  for i,sequence in enumerate(sequences):
    (avgActiveCells, avgPredictedActiveCells, activityTrace, responsesThisSeq) = exp.infer(
      sequence, sequenceNumber=i, enableFeedback=enableFeedback)
    error += avgActiveCells
    activityTraces.append(activityTrace)
    responses.append(responsesThisSeq)
    print " "
  error /= len(sequences)
  print "Average error = ",error
  return error, activityTraces, responses

def dictMerge(acc, x):
  for key in x.keys():
    if not key in acc:
      acc[key] = []
    acc[key] += x[key]

def caller(x):
  return runExp(*x)

def experimentWrapper(pool, noiseProbas, nbSequences, nbSeeds, noiseType, sequenceLen, sharedRange, noiseRange, whichPlot, plotTitle):

  allowedNoises = ("skip", "swap", "replace", "repeat", "crossover", "pollute")
  if noiseType not in allowedNoises:
    mystr = "noiseType must be one of the following: ".join(allowedNoises)
    raise(RuntimeError(mystr))

  metrics = {
    "meanErrsFB" : [], "meanErrsNoFB" : [], "meanErrsNoNoise" : [],
    "stdErrsFB" : [], "stdErrsNoFB" : [], "stdErrsNoNoise" : [],
    "meanPerfsFB" : [], "stdPerfsFB" : [],
    "meanPerfsNoFB" : [], "stdPerfsNoFB" : [],
    "stdsFB" : [],
    "stdsNoFB" : [],
    "activitiesFB" : [], "activitiesNoFB" : [],

    "diffsFB" : [],
    "diffsNoFB" : [],
    "overlapsFBL2" : [], "overlapsNoFBL2" : [],
    "overlapsFBL2Next" : [], "overlapsNoFBL2Next" : [],
    "overlapsFBL4" : [], "overlapsNoFBL4" : [],
    "overlapsFBL4Next" : [], "overlapsNoFBL4Next" : [],
    "corrsPredCorrectFBL4" : [], "corrsPredCorrectNoFBL4" : [],
    "corrsPredCorrectFBL4Next" : [], "corrsPredCorrectNoFBL4Next" : [],
    "diffsFBL4Pred" : [], "diffsNoFBL4Pred" : [],
    "diffsFBL4PredNext" : [], "diffsNoFBL4PredNext" : [],
    "diffsFBL2" : [], "diffsNoFBL2" : [],
    "diffsFBL2Next" : [], "diffsNoFBL2Next" : [],
    "errorsFB" : [], "errorsNoFB" : [], "errorsNoNoise" : [],
    "perfsFB" : [], "perfsNoFB" : []
  }


  # Random seed has an influence only on the 'randomized stimulus insertion'
  # experiment (can make non-zero error bars on the red line)
  seeds = [seedx + 12345 for seedx in range(nbSeeds)]
  print "Using seeds:", seeds
  for noiseProba in noiseProbas:
    for numSequences in nbSequences:

      # These should be initialized for each new noiseProba (but not for each seed)
      metrics["corrsPredCorrectFBL4"] = []
      metrics["corrsPredCorrectNoFBL4"]= []
      metrics["corrsPredCorrectFBL4Next"]= []
      metrics["corrsPredCorrectNoFBL4Next"] = []

      errorsFB=[]; errorsNoFB=[]; errorsNoNoise=[]
      perfsFB = []; perfsNoFB = []

      expParams = [(noiseProba, numSequences, seed,
                    noiseType, sequenceLen, sharedRange,
                    noiseRange) for seed in seeds]
      #metricSets = map(caller, expParams)
      metricSets = pool.map(caller, expParams)
      for metricSet in metricSets:
        dictMerge(metrics, metricSet)

      # Mean performance / error for this set of parameters (across all seeds and sequences for each seed)
      #metrics["meanPerfsFB"].append(numpy.mean(metrics["perfsFB"]))
      #metrics["meanPerfsNoFB"].append(numpy.mean(metrics["perfsNoFB"]))
      #metrics["stdPerfsFB"].append(numpy.std(metrics["perfsFB"]))
      #metrics["stdPerfsNoFB"].append(numpy.std(metrics["perfsNoFB"]))
      metrics["meanPerfsFB"].append(numpy.mean(metrics["corrsPredCorrectFBL4"]))
      metrics["meanPerfsNoFB"].append(numpy.mean(metrics["corrsPredCorrectNoFBL4"]))
      metrics["stdPerfsFB"].append(numpy.std(metrics["corrsPredCorrectFBL4"]))
      metrics["stdPerfsNoFB"].append(numpy.std(metrics["corrsPredCorrectNoFBL4"]))

      aFB = numpy.array(metrics["activitiesFB"])[:,:]; aNoFB = numpy.array(metrics["activitiesNoFB"])[:,:]
      oFB = numpy.array(metrics["overlapsFBL2"])[:,:]; oNoFB = numpy.array(metrics["overlapsNoFBL2"])[:,:];
      dpredFB = numpy.array(metrics["diffsFBL4Pred"])[:,:]; dpredNoFB = numpy.array(metrics["diffsNoFBL4Pred"])[:,:];
      oFBNext = numpy.array(metrics["overlapsFBL2Next"])[:,:]; oNoFBNext = numpy.array(metrics["overlapsNoFBL2Next"])[:,:];
      xx = numpy.arange(aFB.shape[1])

      if whichPlot == "activities":
        plt.figure()
        plt.errorbar(xx, numpy.mean(aFB, axis=0), yerr=numpy.std(aFB, axis=0), color='r', label='Feedback enabled');
        plt.errorbar(xx, numpy.mean(aNoFB , axis=0), yerr=numpy.std(aNoFB, axis=0), color='b', label='Feedback disabled')
        # plt.errorbar(xx, numpy.mean((aFB - 40.0) / 280.0, axis=0), yerr=numpy.std((aFB - 40.0) / 280.0, axis=0), color='r', label='Feedback enabled');
        # plt.errorbar(xx, numpy.mean((aNoFB - 40.0) / 280.0, axis=0), yerr=numpy.std((aNoFB - 40.0) / 280.0, axis=0), color='b', label='Feedback disabled')
        if noiseType == 'skip':
          plt.axvspan(sharedRange[0]+.5, sharedRange[1] -2 +.5, alpha=0.25, color='pink', label="Shared Range") # -2 because omission removes one time step from the sequences
        else:
          plt.axvspan(sharedRange[0]+.5, sharedRange[1] -1 +.5, alpha=0.25, color='pink')
        plt.axvline(sequenceLen/2, 0, 1, ls='--', label='Perturbation', color='black')
        plt.legend(loc='best')
        plt.title(plotTitle)
        plt.savefig(plotTitle+".png")
        plt.show()

      # Do NOT use this with "skip" noise (omitted item). It can't work, since the new sequence is permanently out of sync with the old one after the omission.
      if whichPlot == "corrspredcorrect":
        plt.figure()
        z1 = numpy.array(metrics["corrsPredCorrectFBL4"])
        if noiseType == 'crossover':
          z2 = numpy.array(metrics["corrsPredCorrectFBL4Next"])
          z1[:,24:] = z2[:,24:]
          # If needed, one can simply eliminate the first sequence in each batch, as follows:
          # z1 = numpy.delete(z1, range(0, z1.shape[0], numSequences), axis=0)  # Eliminate the first sequence in each batch
        plt.errorbar(xx[1:], numpy.mean(z1 , axis=0)[1:], yerr=numpy.std(z1[1:], axis=0)[1:], color='r', label='Feedback enabled');
        #for i, x in enumerate(xx[1:]):
        #  plt.boxplot(x, color='r');
        z1 = numpy.array(metrics["corrsPredCorrectNoFBL4"])
        if noiseType == 'crossover':
          z2 = numpy.array(metrics["corrsPredCorrectNoFBL4Next"])
          z1[:, 24:] = z2[:, 24:]
          # z1 = numpy.delete(z1, range(0, z1.shape[0], numSequences), axis=0)  # Eliminate the first sequence in each batch
        plt.errorbar(xx[1:], numpy.mean(z1 , axis=0)[1:], yerr=numpy.std(z1[1:], axis=0)[1:], color='b', label='Feedback disabled')
        if sharedRange[0] != sharedRange[1]:
          plt.axvspan(sharedRange[0]+.5, sharedRange[1] -1 +.5, alpha=0.25, color='pink')
        plt.axvline(sequenceLen/2+1, 0, 1, ls='--', label='Perturbation', color='black')
        plt.ylabel("Prediction Performance"); #plt.xticks(noiseProbas)
        plt.legend(loc='best')
        plt.title(plotTitle)
        plt.savefig(plotTitle+".png")
        plt.show()
        if noiseType == "crossover":
          # Then, we show the mean similarities of Layer 2 representations to original L2 representations of both source sequences used in the crossover
          plt.figure()
          plt.errorbar(xx[2:], numpy.mean(oFBNext, axis=0)[2:], yerr=numpy.std(oFBNext, axis=0)[2:], color='m', label='Sequence 2');
          plt.errorbar(xx[2:], numpy.mean(oFB, axis=0)[2:], yerr=numpy.std(oFB, axis=0)[2:], color='c', label='Sequence 1')
          plt.axvspan(sharedRange[0] + .5, sharedRange[1] -1 + .5, alpha=0.25, color='pink')
          plt.axvspan(sharedRange[1]-1 + .5, plt.xlim()[1], alpha=0.1, color='blue')
          plt.legend(loc='best')
          plt.title(plotTitle+": top-layer representations")
          plt.savefig(plotTitle+": top-layer representations.png")


  # Plot the average prediction performance, as a function of noise probability OR number of sequences, both with and without feedback
  if whichPlot == "perfs":
    plt.figure()
    xisnbseq=0
    xisnoisep=0
    if len(noiseProbas)>1:
      xisnoisep=1
      xx = noiseProbas
    if len(nbSequences)>1:
      xisnbseq=1
      xx = nbSequences
    plt.errorbar(xx, metrics["meanPerfsFB"], yerr=metrics["stdPerfsFB"], color='r', label='Feedback enabled')
    plt.errorbar(xx, metrics["meanPerfsNoFB"], yerr=metrics["stdPerfsNoFB"], color='b', label='Feedback disabled')
    plt.xlim([numpy.min(xx)*.9, numpy.max(xx)*1.1])
    plt.xticks(xx)
    if xisnoisep:
      plt.xlabel("Noise probability")
    if xisnbseq:
      plt.xlabel("Nb. of learned sequences")

    plt.ylabel("Avg. Prediction Performance");
    plt.title(plotTitle)
    plt.show()
    plt.savefig(plotTitle+".png")


def runExp(noiseProba, numSequences, seed, noiseType, sequenceLen, sharedRange, noiseRange):

  metrics = {
    "meanErrsFB" : [], "meanErrsNoFB" : [], "meanErrsNoNoise" : [],
    "stdErrsFB" : [], "stdErrsNoFB" : [], "stdErrsNoNoise" : [],
    "meanPerfsFB" : [], "stdPerfsFB" : [],
    "meanPerfsNoFB" : [], "stdPerfsNoFB" : [],
    "stdsFB" : [],
    "stdsNoFB" : [],
    "activitiesFB" : [], "activitiesNoFB" : [],

    "diffsFB" : [],
    "diffsNoFB" : [],
    "overlapsFBL2" : [], "overlapsNoFBL2" : [],
    "overlapsFBL2Next" : [], "overlapsNoFBL2Next" : [],
    "overlapsFBL4" : [], "overlapsNoFBL4" : [],
    "overlapsFBL4Next" : [], "overlapsNoFBL4Next" : [],
    "corrsPredCorrectFBL4" : [], "corrsPredCorrectNoFBL4" : [],
    "corrsPredCorrectFBL4Next" : [], "corrsPredCorrectNoFBL4Next" : [],
    "diffsFBL4Pred" : [], "diffsNoFBL4Pred" : [],
    "diffsFBL4PredNext" : [], "diffsNoFBL4PredNext" : [],
    "diffsFBL2" : [], "diffsNoFBL2" : [],
    "diffsFBL2Next" : [], "diffsNoFBL2Next" : [],
    "errorsFB" : [], "errorsNoFB" : [], "errorsNoNoise" : [],
    "perfsFB" : [], "perfsNoFB" : []
  }

  profile = False,
  L4Overrides = {"cellsPerColumn": 8}

  numpy.random.seed(seed)

  # Create the sequences and arrays
  print "Generating sequences..."
  sequenceMachine, generatedSequences, numbers = generateSequences(
    sequenceLength=sequenceLen, sequenceCount=numSequences,
    sharedRange=sharedRange,
    seed=seed)

  sequences = convertSequenceMachineSequence(generatedSequences)
  noisySequences = deepcopy(sequences)

  for x in range(sequenceLen):
    for y in range(x):
      if len(sequences[0][x] | sequences[0][y]) < 45:
        print x, y, len(sequences[0][x] | sequences[0][y])

  # Apply noise to sequences
  noisySequences = addTemporalNoise(sequenceMachine, noisySequences,
                        noiseStart=noiseRange[0], noiseEnd=noiseRange[1],
                        noiseProba=noiseProba)

  # *In addition* to this, add crossover or single-point noise
  if noiseType == "crossover":
    noisySequences = crossSequences(sequenceMachine, noisySequences,
                                    pos=sequenceLen/2)
  elif noiseType in ("repeat", "replace", "skip", "swap"):
    noisySequences = addPerturbation(sequenceMachine, noisySequences,
                                    noiseType=noiseType, pos=sequenceLen/2, number=1)

  # inferenceErrors[0, ...] - average error with feedback
  # inferenceErrors[1, ...] - average error without feedback
  # inferenceErrors[i, j]   - average error with noiseLevel = j
  inferenceErrors = numpy.zeros((2, 2))


  #Setup experiment and train the network on sequences
  print "Learning sequences..."
  exp = FeedbackExperiment(
    numLearningPasses= 2*sequenceLen, # To handle high order sequences
    seed=seed,
    L4Overrides=L4Overrides,
  )
  exp.learnSequences(sequences)

  print "Number of columns in exp: ", exp.numColumns
  print "Sequences learned!"

  # Run inference without any noise. This becomes our baseline error
  standardError, activityNoNoise, responsesNoNoise = runInference(exp, sequences)
  inferenceErrors[0,0] = standardError
  inferenceErrors[1,0] = standardError

  inferenceErrors[0,1], activityFB, responsesFB = runInference(
      exp, noisySequences, enableFeedback=True)
  inferenceErrors[1,1], activityNoFB, responsesNoFB = runInference(
      exp, noisySequences, enableFeedback=False)

  # Now that actual processing is done, we compute various statistics and plot graphs.

  # We compute the overlap of L4 responses to noisy vs. original, non-noisy sequences,  at each time step in each sequence, both for with FB and w/o FB.
  seqlen = len(noisySequences[0])
  sdrlen = 2048 * 8  # Should be the total number of cells in L4. Need to make this more parametrized!


    # When using end-swapped sequences, with a shared range, there are weird bugs related to the first sequence in each batch.
    # 1- The 1st sequence never recovers correct predictions after the shared section ends. All others do.
    # 2- All other sequences make 2 predictions in each item of the end of the sequence (the correct one, and another in the same
    # minicolumn), instead of 1 - except the 1st sequence (which makes a single incorrect prediction, as stated above), and the last
    # (which has its end swapped with the first).
    # I "solve" the problem by throwing away the first sequence at test time (note the '1'):
  for numseq in range(1, len(responsesNoNoise)):

    metrics["diffsFB"].append( [len(responsesNoNoise[numseq]['L4Responses'][x].symmetric_difference(responsesFB[numseq]['L4Responses'][x])) for x in range(seqlen)] )
    metrics["diffsNoFB"].append( [len(responsesNoNoise[numseq]['L4Responses'][x].symmetric_difference(responsesNoFB[numseq]['L4Responses'][x])) for x in range(seqlen)] )
    metrics["overlapsFBL2"].append( [len(responsesNoNoise[numseq]['L2Responses'][x].intersection(responsesFB[numseq]['L2Responses'][x])) for x in range(seqlen)] )
    metrics["overlapsNoFBL2"].append( [len(responsesNoNoise[numseq]['L2Responses'][x].intersection(responsesNoFB[numseq]['L2Responses'][x])) for x in range(seqlen)] )
    metrics["overlapsFBL2Next"].append( [len(responsesNoNoise[(numseq + 1) % numSequences]['L2Responses'][x].intersection(responsesFB[numseq]['L2Responses'][x])) for x in range(seqlen)] )
    metrics["overlapsNoFBL2Next"].append( [len(responsesNoNoise[(numseq + 1) % numSequences]['L2Responses'][x].intersection(responsesNoFB[numseq]['L2Responses'][x])) for x in range(seqlen)] )
    metrics["overlapsFBL4"].append( [len(responsesNoNoise[numseq]['L4Responses'][x].intersection(responsesFB[numseq]['L4Responses'][x])) for x in range(seqlen)] )
    metrics["overlapsNoFBL4"].append( [len(responsesNoNoise[numseq]['L4Responses'][x].intersection(responsesNoFB[numseq]['L4Responses'][x])) for x in range(seqlen)] )
    metrics["overlapsFBL4Next"].append( [len(responsesNoNoise[(numseq + 1) % numSequences]['L4Responses'][x].intersection(responsesFB[numseq]['L4Responses'][x])) for x in range(seqlen)] )
    metrics["overlapsNoFBL4Next"].append( [len(responsesNoNoise[(numseq + 1) % numSequences]['L4Responses'][x].intersection(responsesNoFB[numseq]['L4Responses'][x])) for x in range(seqlen)] )
    metrics["diffsFBL4Pred"].append( [len(responsesNoNoise[numseq]['L4Responses'][x].symmetric_difference(responsesFB[numseq]['L4Predicted'][x])) for x in range(seqlen)] )
    metrics["diffsNoFBL4Pred"].append( [len(responsesNoNoise[numseq]['L4Responses'][x].symmetric_difference(responsesNoFB[numseq]['L4Predicted'][x])) for x in range(seqlen)] )
    cpcfb = []; cpcnofb=[]; cpcfbnext = []; cpcnofbnext=[];
    for x in range(seqlen):
        z1 = numpy.zeros(sdrlen+1); z1[list(responsesNoNoise[numseq]['L4Responses'][x])] = 1; z1[-1] = 1
        z2 = numpy.zeros(sdrlen+1); z2[list(responsesFB[numseq]['L4Predicted'][x])] = 1; z2[-1] = 1
        cpcfb.append(numpy.corrcoef(z1, z2)[0,1])
        z1 = numpy.zeros(sdrlen+1); z1[list(responsesNoNoise[numseq]['L4Responses'][x])] = 1; z1[-1] = 1
        z2 = numpy.zeros(sdrlen+1); z2[list(responsesNoFB[numseq]['L4Predicted'][x])] = 1; z2[-1] = 1
        cpcnofb.append(numpy.corrcoef(z1, z2)[0,1])

        z1 = numpy.zeros(sdrlen+1); z1[list(responsesNoNoise[(numseq+1) % numSequences]['L4Responses'][x])] = 1; z1[-1] = 1
        z2 = numpy.zeros(sdrlen+1); z2[list(responsesFB[numseq]['L4Predicted'][x])] = 1; z2[-1] = 1
        cpcfbnext.append(numpy.corrcoef(z1, z2)[0,1])
        z1 = numpy.zeros(sdrlen+1); z1[list(responsesNoNoise[(numseq+1) % numSequences]['L4Responses'][x])] = 1; z1[-1] = 1
        z2 = numpy.zeros(sdrlen+1); z2[list(responsesNoFB[numseq]['L4Predicted'][x])] = 1; z2[-1] = 1
        cpcnofbnext.append(numpy.corrcoef(z1, z2)[0,1])


    metrics["corrsPredCorrectNoFBL4"].append(cpcnofb)
    metrics["corrsPredCorrectFBL4"].append(cpcfb)
    metrics["corrsPredCorrectNoFBL4Next"].append(cpcnofbnext)
    metrics["corrsPredCorrectFBL4Next"].append(cpcfbnext)

    metrics["diffsFBL2"].append( [len(responsesNoNoise[numseq]['L2Responses'][x].symmetric_difference(responsesFB[numseq]['L2Responses'][x])) for x in range(seqlen)] )
    metrics["diffsNoFBL2"].append( [len(responsesNoNoise[numseq]['L2Responses'][x].symmetric_difference(responsesNoFB[numseq]['L2Responses'][x])) for x in range(seqlen)] )
    metrics["diffsFBL2Next"].append( [len(responsesNoNoise[(numseq + 1) % numSequences]['L2Responses'][x].symmetric_difference(responsesFB[numseq]['L2Responses'][x])) for x in range(seqlen)] )
    metrics["diffsNoFBL2Next"].append( [len(responsesNoNoise[(numseq + 1) % numSequences]['L2Responses'][x].symmetric_difference(responsesNoFB[numseq]['L2Responses'][x])) for x in range(seqlen)] )

    print "Size of L2 responses (FB):", [len(responsesFB[numseq]['L2Responses'][x]) for x in range(seqlen)]
    print "Size of L2 responses (NoNoise):", [len(responsesNoNoise[numseq]['L2Responses'][x]) for x in range(seqlen)]
    print "Size of L4 responses (FB):", [len(responsesFB[numseq]['L4Responses'][x]) for x in range(seqlen)]
    print "Size of L4 responses (NoFB):", [len(responsesNoFB[numseq]['L4Responses'][x]) for x in range(seqlen)]
    print "Size of L4 responses (NoNoise):", [len(responsesNoNoise[numseq]['L4Responses'][x]) for x in range(seqlen)]
    print "Size of L4 predictions (FB):", [len(responsesFB[numseq]['L4Predicted'][x]) for x in range(seqlen)]
    print "Size of L4 predictions (NoFB):", [len(responsesNoFB[numseq]['L4Predicted'][x]) for x in range(seqlen)]
    print "Size of L4 predictions (NoNoise):", [len(responsesNoNoise[numseq]['L4Predicted'][x]) for x in range(seqlen)]
    print "L2 overlap with current (FB): ", metrics["overlapsFBL2"][-1]
    print "L4 overlap with current (FB): ", metrics["overlapsFBL4"][-1]
    print "L4 overlap with current (NoFB): ", metrics["overlapsNoFBL4"][-1]
    print "L4 correlation pred/correct (FB): ", metrics["corrsPredCorrectFBL4"][-1]
    print "L4 correlation pred/correct (FBNext): ", metrics["corrsPredCorrectFBL4Next"][-1]
    print "L4 correlation pred/correct (NoFB): ", metrics["corrsPredCorrectNoFBL4"][-1]
  #   print "NoNoise sequence:", [list(x)[:2] for x in sequences[numseq]]
  #   print "Noise sequence:", [list(x)[:2] for x in noisySequences[numseq]]
    print "NoNoise L4 responses:", [sorted(list(x))[:2] for x in responsesNoNoise[numseq]['L4Responses']]
    print "NoNoise L4 responses (next):", [sorted(list(x))[:2] for x in responsesNoNoise[(numseq + 1) % numSequences]['L4Responses']]
    print "NoFB L4 responses:", [sorted(list(x))[:2] for x in responsesNoFB[numseq]['L4Responses']]
    print "NoNoise L4 predictions:", [sorted(list(x))[:2] for x in responsesNoNoise[numseq]['L4Predicted']]
    print "NoFB L4 predictions:", [sorted(list(x))[:2] for x in responsesNoFB[numseq]['L4Predicted']]
    print ""

  # Compute mean performance / error for this seed.
  #metrics["perfsFB"].append(numpy.mean(numpy.array(diffsFB)[:,6:]))
  metrics["perfsFB"].append(numpy.mean(metrics["corrsPredCorrectFBL4"]))
  metrics["perfsNoFB"].append(numpy.mean(metrics["corrsPredCorrectNoFBL4"]))
  metrics["errorsNoNoise"].append(inferenceErrors[0,0])
  metrics["errorsFB"].append(inferenceErrors[0,1])
  metrics["errorsNoFB"].append(inferenceErrors[1,1])

  # Accumulating all the activity traces of all sequences (NOTE: Here '+' means list concatenation!)
  metrics["activitiesFB"] += activityFB
  metrics["activitiesNoFB"] += activityNoFB
  return metrics



if __name__ == "__main__":

  pool = Pool(8)

  plt.ion()


  experimentWrapper(pool, noiseProbas=(.1,), nbSequences=(5,), nbSeeds=8, noiseType="swap", sequenceLen=30, sharedRange=(0,0), noiseRange=(0,30), whichPlot="corrspredcorrect", plotTitle="Prediction performance with swapped stimuli (no shared range)")
  experimentWrapper(pool, noiseProbas=(.1,), nbSequences=(5,), nbSeeds=8, noiseType="swap", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,30), whichPlot="corrspredcorrect", plotTitle="Prediction performance with swapped stimuli (shared range)")
  experimentWrapper(pool, noiseProbas=(.1,), nbSequences=(5,), nbSeeds=8, noiseType="replace", sequenceLen=30, sharedRange=(0,0), noiseRange=(0,30), whichPlot="corrspredcorrect", plotTitle="Prediction performance with randomized stimulus (no shared range)")
  experimentWrapper(pool, noiseProbas=(.1,), nbSequences=(5,), nbSeeds=8, noiseType="replace", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,30), whichPlot="corrspredcorrect", plotTitle="Prediction performance with randomized stimulus (shared range)")



  experimentWrapper(pool, noiseProbas=(.1,), nbSequences=(5,), nbSeeds=8, noiseType="crossover", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,30), whichPlot="corrspredcorrect", plotTitle="End-swapped sequences (shared range)")
  experimentWrapper(pool, noiseProbas=(.1,), nbSequences=(5,), nbSeeds=8, noiseType="crossover", sequenceLen=30, sharedRange=(0,0), noiseRange=(0,30), whichPlot="corrspredcorrect", plotTitle="End-swapped sequences (no shared range)")



  experimentWrapper(pool, noiseProbas=(.1,), nbSequences=(5,), nbSeeds=8, noiseType="repeat", sequenceLen=30, sharedRange=(0,0), noiseRange=(0,30), whichPlot="corrspredcorrect", plotTitle="Prediction performance with repeated stimulus (no shared range)")
  experimentWrapper(pool, noiseProbas=(.1,), nbSequences=(5,), nbSeeds=8, noiseType="repeat", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,30), whichPlot="corrspredcorrect", plotTitle="Prediction performance with repeated stimulus (shared range)")
  # # # Don't use corrspredcorrect with "skip" noise - activities would be fine though.
  # experimentWrapper(pool, noiseProbas=(.1,), nbSequences=(3,), nbSeeds=8, noiseType="skip", sequenceLen=30, sharedRange=(0,0), noiseRange=(0,30), whichPlot="corrspredcorrect", plotTitle="Prediction performance with omitted stimulus (no shared range)")
  # experimentWrapper(pool, noiseProbas=(.1,), nbSequences=(5,), nbSeeds=8, noiseType="skip", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,30), whichPlot="corrspredcorrect", plotTitle="Prediction performance with omitted stimulus (shared range)")

  experimentWrapper(pool, noiseProbas=( .1, .2, .3, .4, .5), nbSequences=(5,), nbSeeds=8, noiseType="pollute", sequenceLen=30, sharedRange=(0,0), noiseRange=(0,30), whichPlot="perfs", plotTitle="Prediction performance vs noise level (no shared range)")
  experimentWrapper(pool, noiseProbas=( .1, .2, .3, .4, .5), nbSequences=(5,), nbSeeds=8, noiseType="pollute", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,30), whichPlot="perfs", plotTitle="Prediction performance vs noise level (shared range)")


  # When using the correlation b/w predicted and correct as a measure, increasing model load has little effect, with or without feedback.
  # experimentWrapper(pool, noiseProbas=( .25,), nbSequences=(3,30), nbSeeds=3, noiseType="pollute", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,30), whichPlot="perfs", plotTitle="Test")
  # experimentWrapper(pool, noiseProbas=( .1, .25), nbSequences=(5,), nbSeeds=3, noiseType="pollute", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,30), whichPlot="perfs", plotTitle="Test")
