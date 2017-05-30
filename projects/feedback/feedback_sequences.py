
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
        if p >= pos and p < pos+number:
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





def runExp(noiseProbas, nbSequences, nbSeeds, noiseType, sequenceLen, sharedRange, noiseRange, whichPlot, plotTitle):

  allowedNoises = ("skip", "replace", "repeat", "crossover", "pollute")
  if noiseType not in allowedNoises:
    raise(RuntimeError("noiseType must be one of the following: ".join(allowedNoises)))

  meanErrsFB = []; meanErrsNoFB = []; meanErrsNoNoise = []
  stdErrsFB = []; stdErrsNoFB = []; stdErrsNoNoise = []
  meanPerfsFB = []; stdPerfsFB = []
  meanPerfsNoFB = []; stdPerfsNoFB = []
  stdsFB = []
  stdsNoFB=[]
  activitiesFB=[]; activitiesNoFB=[]

  diffsFB = []
  diffsNoFB = []
  overlapsFBL2=[]; overlapsNoFBL2=[]
  overlapsFBL2Next=[]; overlapsNoFBL2Next=[]
  overlapsFBL4=[]; overlapsNoFBL4=[]
  overlapsFBL4Next=[]; overlapsNoFBL4Next=[]
  corrsPredCorrectFBL4=[]; corrsPredCorrectNoFBL4=[]
  diffsFBL4Pred=[]; diffsNoFBL4Pred=[]
  diffsFBL4PredNext=[]; diffsNoFBL4PredNext=[]
  diffsFBL2=[]; diffsNoFBL2=[]
  diffsFBL2Next=[]; diffsNoFBL2Next=[]

  #noiseProbas = (.1, .15, .2, .25, .3, .35, .4, .45, 1.0)

  for noiseProba in noiseProbas:  # Varying noiseProba only produces a small range of difference if noise is during the whole period...
    for numSequences in nbSequences:

      errorsFB=[]; errorsNoFB=[]; errorsNoNoise=[]
      perfsFB = []; perfsNoFB = []

      #for probaZero in probaZeros:
      seed = 42
      for seedx in range(nbSeeds): #numSequences in 10, 30, 50:
          # Train a single network and show error for a single example noisy sequence
          seed = seedx + 123
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

          # Apply noise to sequences
          noisySequences = addTemporalNoise(sequenceMachine, noisySequences,
                                noiseStart=noiseRange[0], noiseEnd=noiseRange[1],
                                noiseProba=noiseProba)

          # *In addition* to this, add crossover or single-point noise
          if noiseType == "crossover":
            noisySequences = crossSequences(sequenceMachine, noisySequences,
                                            pos=sequenceLen/2)
          elif noiseType in ("repeat", "replace", "skip"):
            noisySequences = addPerturbation(sequenceMachine, noisySequences,
                                            noiseType=noiseType, pos=sequenceLen/2, number=1)

          # inferenceErrors[0, ...] - average error with feedback
          # inferenceErrors[1, ...] - average error without feedback
          # inferenceErrors[i, j]   - average error with noiseLevel = j
          inferenceErrors = numpy.zeros((2, 2))


          #Setup experiment and train the network on sequences
          print "Learning sequences..."
          exp = FeedbackExperiment(
            numLearningPasses= 2*sequenceLen,    # To handle high order sequences
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
          for numseq in range(len(responsesNoNoise)):

              diffsFB.append( [len(responsesNoNoise[numseq]['L4Responses'][x].symmetric_difference(responsesFB[numseq]['L4Responses'][x])) for x in range(seqlen)] )
              diffsNoFB.append( [len(responsesNoNoise[numseq]['L4Responses'][x].symmetric_difference(responsesNoFB[numseq]['L4Responses'][x])) for x in range(seqlen)] )
              overlapsFBL2.append( [len(responsesNoNoise[numseq]['L2Responses'][x].intersection(responsesFB[numseq]['L2Responses'][x])) for x in range(seqlen)] )
              overlapsNoFBL2.append( [len(responsesNoNoise[numseq]['L2Responses'][x].intersection(responsesNoFB[numseq]['L2Responses'][x])) for x in range(seqlen)] )
              overlapsFBL2Next.append( [len(responsesNoNoise[(numseq + 1) % numSequences]['L2Responses'][x].intersection(responsesFB[numseq]['L2Responses'][x])) for x in range(seqlen)] )
              overlapsNoFBL2Next.append( [len(responsesNoNoise[(numseq + 1) % numSequences]['L2Responses'][x].intersection(responsesNoFB[numseq]['L2Responses'][x])) for x in range(seqlen)] )
              overlapsFBL4.append( [len(responsesNoNoise[numseq]['L4Responses'][x].intersection(responsesFB[numseq]['L4Responses'][x])) for x in range(seqlen)] )
              overlapsNoFBL4.append( [len(responsesNoNoise[numseq]['L4Responses'][x].intersection(responsesNoFB[numseq]['L4Responses'][x])) for x in range(seqlen)] )
              overlapsFBL4Next.append( [len(responsesNoNoise[(numseq + 1) % numSequences]['L4Responses'][x].intersection(responsesFB[numseq]['L4Responses'][x])) for x in range(seqlen)] )
              overlapsNoFBL4Next.append( [len(responsesNoNoise[(numseq + 1) % numSequences]['L4Responses'][x].intersection(responsesNoFB[numseq]['L4Responses'][x])) for x in range(seqlen)] )
              diffsFBL4Pred.append( [len(responsesNoNoise[numseq]['L4Responses'][x].symmetric_difference(responsesFB[numseq]['L4Predictive'][x])) for x in range(seqlen)] )
              diffsNoFBL4Pred.append( [len(responsesNoNoise[numseq]['L4Responses'][x].symmetric_difference(responsesNoFB[numseq]['L4Predictive'][x])) for x in range(seqlen)] )
              cpcfb = []; cpcnofb=[]
              for x in range(seqlen):
                  z1 = numpy.zeros(sdrlen+1); z1[list(responsesNoNoise[numseq]['L4Responses'][x])] = 1; z1[-1] = 1
                  z2 = numpy.zeros(sdrlen+1); z2[list(responsesFB[numseq]['L4Predictive'][x])] = 1; z2[-1] = 1
                  cpcfb.append(numpy.corrcoef(z1, z2)[0,1])
                  z1 = numpy.zeros(sdrlen+1); z1[list(responsesNoNoise[numseq]['L4Responses'][x])] = 1; z1[-1] = 1
                  z2 = numpy.zeros(sdrlen+1); z2[list(responsesNoFB[numseq]['L4Predictive'][x])] = 1; z2[-1] = 1
                  cpcnofb.append(numpy.corrcoef(z1, z2)[0,1])


              corrsPredCorrectNoFBL4.append(cpcnofb[1:])
              corrsPredCorrectFBL4.append(cpcfb[1:])

              diffsFBL2.append( [len(responsesNoNoise[numseq]['L2Responses'][x].symmetric_difference(responsesFB[numseq]['L2Responses'][x])) for x in range(seqlen)] )
              diffsNoFBL2.append( [len(responsesNoNoise[numseq]['L2Responses'][x].symmetric_difference(responsesNoFB[numseq]['L2Responses'][x])) for x in range(seqlen)] )
              diffsFBL2Next.append( [len(responsesNoNoise[(numseq + 1) % numSequences]['L2Responses'][x].symmetric_difference(responsesFB[numseq]['L2Responses'][x])) for x in range(seqlen)] )
              diffsNoFBL2Next.append( [len(responsesNoNoise[(numseq + 1) % numSequences]['L2Responses'][x].symmetric_difference(responsesNoFB[numseq]['L2Responses'][x])) for x in range(seqlen)] )

              print "Size of L2 responses (FB):", [len(responsesFB[numseq]['L2Responses'][x]) for x in range(seqlen)]
              print "Size of L2 responses (NoNoise):", [len(responsesNoNoise[numseq]['L2Responses'][x]) for x in range(seqlen)]
              print "Size of L4 responses (FB):", [len(responsesFB[numseq]['L4Responses'][x]) for x in range(seqlen)]
              print "Size of L4 responses (NoFB):", [len(responsesNoFB[numseq]['L4Responses'][x]) for x in range(seqlen)]
              print "Size of L4 responses (NoNoise):", [len(responsesNoNoise[numseq]['L4Responses'][x]) for x in range(seqlen)]
              print "Size of L4 predictions (FB):", [len(responsesFB[numseq]['L4Predictive'][x]) for x in range(seqlen)]
              print "Size of L4 predictions (NoFB):", [len(responsesNoFB[numseq]['L4Predictive'][x]) for x in range(seqlen)]
              print "Size of L4 predictions (NoNoise):", [len(responsesNoNoise[numseq]['L4Predictive'][x]) for x in range(seqlen)]
              print "L2 overlap with current (FB): ", overlapsFBL2[-1]
              print "L4 overlap with current (FB): ", overlapsFBL4[-1]
              print "L4 overlap with current (NoFB): ", overlapsNoFBL4[-1]
              print "L4 correlation pred/correct (FB): ", corrsPredCorrectFBL4[-1]
              print "L4 correlation pred/correct (NoFB): ", corrsPredCorrectNoFBL4[-1]
              print "NoNoise sequence:", [list(x)[:2] for x in sequences[numseq]]
              print "Noise sequence:", [list(x)[:2] for x in noisySequences[numseq]]
              print "NoNoise L4 responses:", [list(x)[:2] for x in responsesNoNoise[numseq]['L4Responses']]
              print "NoFB L4 responses:", [list(x)[:2] for x in responsesNoFB[numseq]['L4Responses']]
              print ""

          # Compute mean performance / error for this seed.
          #perfsFB.append(numpy.mean(numpy.array(diffsFB)[:,6:]))
          perfsFB.append(numpy.mean(corrsPredCorrectFBL4))
          perfsNoFB.append(numpy.mean(corrsPredCorrectNoFBL4))
          errorsNoNoise.append(inferenceErrors[0,0])
          errorsFB.append(inferenceErrors[0,1])
          errorsNoFB.append(inferenceErrors[1,1])

          # Accumulating all the activity traces of all sequences (NOTE: Here '+' means list concatenation!)
          activitiesFB += activityFB
          activitiesNoFB += activityNoFB

      # Mean performance / error for this set of parameters (across all seeds and sequences for each seed)
      meanPerfsFB.append(numpy.mean(perfsFB))
      meanPerfsNoFB.append(numpy.mean(perfsNoFB))
      stdPerfsFB.append(numpy.std(perfsFB))
      stdPerfsNoFB.append(numpy.std(perfsNoFB))

      meanErrsFB.append(numpy.mean(errorsFB))
      meanErrsNoFB.append(numpy.mean(errorsNoFB))
      meanErrsNoNoise.append(numpy.mean(errorsNoNoise))
      stdErrsFB.append(numpy.std(errorsFB))
      stdErrsNoFB.append(numpy.std(errorsNoFB))
      stdErrsNoNoise.append(numpy.std(errorsNoNoise))

      # In the following, starting from column 1 to remove the initial response (which is always maximal surprise)
      # So actual length of sequences is sequenceLen - 1 (except for omission noise, sequenceLen - 2)
      aFB = numpy.array(activitiesFB)[:,1:]; aNoFB = numpy.array(activitiesNoFB)[:,1:]
      oFB = numpy.array(overlapsFBL2)[:,1:]; oNoFB = numpy.array(overlapsNoFBL2)[:,1:];
      dpredFB = numpy.array(diffsFBL4Pred)[:,1:]; dpredNoFB = numpy.array(diffsNoFBL4Pred)[:,1:];
      oFBNext = numpy.array(overlapsFBL2Next)[:,1:]; oNoFBNext = numpy.array(overlapsNoFBL2Next)[:,1:];
      xx = numpy.arange(aFB.shape[1])+1  # This +1 here
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
        plt.errorbar(xx, numpy.mean(corrsPredCorrectFBL4 , axis=0), yerr=numpy.std(corrsPredCorrectFBL4, axis=0), color='r', label='Feedback enabled');
        plt.errorbar(xx, numpy.mean(corrsPredCorrectNoFBL4 , axis=0), yerr=numpy.std(corrsPredCorrectNoFBL4, axis=0), color='b', label='Feedback disabled')
        if noiseType == 'skip':
          plt.axvspan(sharedRange[0]+.5, sharedRange[1] -2 +.5, alpha=0.25, color='pink', label="Shared Range") # -2 because omission removes one time step from the sequences

        else:
          plt.axvspan(sharedRange[0]+.5, sharedRange[1] -1 +.5, alpha=0.25, color='pink')
        plt.axvline(sequenceLen/2, 0, 1, ls='--', label='Perturbation', color='black')
        plt.ylabel("Prediction Performance"); #plt.xticks(noiseProbas)
        plt.legend(loc='best')
        plt.title(plotTitle)
        plt.savefig(plotTitle+".png")
        plt.show()

      if whichPlot == "overlaps":
        # We actually draw two plots.
        # First, we plot the average prediction error in Layer 4, computed as the total activity
        plt.figure()
        plt.errorbar(xx, numpy.mean((aFB - 40.0) / 280.0, axis=0), yerr=numpy.std((aFB - 40.0) / 280.0, axis=0), color='r', label='Feedback enabled')
        plt.errorbar(xx, numpy.mean((aNoFB - 40.0) / 280.0, axis=0), yerr=numpy.std((aNoFB - 40.0) / 280.0, axis=0), color='b', label='Feedback disabled')
        plt.axvspan(sharedRange[0]+.5, sharedRange[1] -1 + .5, alpha=0.25, color='pink')
        plt.axvspan(sharedRange[1]-1 + .5, plt.xlim()[1], alpha=0.1, color='blue')
        plt.title(plotTitle+": prediction errors")
        plt.savefig(plotTitle+": prediction errors.png")
        plt.show()

        # Then, we show the mean similarities of Layer 2 representations to original L2 representations of both source sequences used in the crossover
        plt.figure()
        plt.errorbar(xx, numpy.mean(oFBNext, axis=0), yerr=numpy.std(oFBNext, axis=0), color='m', label='Sequence 2'); plt.errorbar(xx, numpy.mean(oFB, axis=0), yerr=numpy.std(oFB, axis=0), color='c', label='Sequence 1')
        plt.axvspan(sharedRange[0] + .5, sharedRange[1] -1 + .5, alpha=0.25, color='pink')
        plt.axvspan(sharedRange[1]-1 + .5, plt.xlim()[1], alpha=0.1, color='blue')
        plt.legend(loc='best')
        plt.title(plotTitle+": top-layer representations")
        plt.savefig(plotTitle+": top-layer representations.png")

        plt.show()

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
        plt.errorbar(xx, meanPerfsFB, yerr=stdPerfsFB, color='r', label='Feedback enabled')
        plt.errorbar(xx, meanPerfsNoFB, yerr=stdPerfsNoFB, color='b', label='Feedback disabled')
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


if __name__ == "__main__":

  plt.ion()


  runExp(noiseProbas=(.1,), nbSequences=(5,), nbSeeds=5, noiseType="replace", sequenceLen=30, sharedRange=(0,0), noiseRange=(0,30), whichPlot="corrspredcorrect", plotTitle="Prediction performance with randomized stimulus (no shared range)")
  runExp(noiseProbas=(.1,), nbSequences=(5,), nbSeeds=5, noiseType="replace", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,30), whichPlot="corrspredcorrect", plotTitle="Prediction performance with randomized stimulus (shared range)")
  runExp(noiseProbas=(.1,), nbSequences=(5,), nbSeeds=5, noiseType="repeat", sequenceLen=30, sharedRange=(0,0), noiseRange=(0,30), whichPlot="corrspredcorrect", plotTitle="Prediction performance with repeated stimulus (no shared range)")
  runExp(noiseProbas=(.1,), nbSequences=(5,), nbSeeds=5, noiseType="repeat", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,30), whichPlot="corrspredcorrect", plotTitle="Prediction performance with repeated stimulus (shared range)")
  # # Don't use corrspredcorrect with "skip" noise - activities would be fine though.
  # runExp(noiseProbas=(.1,), nbSequences=(3,), nbSeeds=5, noiseType="skip", sequenceLen=30, sharedRange=(0,0), noiseRange=(0,30), whichPlot="corrspredcorrect", plotTitle="Prediction performance with omitted stimulus (no shared range)")
  # runExp(noiseProbas=(.1,), nbSequences=(5,), nbSeeds=5, noiseType="skip", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,30), whichPlot="corrspredcorrect", plotTitle="Prediction performance with omitted stimulus (shared range)")
  runExp(noiseProbas=(.1,), nbSequences=(5,), nbSeeds=5, noiseType="crossover", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,30), whichPlot="overlaps", plotTitle="End-swapped sequences (shared range)")
  runExp(noiseProbas=(.1,), nbSequences=(5,), nbSeeds=5, noiseType="crossover", sequenceLen=30, sharedRange=(0,0), noiseRange=(0,30), whichPlot="overlaps", plotTitle="End-swapped sequences (no shared range)")

  runExp(noiseProbas=( .1, .2, .3, .4, .5), nbSequences=(5,), nbSeeds=3, noiseType="pollute", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,30), whichPlot="perfs", plotTitle="Prediction performance vs noise level (shared range) (high RTB)")
  runExp(noiseProbas=( .1, .2, .3, .4, .5), nbSequences=(5,), nbSeeds=3, noiseType="pollute", sequenceLen=30, sharedRange=(0,0), noiseRange=(0,30), whichPlot="perfs", plotTitle="Prediction performance vs noise level (no shared range) (high RTB)")
