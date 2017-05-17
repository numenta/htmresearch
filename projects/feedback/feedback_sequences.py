# This code implements putting some random noise over extended periods (determined by parameters noiseStart and noiseEnd)

# The difference brought by feedback is very large when noise is only on the start of the sequence, much smaller for noise on whole sequence !



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
#from htmresearch.frameworks.layers.feedback_experiment import FeedbackExperiment
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
        #elif p == howMany + 5:
        #    newSequence.append(sdr)
        #    newSequence.append(sdr)
        #elif p == howMany + 6:
        #    pass
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
        #if p == pos: # or p == pos+1:
          newSequence.append(sequences[(numseq +1) % len(sequences)][p])
          #newSequence.append(set())
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
    raise("noiseType must be one of the following: ".join(allowedNoises))

  meanErrsFB = []; meanErrsNoFB = []; meanErrsNoNoise = []
  stdErrsFB = []; stdErrsNoFB = []; stdErrsNoNoise = []
  perfsFB = []
  perfsNoFB=[]
  stdsFB = []
  stdsNoFB=[]
  activitiesFB=[]; activitiesNoFB=[]

  diffsFB = []
  diffsNoFB = []
  overlapsFBL2=[]; overlapsNoFBL2=[]
  overlapsFBL2Next=[]; overlapsNoFBL2Next=[]
  overlapsFBL4=[]; overlapsNoFBL4=[]
  overlapsFBL4Next=[]; overlapsNoFBL4Next=[]
  diffsFBL4Pred=[]; diffsNoFBL4Pred=[]
  diffsFBL4PredNext=[]; diffsNoFBL4PredNext=[]
  diffsFBL2=[]; diffsNoFBL2=[]
  diffsFBL2Next=[]; diffsNoFBL2Next=[]

  #noiseProbas = (.1, .15, .2, .25, .3, .35, .4, .45, 1.0)

  for noiseProba in noiseProbas:  # Varying noiseProba only produces a small range of difference if noise is during the whole period...
    for numSequences in nbSequences:

      errorsFB=[]; errorsNoFB=[]; errorsNoNoise=[]

      #for probaZero in probaZeros:
      seed = 42
      for seedx in range(nbSeeds): #numSequences in 10, 30, 50:
          # Train a single network and show error for a single example noisy sequence
          #noiseProba = .2
          #probaZero = .2
          seed = seedx + 123
          profile = False,
          L4Overrides = {"cellsPerColumn": 8}
          #L4Overrides = {"cellsPerColumn": 4}
          #seed = 42

          numpy.random.seed(seed)

          # Create the sequences and arrays
          print "Generating sequences..."
          sequenceMachine, generatedSequences, numbers = generateSequences(
            sequenceLength=sequenceLen, sequenceCount=numSequences,
            #sharedRange=(5,int(0.8*sequenceLen)),  # Need min of 3
            #sharedRange=(5,sequenceLen),
            #sharedRange=(0,0),
            sharedRange=sharedRange,
            seed=seed)

          sequences = convertSequenceMachineSequence(generatedSequences)
          noisySequences = deepcopy(sequences)

          # Apply noise to sequences
          noisySequences = addTemporalNoise(sequenceMachine, noisySequences,
                                #noiseStart=0, noiseEnd=sequenceLen,
                                noiseStart=noiseRange[0], noiseEnd=noiseRange[1],
                                noiseProba=noiseProba)

          # *In addition* to this, add crossover or single-point noise
          if noiseType == "crossover":
            noisySequences = crossSequences(sequenceMachine, noisySequences,
                                            pos=sequenceLen/2)
          elif noiseType in ("repeat", "replace", "skip"):
            #def addPerturbation(sequenceMachine, sequences, noiseType, pos, number=1):

            noisySequences = addPerturbation(sequenceMachine, noisySequences,
                                            #noiseStart=0, noiseEnd=sequenceLen,
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



          # responsesX dimensions:  NbSeqs x L2/L4 x NbTimeSteps
          # We compute the overlap of L4 responses to noisy vs. original, non-noisy sequences,  at each time step in each sequence, both for with FB and w/o FB.
          # NOTE: For this measure, noise must NOT change the length of the sequences!
          seqlen = len(noisySequences[0])
          for numseq in range(len(responsesNoNoise)):
              #diffsFB.append( [len(responsesNoNoise[numseq]['L4Responses'][x].intersection(responsesFB[numseq]['L4Responses'][x])) for x in range(seqlen)] )
              #diffsNoFB.append( [len(responsesNoNoise[numseq]['L4Responses'][x].intersection(responsesNoFB[numseq]['L4Responses'][x])) for x in range(seqlen)] )
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
              diffsFBL4PredNext.append( [len(responsesNoNoise[(numseq + 1) % numSequences]['L4Responses'][x].symmetric_difference(responsesFB[numseq]['L4Predictive'][x])) for x in range(seqlen)] )
              diffsNoFBL4PredNext.append( [len(responsesNoNoise[(numseq + 1) % numSequences]['L4Responses'][x].symmetric_difference(responsesNoFB[numseq]['L4Predictive'][x])) for x in range(seqlen)] )
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
              print "L4 overlap with current (NoFB): ", overlapsNoFBL4[-1]
              print "L4 overlap with next (NoFB): ", overlapsNoFBL4Next[-1]
              print "L4 diffs pred (FB): ", diffsFBL4Pred[-1]
              print "L4 diffs pred (NoFB): ", diffsNoFBL4Pred[-1]
              print "NoNoise sequence:", [list(x)[:2] for x in sequences[numseq]]
              print "Noise sequence:", [list(x)[:2] for x in noisySequences[numseq]]
              print "NoNoise L4 responses:", [list(x)[:2] for x in responsesNoNoise[numseq]['L4Responses']]
              print "NoFB L4 responses:", [list(x)[:2] for x in responsesNoFB[numseq]['L4Responses']]
              print ""

          #plt.plot(np.array(diffsNoFB).T, 'b')[
          perfsFB.append(numpy.sum(numpy.array(diffsFB)[:,6:]))  # Exclude the early bit of the sequence (burn-in)
          perfsNoFB.append(numpy.sum(numpy.array(diffsNoFB)[:,6:]))
          errorsNoNoise.append(inferenceErrors[0,0])
          errorsFB.append(inferenceErrors[0,1])
          errorsNoFB.append(inferenceErrors[1,1])
          # No, need to chose a dimension to compute std dev over...
          #stdsFB.append(numpy.std(numpy.array(diffsFB)[:,5:]))  # Exclude the early bit of the sequence (burn-in)
          #stdsNoFB.append(numpy.std(numpy.array(diffsNoFB)[:,5:]))

          #plt.clf(); plt.plot(numpy.array(activityFB).T, 'r'); plt.plot(numpy.array(activityNoFB).T, 'b');
          #plt.savefig("ActivityCrossoverNoSharedRange.pdf")

          # Accumulating all the activity traces of all sequences (NOTE: Here '+' means list concatenation!)
          activitiesFB += activityFB
          activitiesNoFB += activityNoFB


      # Mean error for this trial, across all sequences, both with and without FB
      meanErrsFB.append(numpy.mean(errorsFB))
      meanErrsNoFB.append(numpy.mean(errorsNoFB))
      meanErrsNoNoise.append(numpy.mean(errorsNoNoise))
      stdErrsFB.append(numpy.std(errorsFB))
      stdErrsNoFB.append(numpy.std(errorsNoFB))
      stdErrsNoNoise.append(numpy.std(errorsNoNoise))

      #meanErrsFB.append(numpy.mean(perfsFB))
      #meanErrsNoFB.append(numpy.mean(perfsNoFB))
      #stdErrsFB.append(numpy.std(perfsFB))
      #stdErrsNoFB.append(numpy.std(perfsNoFB))

      # In the following, starting from column 1 to remove the initial response (which is always maximal surprise)
      # So actual length of sequences is sequenceLen - 1 (except for omission noise, sequenceLen - 2)
      aFB = numpy.array(activitiesFB)[:,1:]; aNoFB = numpy.array(activitiesNoFB)[:,1:]
      oFB = numpy.array(overlapsFBL2)[:,1:]; oNoFB = numpy.array(overlapsNoFBL2)[:,1:];
      dpredFB = numpy.array(diffsFBL4Pred)[:,1:]; dpredNoFB = numpy.array(diffsNoFBL4Pred)[:,1:];
      oFBNext = numpy.array(overlapsFBL2Next)[:,1:]; oNoFBNext = numpy.array(overlapsNoFBL2Next)[:,1:];
      xx = numpy.arange(aFB.shape[1])+1  # This +1 here
      if whichPlot == "activities":
        plt.figure()
        plt.errorbar(xx, numpy.mean((aFB - 40.0) / 280.0, axis=0), yerr=numpy.std((aFB - 40.0) / 280.0, axis=0), color='r', label='Feedback enabled');
        plt.errorbar(xx, numpy.mean((aNoFB - 40.0) / 280.0, axis=0), yerr=numpy.std((aNoFB - 40.0) / 280.0, axis=0), color='b', label='Feedback disabled')
        if noiseType == 'skip':
          plt.axvspan(sharedRange[0]+.5, sharedRange[1] -2 +.5, alpha=0.25, color='pink', label="Shared Range") # -2 because omission removes one time step from the sequences

        else:
          plt.axvspan(sharedRange[0]+.5, sharedRange[1] -1 +.5, alpha=0.25, color='pink')
        plt.axvline(sequenceLen/2, 0, 1, ls='--', label='Perturbation', color='black')
        plt.legend(loc='best')
        plt.title(plotTitle)
        plt.savefig(plotTitle+".png")
        plt.show()

      if whichPlot == "diffpreds":
        plt.figure()
        plt.errorbar(xx, numpy.mean((dpredFB - 40.0) / 280.0, axis=0), yerr=numpy.std((dpredFB - 40.0) / 280.0, axis=0), color='r', label='Feedback enabled');
        plt.errorbar(xx, numpy.mean((dpredNoFB - 40.0) / 280.0, axis=0), yerr=numpy.std((dpredNoFB - 40.0) / 280.0, axis=0), color='b', label='Feedback disabled')
        if noiseType == 'skip':
          plt.axvspan(sharedRange[0]+.5, sharedRange[1] -2 +.5, alpha=0.25, color='pink', label="Shared Range") # -2 because omission removes one time step from the sequences

        else:
          plt.axvspan(sharedRange[0]+.5, sharedRange[1] -1 +.5, alpha=0.25, color='pink')
        plt.axvline(sequenceLen/2, 0, 1, ls='--', label='Perturbation', color='black')
        plt.legend(loc='best')
        plt.title(plotTitle)
        plt.savefig(plotTitle+".png")
        plt.show()

      if whichPlot == "overlaps":
          # First, we plot the average prediction error in Layer 4, computed as the total activity
        plt.figure()
        #plt.plot(numpy.mean(numpy.array(activitiesFB), axis=0), 'r');  plt.plot(numpy.mean(numpy.array(activitiesNoFB), axis=0), 'b')
        plt.errorbar(xx, numpy.mean((aFB - 40.0) / 280.0, axis=0), yerr=numpy.std((aFB - 40.0) / 280.0, axis=0), color='r', label='Feedback enabled')
        plt.errorbar(xx, numpy.mean((aNoFB - 40.0) / 280.0, axis=0), yerr=numpy.std((aNoFB - 40.0) / 280.0, axis=0), color='b', label='Feedback disabled')
        plt.axvspan(sharedRange[0]+.5, sharedRange[1] -1 + .5, alpha=0.25, color='pink')
        plt.axvspan(sharedRange[1]-1 + .5, plt.xlim()[1], alpha=0.1, color='blue')
        plt.title(plotTitle+": prediction errors")
        plt.savefig(plotTitle+": prediction errors.png")
        plt.show()

        # Then, we show the mean similarities of Layer 2 representations to original L2 representations of both source sequences used in the crossover
        plt.figure()
        #plt.plot(numpy.mean(numpy.array(overlapsFBL2), axis=0), 'c');  plt.plot(numpy.mean(numpy.array(overlapsFBL2Next), axis=0), 'm')
        #plt.plot(numpy.array(overlapsFBL4).transpose()[2:,:], 'c');  plt.plot(numpy.array(overlapsFBL4Next).transpose()[2:,:], 'm')
        plt.errorbar(xx, numpy.mean(oFBNext, axis=0), yerr=numpy.std(oFBNext, axis=0), color='m', label='Sequence 2'); plt.errorbar(xx, numpy.mean(oFB, axis=0), yerr=numpy.std(oFB, axis=0), color='c', label='Sequence 1')
        plt.axvspan(sharedRange[0] + .5, sharedRange[1] -1 + .5, alpha=0.25, color='pink')
        plt.axvspan(sharedRange[1]-1 + .5, plt.xlim()[1], alpha=0.1, color='blue')
        plt.legend(loc='best')
        plt.title(plotTitle+": top-layer representations")
        plt.savefig(plotTitle+": top-layer representations.png")

        plt.show()
        #plt.figure(); plt.plot(numpy.mean(numpy.array(overlapsFBL2), axis=0), 'c');  plt.plot(numpy.mean(numpy.array(overlapsFBL2Next), axis=0), 'm')
        #plt.show()
      #plt.figure(); plt.plot(numpy.mean(numpy.array(diffsFBL2), axis=0), 'c');  plt.plot(numpy.mean(numpy.array(diffsFBL2Next), axis=0), 'm')

  # Plot the total number of errors, as a function of noise probability OR number of sequences, both with and without feedback
  if whichPlot == "errors":
        plt.figure()
        xisnbseq=0
        xisnoisep=0
        if len(noiseProbas)>1:
          xisnoisep=1
          xx = noiseProbas
        if len(nbSequences)>1:
          xisnbseq=1
          xx = nbSequences
        #xx = nbSequences
        plt.errorbar(xx, meanErrsFB, yerr=stdErrsFB, color='r', label='Feedback enabled')
        plt.errorbar(xx, meanErrsNoFB, yerr=stdErrsNoFB, color='b', label='Feedback disabled')
        plt.xlim([numpy.min(xx)*.9, numpy.max(xx)*1.1])
        plt.xticks(xx)
        #plt.figure(); plt.plot(meanErrsFB, 'r'); plt.plot(meanErrsNoFB, 'b');
        if xisnoisep:
          plt.xlabel("Noise probability")
        if xisnbseq:
          plt.xlabel("Nb. of learned sequences")

        plt.ylabel("Avg. Error"); #plt.xticks(noiseProbas)
        plt.title(plotTitle)
        plt.show()
        plt.savefig(plotTitle+".png")
  #plt.savefig("ActivityNoiseInitial.pdf")
  #plt.clf(); plt.plot(errorsFB, 'r'); plt.plot(errorsNoFB, 'b'); plt.xlabel("Zeroing probability"); plt.ylabel("Avg. Error"); plt.xticks(range(len(probaZeros)), [str(x) for x in probaZeros])
  #plt.savefig("ActivityPartialWholeSeq.pdf")


if __name__ == "__main__":

  plt.ion()

  # If using whichPlot="overlaps" or whichPlot="activities", BOTH noiseProbas and nbSequences should be length-1 lists !
  # If using whichPlot="errors", EITHER noiseProbas OR nbSequences should be a multi-item list, with the other being a length-1 list.
  # The multi-element list is assumed to be the relevant indepdenent variable (i.e. what's to be plotted as x-axis)

  # # 8 cells
  # runExp(noiseProbas=(.01,), nbSequences=(10,), nbSeeds=10, noiseType="skip", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,30), whichPlot="activities", plotTitle="Prediction errors for perturbed sequences (omission)")
  # runExp(noiseProbas=(.01,), nbSequences=(10,), nbSeeds=10, noiseType="repeat", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,30), whichPlot="activities", plotTitle="Prediction errors for perturbed sequences (repetition)")
  #runExp(noiseProbas=(.01,), nbSequences=(10,), nbSeeds=10, noiseType="replace", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,30), whichPlot="activities", plotTitle="Prediction errors: randomized element")
  #runExp(noiseProbas=(.01,), nbSequences=(10,), nbSeeds=10, noiseType="replace", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,30), whichPlot="activities", plotTitle="Prediction errors: randomized element (no inertia)")
  # runExp(noiseProbas=(.01,), nbSequences=(10,), nbSeeds=10, noiseType="crossover", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,30), whichPlot="overlaps", plotTitle="End-swapped sequences (with prefix inertia and simple ff sel)")

  # Increasing noise, over whole sequence
  # runExp(noiseProbas=( .1,  .2, .3, .4), nbSequences=(30,), nbSeeds=10, noiseType="pollute", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,30), whichPlot="errors", plotTitle="Prediction errors under continuous noise")
  # Increasing noise, only during middle section
  # runExp(noiseProbas=( .1,  .2, .3, .4), nbSequences=(30,), nbSeeds=10, noiseType="pollute", sequenceLen=30, sharedRange=(5,24), noiseRange=(5,24), whichPlot="errors", plotTitle="Prediction errors under continuous noise in shared section")
  # Increasing noise, only during initial unique section
  # runExp(noiseProbas=( .1,  .2, .3, .4), nbSequences=(30,), nbSeeds=10, noiseType="pollute", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,5), whichPlot="errors", plotTitle="Prediction errors under continuous noise (initial segment)")


  # Increasing model load (number of learned sequences)
  # runExp(noiseProbas=(.25,), nbSequences=(2, 5, 10, 20, 30), nbSeeds=10, noiseType="pollute", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,30), whichPlot="errors", plotTitle="Prediction errors as a function of model load")
  # # Test # runExp(noiseProbas=(.25,), nbSequences=(2, 5, 10, 20), nbSeeds=7, noiseType="pollute", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,30), whichPlot="errors", plotTitle="Prediction errors as a function of model load")

  #runExp(noiseProbas=(.01,), nbSequences=(10,), nbSeeds=6, noiseType="replace", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,30), whichPlot="activities", plotTitle="Prediction errors for perturbed sequences (insertion)")
  #runExp(noiseProbas=(.01,), nbSequences=(5,), nbSeeds=6, noiseType="crossover", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,30), whichPlot="overlaps", plotTitle="Test: End-swapped sequences")
  runExp(noiseProbas=(.25,), nbSequences=(20,), nbSeeds=3, noiseType="pollute", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,30), whichPlot="diffpreds", plotTitle="Test: Continuous noise")
