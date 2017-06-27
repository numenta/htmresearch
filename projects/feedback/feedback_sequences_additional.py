
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
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
matplotlib.rcParams['pdf.fonttype'] = 42
plt.ion()

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

def runInference(exp, sequences, enableFeedback=True, apicalTiebreak=True,
                    apicalModulationBasalThreshold=True, inertia=True):
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
      sequence, sequenceNumber=i, enableFeedback=enableFeedback, apicalTiebreak=apicalTiebreak,
      apicalModulationBasalThreshold=apicalModulationBasalThreshold, inertia=inertia)
    error += avgActiveCells
    activityTraces.append(activityTrace)
    responses.append(responsesThisSeq)
    print " "
  error /= len(sequences)
  print "Average error = ",error
  return error, activityTraces, responses





def runExp(noiseProba, numSequences, nbSeeds, noiseType, sequenceLen, sharedRange, noiseRange, whichPlot, plotTitle):

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

  diffsNoAT = []; overlapsNoATL2=[]; overlapsNoATL2Next=[]; overlapsNoATL4=[]
  overlapsNoATL4Next=[]
  corrsPredCorrectNoATL4=[]; diffsNoATL4Pred=[]; diffsNoATL4PredNext=[]
  diffsNoATL2=[]; diffsNoATL2Next=[]
  diffsNoAM = []; overlapsNoAML2=[]; overlapsNoAML2Next=[]; overlapsNoAML4=[]
  overlapsNoAML4Next=[]
  corrsPredCorrectNoAML4=[]; diffsNoAML4Pred=[]; diffsNoAML4PredNext=[]
  diffsNoAML2=[]; diffsNoAML2Next=[]
  diffsNoIN = []; overlapsNoINL2=[]; overlapsNoINL2Next=[]; overlapsNoINL4=[]
  overlapsNoINL4Next=[]
  corrsPredCorrectNoINL4=[]; diffsNoINL4Pred=[]; diffsNoINL4PredNext=[]
  diffsNoINL2=[]; diffsNoINL2Next=[]

  errorsFB=[]; errorsNoFB=[]; errorsNoNoise=[]
  perfsFB = []; perfsNoFB = []

  #for probaZero in probaZeros:
  seed = 42
  for seedx in range(nbSeeds):

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

      inferenceErrors = []


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
      inferenceErrors.append(standardError)

      runError, activityFB, responsesFB = runInference(
          exp, noisySequences, enableFeedback=True)
      runError, activityNoFB, responsesNoFB = runInference(
          exp, noisySequences, enableFeedback=False)
      runError, activityNoAT, responsesNoAT = runInference(
          exp, noisySequences, enableFeedback=True, apicalTiebreak=False)
      runError, activityNoAT, responsesNoAM = runInference(
          exp, noisySequences, enableFeedback=True, apicalModulationBasalThreshold=False)
      runError, activityNoIN, responsesNoIN = runInference(
          exp, noisySequences, enableFeedback=True, inertia=False)

      # Now that actual processing is done, we compute various statistics and plot graphs.

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

          diffsNoAT.append( [len(responsesNoNoise[numseq]['L4Responses'][x].symmetric_difference(responsesNoAT[numseq]['L4Responses'][x])) for x in range(seqlen)] )
          overlapsNoATL2.append( [len(responsesNoNoise[numseq]['L2Responses'][x].intersection(responsesNoAT[numseq]['L2Responses'][x])) for x in range(seqlen)] )
          overlapsNoATL2Next.append( [len(responsesNoNoise[(numseq + 1) % numSequences]['L2Responses'][x].intersection(responsesNoAT[numseq]['L2Responses'][x])) for x in range(seqlen)] )
          overlapsNoATL4.append( [len(responsesNoNoise[numseq]['L4Responses'][x].intersection(responsesNoAT[numseq]['L4Responses'][x])) for x in range(seqlen)] )
          overlapsNoATL4Next.append( [len(responsesNoNoise[(numseq + 1) % numSequences]['L4Responses'][x].intersection(responsesNoAT[numseq]['L4Responses'][x])) for x in range(seqlen)] )
          diffsNoATL4Pred.append( [len(responsesNoNoise[numseq]['L4Responses'][x].symmetric_difference(responsesNoAT[numseq]['L4Predictive'][x])) for x in range(seqlen)] )

          diffsNoAM.append( [len(responsesNoNoise[numseq]['L4Responses'][x].symmetric_difference(responsesNoAM[numseq]['L4Responses'][x])) for x in range(seqlen)] )
          overlapsNoAML2.append( [len(responsesNoNoise[numseq]['L2Responses'][x].intersection(responsesNoAM[numseq]['L2Responses'][x])) for x in range(seqlen)] )
          overlapsNoAML2Next.append( [len(responsesNoNoise[(numseq + 1) % numSequences]['L2Responses'][x].intersection(responsesNoAM[numseq]['L2Responses'][x])) for x in range(seqlen)] )
          overlapsNoAML4.append( [len(responsesNoNoise[numseq]['L4Responses'][x].intersection(responsesNoAM[numseq]['L4Responses'][x])) for x in range(seqlen)] )
          overlapsNoAML4Next.append( [len(responsesNoNoise[(numseq + 1) % numSequences]['L4Responses'][x].intersection(responsesNoAM[numseq]['L4Responses'][x])) for x in range(seqlen)] )
          diffsNoAML4Pred.append( [len(responsesNoNoise[numseq]['L4Responses'][x].symmetric_difference(responsesNoAM[numseq]['L4Predictive'][x])) for x in range(seqlen)] )

          diffsNoIN.append( [len(responsesNoNoise[numseq]['L4Responses'][x].symmetric_difference(responsesNoIN[numseq]['L4Responses'][x])) for x in range(seqlen)] )
          overlapsNoINL2.append( [len(responsesNoNoise[numseq]['L2Responses'][x].intersection(responsesNoIN[numseq]['L2Responses'][x])) for x in range(seqlen)] )
          overlapsNoINL2Next.append( [len(responsesNoNoise[(numseq + 1) % numSequences]['L2Responses'][x].intersection(responsesNoIN[numseq]['L2Responses'][x])) for x in range(seqlen)] )
          overlapsNoINL4.append( [len(responsesNoNoise[numseq]['L4Responses'][x].intersection(responsesNoIN[numseq]['L4Responses'][x])) for x in range(seqlen)] )
          overlapsNoINL4Next.append( [len(responsesNoNoise[(numseq + 1) % numSequences]['L4Responses'][x].intersection(responsesNoIN[numseq]['L4Responses'][x])) for x in range(seqlen)] )
          diffsNoINL4Pred.append( [len(responsesNoNoise[numseq]['L4Responses'][x].symmetric_difference(responsesNoIN[numseq]['L4Predictive'][x])) for x in range(seqlen)] )


          cpcfb = []; cpcnofb=[]; cpcnoat=[]; cpcnoam=[]; cpcnoin=[];
          for x in range(seqlen):
              z1 = numpy.zeros(sdrlen+1); z1[list(responsesNoNoise[numseq]['L4Responses'][x])] = 1; z1[-1] = 1
              z2 = numpy.zeros(sdrlen+1); z2[list(responsesFB[numseq]['L4Predictive'][x])] = 1; z2[-1] = 1
              cpcfb.append(numpy.corrcoef(z1, z2)[0,1])
              z1 = numpy.zeros(sdrlen+1); z1[list(responsesNoNoise[numseq]['L4Responses'][x])] = 1; z1[-1] = 1
              z2 = numpy.zeros(sdrlen+1); z2[list(responsesNoFB[numseq]['L4Predictive'][x])] = 1; z2[-1] = 1
              cpcnofb.append(numpy.corrcoef(z1, z2)[0,1])
              z1 = numpy.zeros(sdrlen+1); z1[list(responsesNoNoise[numseq]['L4Responses'][x])] = 1; z1[-1] = 1
              z2 = numpy.zeros(sdrlen+1); z2[list(responsesNoAT[numseq]['L4Predictive'][x])] = 1; z2[-1] = 1
              cpcnoat.append(numpy.corrcoef(z1, z2)[0,1])
              z1 = numpy.zeros(sdrlen+1); z1[list(responsesNoNoise[numseq]['L4Responses'][x])] = 1; z1[-1] = 1
              z2 = numpy.zeros(sdrlen+1); z2[list(responsesNoAM[numseq]['L4Predictive'][x])] = 1; z2[-1] = 1
              cpcnoam.append(numpy.corrcoef(z1, z2)[0,1])
              z1 = numpy.zeros(sdrlen+1); z1[list(responsesNoNoise[numseq]['L4Responses'][x])] = 1; z1[-1] = 1
              z2 = numpy.zeros(sdrlen+1); z2[list(responsesNoIN[numseq]['L4Predictive'][x])] = 1; z2[-1] = 1
              cpcnoin.append(numpy.corrcoef(z1, z2)[0,1])

          # Note that the correlations are appended across all seeds and sequences
          corrsPredCorrectNoFBL4.append(cpcnofb[1:])
          corrsPredCorrectNoATL4.append(cpcnoat[1:])
          corrsPredCorrectNoINL4.append(cpcnoin[1:])
          corrsPredCorrectNoAML4.append(cpcnoam[1:])
          corrsPredCorrectFBL4.append(cpcfb[1:])

        #   diffsFBL2.append( [len(responsesNoNoise[numseq]['L2Responses'][x].symmetric_difference(responsesFB[numseq]['L2Responses'][x])) for x in range(seqlen)] )
        #   diffsNoFBL2.append( [len(responsesNoNoise[numseq]['L2Responses'][x].symmetric_difference(responsesNoFB[numseq]['L2Responses'][x])) for x in range(seqlen)] )
        #   diffsFBL2Next.append( [len(responsesNoNoise[(numseq + 1) % numSequences]['L2Responses'][x].symmetric_difference(responsesFB[numseq]['L2Responses'][x])) for x in range(seqlen)] )
        #   diffsNoFBL2Next.append( [len(responsesNoNoise[(numseq + 1) % numSequences]['L2Responses'][x].symmetric_difference(responsesNoFB[numseq]['L2Responses'][x])) for x in range(seqlen)] )

          print "Size of L2 responses (FB):", [len(responsesFB[numseq]['L2Responses'][x]) for x in range(seqlen)]
          print "Size of L2 responses (NoNoise):", [len(responsesNoNoise[numseq]['L2Responses'][x]) for x in range(seqlen)]
          print "Size of L4 responses (FB):", [len(responsesFB[numseq]['L4Responses'][x]) for x in range(seqlen)]
          print "Size of L4 responses (NoFB):", [len(responsesNoFB[numseq]['L4Responses'][x]) for x in range(seqlen)]
          print "Size of L4 responses (NoAT):", [len(responsesNoAT[numseq]['L4Responses'][x]) for x in range(seqlen)]
          print "Size of L4 responses (NoAM):", [len(responsesNoAM[numseq]['L4Responses'][x]) for x in range(seqlen)]
          print "Size of L4 responses (NoIN):", [len(responsesNoIN[numseq]['L4Responses'][x]) for x in range(seqlen)]
          print "Size of L4 responses (NoNoise):", [len(responsesNoNoise[numseq]['L4Responses'][x]) for x in range(seqlen)]
          print "Size of L4 predictions (FB):", [len(responsesFB[numseq]['L4Predictive'][x]) for x in range(seqlen)]
          print "Size of L4 predictions (NoFB):", [len(responsesNoFB[numseq]['L4Predictive'][x]) for x in range(seqlen)]
          print "Size of L4 predictions (NoAT):", [len(responsesNoAT[numseq]['L4Predictive'][x]) for x in range(seqlen)]
          print "Size of L4 predictions (NoAM):", [len(responsesNoAM[numseq]['L4Predictive'][x]) for x in range(seqlen)]
          print "Size of L4 predictions (NoIN):", [len(responsesNoIN[numseq]['L4Predictive'][x]) for x in range(seqlen)]
          print "Size of L4 predictions (NoNoise):", [len(responsesNoNoise[numseq]['L4Predictive'][x]) for x in range(seqlen)]
          print "L2 overlap with current (FB): ", overlapsFBL2[-1]
          print "L4 overlap with current (FB): ", overlapsFBL4[-1]
          print "L4 overlap with current (NoFB): ", overlapsNoFBL4[-1]
          print "L4 correlation pred/correct (FB): ", corrsPredCorrectFBL4[-1]
          print "L4 correlation pred/correct (NoFB): ", corrsPredCorrectNoFBL4[-1]
          print "L4 correlation pred/correct (NoAT): ", corrsPredCorrectNoATL4[-1]
          print "L4 correlation pred/correct (NoAM): ", corrsPredCorrectNoATL4[-1]
          print "L4 correlation pred/correct (NoIN): ", corrsPredCorrectNoATL4[-1]
          print "NoNoise sequence:", [list(x)[:2] for x in sequences[numseq]]
          print "Noise sequence:", [list(x)[:2] for x in noisySequences[numseq]]
          print "NoNoise L4 responses:", [list(x)[:2] for x in responsesNoNoise[numseq]['L4Responses']]
          print "NoFB L4 responses:", [list(x)[:2] for x in responsesNoFB[numseq]['L4Responses']]
          print ""

  plt.figure()
  allDataSets = (corrsPredCorrectFBL4, corrsPredCorrectNoFBL4, corrsPredCorrectNoATL4,
        corrsPredCorrectNoAML4, corrsPredCorrectNoINL4)
  allmeans = [numpy.mean(x) for x in allDataSets]
  allstds = [numpy.std(x) for x in allDataSets]
  nbbars = len(allmeans)
  plt.bar(2*(1+numpy.arange(nbbars))-.5, allmeans, 1.0, color='r', edgecolor='none', yerr=allstds, capsize=5, ecolor='k')
  for nn in range(1, nbbars):
      plt.vlines([2, 2 +2*nn], 1.2, 1.2+(nn/10.0), lw=2); plt.hlines(1.2+(nn/10.0), 2, 2+2*nn, lw=2)
      pval = scipy.stats.ranksums(numpy.array(corrsPredCorrectFBL4).ravel(), numpy.array(allDataSets[nn]).ravel())[1]
      if pval > 0.05:
          pvallabel = ' o' #r'$o$'
      elif pval > 0.01:
          pvallabel = '*'
      elif pval > 0.001:
          pvallabel = '**'
      else:
          pvallabel = '***'
      plt.text(3, 1.2+(nn/10.0)+.02, pvallabel, fontdict={"size":14})
  plt.xticks(2*(1+numpy.arange(nbbars)), ('Full', 'No\nFB', 'No Earlier\nFiring', 'No Thresold\nModulation', 'No Slower\nDynamics'))
  plt.ylabel("Avg. Prediction Performance");
  plt.title(plotTitle)
  plt.savefig(plotTitle+".png")
  # scipy.stats.ranksums(numpy.array(corrsPredCorrectFBL4).ravel(), numpy.array(corrsPredCorrectNoATL4).ravel())
  plt.show()
  return (corrsPredCorrectNoFBL4, corrsPredCorrectFBL4, corrsPredCorrectNoATL4, corrsPredCorrectNoAML4, corrsPredCorrectNoINL4)


if __name__ == "__main__":

  plt.ion()

  
  (corrsPredCorrectNoFBL4, corrsPredCorrectFBL4, corrsPredCorrectNoATL4,
        corrsPredCorrectNoAML4, corrsPredCorrectNoINL4) = runExp(noiseProba=.3,
        numSequences=5, nbSeeds=10, noiseType="pollute", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,30), whichPlot="corrspredcorrect", plotTitle="Individual effects: Continuous noise, shared range")

  (corrsPredCorrectNoFBL4, corrsPredCorrectFBL4, corrsPredCorrectNoATL4,
        corrsPredCorrectNoAML4, corrsPredCorrectNoINL4) = runExp(noiseProba=.3,
        numSequences=5, nbSeeds=10, noiseType="pollute", sequenceLen=30, sharedRange=(0,0), noiseRange=(0,30), whichPlot="corrspredcorrect", plotTitle="Individual effects: Continuous noise, no shared range")

  (corrsPredCorrectNoFBL4, corrsPredCorrectFBL4, corrsPredCorrectNoATL4,
        corrsPredCorrectNoAML4, corrsPredCorrectNoINL4) = runExp(noiseProba=.02,
        numSequences=5, nbSeeds=10, noiseType="replace", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,30), whichPlot="corrspredcorrect", plotTitle="Individual effects: Insert random stimulus, shared range")

  (corrsPredCorrectNoFBL4, corrsPredCorrectFBL4, corrsPredCorrectNoATL4,
        corrsPredCorrectNoAML4, corrsPredCorrectNoINL4) = runExp(noiseProba=.02,
        numSequences=5, nbSeeds=10, noiseType="replace", sequenceLen=30, sharedRange=(0,0), noiseRange=(0,30), whichPlot="corrspredcorrect", plotTitle="Individual effects: Insert random stimulus, no shared range")

  # (corrsPredCorrectNoFBL4, corrsPredCorrectFBL4, corrsPredCorrectNoATL4,
  #      corrsPredCorrectNoAML4, corrsPredCorrectNoINL4) = runExp(noiseProba=.25,
  #      numSequences=5, nbSeeds=10, noiseType="replace", sequenceLen=30, sharedRange=(5,24), noiseRange=(0,30), whichPlot="corrspredcorrect", plotTitle="Individual effects: Random insert + continuous noise, shared range")
  #
  # (corrsPredCorrectNoFBL4, corrsPredCorrectFBL4, corrsPredCorrectNoATL4,
  #      corrsPredCorrectNoAML4, corrsPredCorrectNoINL4) = runExp(noiseProba=.25,
  #      numSequences=5, nbSeeds=10, noiseType="replace", sequenceLen=30, sharedRange=(0,0), noiseRange=(0,30), whichPlot="corrspredcorrect", plotTitle="Individual effects: Random insert + continuous noise, no shared range")
