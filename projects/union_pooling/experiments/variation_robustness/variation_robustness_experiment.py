#!/usr/bin/env python
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

import csv
import numpy
from optparse import OptionParser
import os
import pprint
import random
import sys
import time
import yaml

from nupic.data.generators.pattern_machine import PatternMachine
from nupic.data.generators.sequence_machine import SequenceMachine
from nupic.algorithms.monitor_mixin.monitor_mixin_base import MonitorMixinBase

from union_temporal_pooling.experiments.union_temporal_pooler_experiment import (
    UnionTemporalPoolerExperiment)

"""
Variation Robustness Experiment

Data: Sequences generated from an alphabet. Patterns do not overlap between
sequences.

Train Phase: Train network on sequences for some number of repetitions, then
train KNN Classifier on sequences learning the same category label for each
element in the sequence.

Test phase: Input sequence pattern by pattern. Sequence-to-sequence
progression is randomly selected. At each step there is a chance that the
next pattern in the sequence is not shown. Specifically the following
perturbations may occur:

  1) random Jump to another sequence
  2) substitution of some other pattern for the normal expected pattern
  3) skipping expected pattern and presenting next pattern in sequence
  4) addition of some other pattern putting off expected pattern one time step

Goal: Characterize the variation robustness of the UnionTemporalPooler to various
perturbations. Explore trade-off between remaining stable to variations yet
still changing when sequence actually changes.
"""



_SHOW_PROGRESS_INTERVAL = 3000



class PerturbationType(object):
  """An enum class defining the different types of perturbations used by this
  experiment.
  """

  # No perturbation
  none = 0

  # Perturbation switching expected pattern for another random pattern
  substitution = 1

  # Perturbation skipping over expected pattern and continuing with next
  # expected pattern
  skip = 2

  # Perturbation adding in a random pattern delaying expected pattern
  add = 3

  # Perturbation switching to entirely different sequence from its beginning
  sequenceJump = 4



def runTestPhaseRandom(experiment, inputSequences, sequenceCount,
                       sequenceLength, testPresentations,
                       overallPerturbationChance, perturbationTypeChance,
                       consoleVerbosity):
  """
  Performs a number of presentations of sequences with resets afterwards.
  Sequence selection is random.
  At each step of sequence presentation there is a chance of a perturbation.
  Specifically the following perturbations may occur:
    1) Substitution of expected pattern with some other random pattern among
    the set of sequences
    2) Insertion of some other random pattern (among the set of sequences)
    that delays the expected pattern one time step
    3) Skipping of expected pattern and advancing to next pattern in the
    sequence
    4) Jump from current sequence to another randomly selected sequence

  @param experiment                 A UnionTemporalPoolerExperiment
  @param inputSequences             List of sequences each terminated by None.
  @param sequenceCount              The number of sequences in inputSequences
  @param sequenceLength             Length of each sequence not counting Nones.
  @param testPresentations          Number of sequences randomly selected and
                                    presented during the test phase. Sequence
                                    jumps do not count towards the number of
                                    presentations, rather an entire sequence
                                    must be presented (without sequence jumps)
                                    before advancing to the next test
                                    presentation.
  @param overallPerturbationChance  Rate of perturbations during the test phase
  @param perturbationTypeChance     A list of relative chances for each
                                    perturbation type:
                                    0 - substitution chance
                                    1 - addition chance
                                    2 - skip chance
                                    3 - sequence jump chance
                                    Note the chances do not need to sum to 1.0,
                                    and the relative weight of each of each
                                    type chance is what determines likelihood.
  @param consoleVerbosity           Console output verbosity

  @return
        actualCategories -    A list of the actual categories of the patterns
                              presented during the test phase
        classifiedCategores - A list of the classifications of the categories
                              of each pattern presented during the test phase
        perturbationTrace -   A list of the perturbations that occurred during
                              the test phase
  """
  actualCategories = []
  classifiedCategories = []
  perturbationTrace = []

  substitutionChance = perturbationTypeChance[0]
  additionChance = perturbationTypeChance[1]
  skipChance = perturbationTypeChance[2]
  sequenceJumpChance = perturbationTypeChance[3]

  totalTypeChance = (substitutionChance + additionChance + skipChance +
                     sequenceJumpChance)

  # Compute the bounds for a wheel-of-fortune style roll
  patternSubChanceThreshold = float(substitutionChance) / totalTypeChance
  patternAddChanceThreshold = (float(substitutionChance + additionChance) /
                               totalTypeChance)
  patternSkipChanceThreshold = float(substitutionChance + additionChance +
                                     skipChance) / totalTypeChance

  if consoleVerbosity > 0:
    presentationString = "Presentation 0: "

  presentation = 0
  isStartOfSequenceJump = False
  while presentation < testPresentations:

    # Randomly select the next sequence to present
    sequence = random.randint(0, sequenceCount - 1)

    if consoleVerbosity > 0:
      presentationString += "Seq-{0} ".format(sequence)

    # Present selected sequence to network
    sequenceStart = sequence + sequence * sequenceLength
    i = sequenceStart
    sequenceEnd = sequence + 1 + (sequence + 1) * sequenceLength
    while i < sequenceEnd:

      # Roll to determine if there will be a perturbation of next pattern
      if (inputSequences[i] is not None and
          random.random() < overallPerturbationChance):

        # Randomly select a perturbation type
        perturbationType = random.random()
        if perturbationType < patternSubChanceThreshold:

          # Substitute in a random pattern and move on to next pattern
          # in sequence
          currentPattern = getRandomPattern(inputSequences, sequenceStart,
                                            sequenceEnd)
          currentPerturbation = PerturbationType.substitution
          i += 1
        elif perturbationType < patternAddChanceThreshold:

          # Add in an extra random pattern
          currentPattern = getRandomPattern(inputSequences, sequenceStart,
                                            sequenceEnd)
          currentPerturbation = PerturbationType.add
        elif perturbationType < patternSkipChanceThreshold:

          # Skip to next pattern in sequence
          i += 1
          if i == sequenceEnd:
            experiment.runNetworkOnPattern(None,
                                           tmLearn=False,
                                           upLearn=False)
            break;
          currentPattern = inputSequences[i]
          currentPerturbation = PerturbationType.skip
          i += 1
        else:

          # Random jump to another sequence
          isStartOfSequenceJump = True
          break
      else:
        # Normal advancement of sequence
        currentPattern = inputSequences[i]
        if isStartOfSequenceJump:
          currentPerturbation = PerturbationType.sequenceJump
          isStartOfSequenceJump = False
        else:
          currentPerturbation = PerturbationType.none
        i += 1

      experiment.runNetworkOnPattern(currentPattern,
                                     tmLearn=False,
                                     upLearn=False)

      if currentPattern is not None:
        # Store classification
        unionSDR = experiment.up.getUnionSDR()
        denseUnionSDR = numpy.zeros(experiment.up.getNumColumns())
        denseUnionSDR[unionSDR] = 1.0
        classification, _, _, _ = experiment.classifier.infer(denseUnionSDR)

        # Assumes sequence number and sequence category is equivalent
        actualCategories.append(sequence)
        classifiedCategories.append(classification)
        perturbationTrace.append(currentPerturbation)

    # While presenting sequence
    else:
      # Move to next presentation only if a sequence has been completed
      # without any sequence jumps
      presentation += 1
      isStartOfSequenceJump = False

      # Presentation finished; prepare for next one
      if consoleVerbosity > 0:
        print presentationString
        presentationString = "Presentation {0}: ".format(presentation)

    # Finished sequence presentation

  # While running test presentations

  if consoleVerbosity > 0:
    patternSubCount = perturbationTrace.count(PerturbationType.substitution)
    patternSkipCount = perturbationTrace.count(PerturbationType.skip)
    patternAddCount = perturbationTrace.count(PerturbationType.add)
    sequenceJumpCount = perturbationTrace.count(PerturbationType.sequenceJump)
    print ("\nPerturbation Counts: "
           "\nPatternSub: {0} "
           "\nPatternSkip: {1} "
           "\nPatternAdd: {2} "
           "\nSequenceJump {3}").format(patternSubCount, patternSkipCount,
                                        patternAddCount, sequenceJumpCount)

  return actualCategories, classifiedCategories, perturbationTrace



def getRandomPattern(patterns, ignoreStart, ignoreEnd):
  r = random.randint(0, len(patterns)-1)
  while (ignoreStart <= r <= ignoreEnd) or (patterns[r] is None):
    r = random.randint(0, len(patterns)-1)
  return patterns[r]



def getPerturbedSequences(inputSequences, sequenceCount, sequenceLength,
                          exactSubstitutions):
  perturbationTrace = [0] * (sequenceCount * sequenceLength)
  perturbedSequences = list(inputSequences)
  for i in xrange(sequenceCount):
    start = i + i * sequenceLength
    end = i + 1 + (i + 1) * sequenceLength

    # end - 1 because we don't want the None
    sequenceIndices = range(start, end - 1)
    subsample = random.sample(sequenceIndices, exactSubstitutions)
    for j in subsample:
      perturbedSequences[j] = getRandomPattern(inputSequences, start, end - 2)

      # Must subtract number of Nones
      perturbationTrace[j - i] = PerturbationType.substitution

  return perturbedSequences, perturbationTrace



def runTestPhaseFixed(experiment, inputSequences, sequenceCount, sequenceLength,
                      exactSubstitutions, consoleVerbosity):
  """
  Runs a test phase where a fixed number of substitutions perturbations are
  performed, i.e. chance does not affect the number of substitutions that occur.
  Random chance does still affect where these perturbation occur in each
  sequence.
  @param experiment                 A UnionTemporalPoolerExperiment
  @param inputSequences             List of sequences each terminated by None.
  @param sequenceCount              The number of sequences in inputSequences
  @param sequenceLength             Length of each sequence not counting Nones.

  @param exactSubstitutions         The number of substitution perturbations
                                    guaranteed to be made in each sequence.
  @param consoleVerbosity           Console output verbosity

  @return
        actualCategories -    A list of the actual categories of the patterns
                              presented during the test phase
        classifiedCategores - A list of the classifications of the categories
                              of each pattern presented during the test phase
        perturbationTrace -   A list of the perturbations that occurred during
                              the test phase
  """
  actualCategories = []
  classifiedCategories = []

  perturbedSequences, perturbationTrace = getPerturbedSequences(inputSequences,
                                                                sequenceCount,
                                                                sequenceLength,
                                                                exactSubstitutions)
  for i in xrange(len(perturbedSequences)):
    experiment.runNetworkOnPattern(perturbedSequences[i],
                                   tmLearn=False,
                                   upLearn=False)

    if perturbedSequences[i] is not None:
      # Store classification
      unionSDR = experiment.up.getUnionSDR()
      denseUnionSDR = numpy.zeros(experiment.up.getNumColumns())
      denseUnionSDR[unionSDR] = 1.0
      classification, _, _, _ = experiment.classifier.infer(denseUnionSDR)

      # Assumes sequence number and sequence category is equivalent
      actualCategories.append(i / (sequenceLength + 1))
      classifiedCategories.append(classification)

  return actualCategories, classifiedCategories, perturbationTrace



def trainTemporalMemory(experiment, inputSequences, inputCategories,
                        trainingPasses, consoleVerbosity):
  burstingColsString = ""
  for i in xrange(trainingPasses):
    experiment.runNetworkOnSequences(inputSequences,
                                     inputCategories,
                                     tmLearn=True,
                                     upLearn=None,
                                     classifierLearn=False,
                                     verbosity=consoleVerbosity,
                                     progressInterval=_SHOW_PROGRESS_INTERVAL)

    if consoleVerbosity > 1:
      print
      print MonitorMixinBase.mmPrettyPrintMetrics(
        experiment.tm.mmGetDefaultMetrics())
      print
    stats = experiment.getBurstingColumnsStats()
    burstingColsString += "{0}\t{1}\t{2}\t{3}\n".format(i, stats[0], stats[1],
                                                        stats[2])

    experiment.tm.mmClearHistory()
    experiment.up.mmClearHistory()

  if consoleVerbosity > 0:
    print "\nTemporal Memory Bursting Columns stats..."
    print "Pass\tMean\t\tStdDev\t\tMax"
    print burstingColsString



def trainClassifier(experiment, inputSequences, inputCategories,
                    numberOfSequences, trainingPasses, consoleVerbosity):
  classifResString = ""
  for i in xrange(trainingPasses):
    experiment.runNetworkOnSequences(inputSequences,
                                     inputCategories,
                                     tmLearn=False,
                                     upLearn=False,
                                     classifierLearn=True,
                                     verbosity=consoleVerbosity,
                                     progressInterval=_SHOW_PROGRESS_INTERVAL)

    classifResString += "{0}\t\t{1}\t\t{2}\n".format(i,
                                                     experiment.classifier._numPatterns,
                                                     numberOfSequences)
    experiment.tm.mmClearHistory()
    experiment.up.mmClearHistory()
  if consoleVerbosity > 1:
    print "Pass\tClassifier Patterns\tUnique Sequences"
    print classifResString



def generateSequences(patternCardinality, patternDimensionality,
                      numberOfSequences, sequenceLength, consoleVerbosity):
  patternAlphabetSize = sequenceLength * numberOfSequences
  patternMachine = PatternMachine(patternDimensionality, patternCardinality,
                                  patternAlphabetSize)
  sequenceMachine = SequenceMachine(patternMachine)
  numbers = sequenceMachine.generateNumbers(numberOfSequences, sequenceLength)
  inputSequences = sequenceMachine.generateFromNumbers(numbers)
  inputCategories = []
  for i in xrange(numberOfSequences):
    for _ in xrange(sequenceLength):
      inputCategories.append(i)
    inputCategories.append(None)
  if consoleVerbosity > 1:
    for i in xrange(len(inputSequences)):
      if inputSequences[i] is None:
        print
      else:
        print "{0} {1}".format(inputSequences[i], inputCategories[i])

  return inputSequences, inputCategories



def run(params, paramDir, outputDir, consoleVerbosity=0, plotVerbosity=0):
  """
  Runs the variation robustness experiment.

  :param params: A dict containing the following experiment parameters:

        patternDimensionality -       Dimensionality of sequence patterns
        patternCardinality -          Cardinality (# ON bits) of sequence
                                      patterns
        sequenceLength -              Length of sequences shown to network
        numberOfSequences -           Number of unique sequences used
        trainingPasses -              Number of times Temporal Memory is trained
                                      on each sequence
        testPresentations -           Number of sequences presented in test
                                      phase
        perturbationChance -          Chance of sequence perturbations during
                                      test phase
        sequenceJumpPerturbationChance - Chance of a jump-sequence perturbation
                                         type
        temporalMemoryParams -        A dict of Temporal Memory parameter
                                      overrides
        unionPoolerParams -           A dict of Union Pooler parameter overrides
        classifierParams -            A dict of KNNClassifer parameter overrides

  :param paramDir: Path of parameter file
  :param outputDir: Output will be written to this path
  :param consoleVerbosity: Console output verbosity
  """
  startTime = time.time()
  print "Running Variation robustness experiment...\n"
  print "Params dir: {0}".format(os.path.join(os.path.dirname(__file__),
                                              paramDir))
  print "Output dir: {0}\n".format(os.path.join(os.path.dirname(__file__),
                                                outputDir))

  patternDimensionality = params["patternDimensionality"]
  patternCardinality = params["patternCardinality"]
  sequenceLength = params["sequenceLength"]
  numberOfSequences = params["numberOfSequences"]
  trainingPasses = params["trainingPasses"]
  testPresentations = params["testPresentations"]

  exactSubstitutions = (params["exactSubstitutions"] if "exactSubstitutions" in
                        params else None)

  perturbationChance = params["perturbationChance"]
  sequenceJumpChance = params["sequenceJumpPerturbationChance"]

  # These if-else/s are for backwards compatibility with older param files that
  # didn't specify these chances
  if "substitutionPerturbationChance" in params:
    substitutionChance = params["substitutionPerturbationChance"]
  else:
    substitutionChance = (1 - sequenceJumpChance) / 3.0

  if "addPerturbationChance" in params:
    addChance = params["addPerturbationChance"]
  else:
    addChance = (1 - sequenceJumpChance) / 3.0

  if "skipChance" in params:
    skipChance = params["skipChance"]
  else:
    skipChance = (1 - sequenceJumpChance) / 3.0

  perturbationTypeChance = [substitutionChance, addChance, skipChance,
                            sequenceJumpChance]
  tmParamOverrides = params["temporalMemoryParams"]
  upParamOverrides = params["unionPoolerParams"]
  classifierOverrides = params["classifierParams"]

  # Generate a sequence list and an associated labeled list (both containing a
  # set of sequences separated by None)
  print "Generating sequences..."
  inputSequences, inputCategories = generateSequences(patternCardinality,
                                                      patternDimensionality,
                                                      numberOfSequences,
                                                      sequenceLength,
                                                      consoleVerbosity)

  # Set up the Temporal Memory and Union Pooler network
  print "\nCreating network..."
  experiment = UnionTemporalPoolerExperiment(tmOverrides=tmParamOverrides,
                                     upOverrides=upParamOverrides,
                                     classifierOverrides=classifierOverrides,
                                     consoleVerbosity=0)

  # Training only the Temporal Memory on the generated sequences
  print "\nTraining Temporal Memory..."
  trainTemporalMemory(experiment, inputSequences, inputCategories,
                      trainingPasses, consoleVerbosity)

  # With learning off, but TM and UP running, train the classifier.
  print "\nTraining Classifier..."
  trainClassifier(experiment, inputSequences, inputCategories,
                  numberOfSequences, trainingPasses, consoleVerbosity)

  print "\nRunning Test Phase..."
  if exactSubstitutions is None:
    (actualCategories,
     classifiedCategories,
     perturbationTrace) = runTestPhaseRandom(experiment,
                                             inputSequences,
                                             numberOfSequences,
                                             sequenceLength,
                                             testPresentations,
                                             perturbationChance,
                                             perturbationTypeChance,
                                             consoleVerbosity)
  else:
    (actualCategories,
     classifiedCategories,
     perturbationTrace) = runTestPhaseFixed(experiment,
                                            inputSequences,
                                            numberOfSequences,
                                            sequenceLength,
                                            exactSubstitutions,
                                            consoleVerbosity)

  assert len(actualCategories) == len(classifiedCategories)
  assert len(actualCategories) == len(perturbationTrace)

  correctClassificationTrace = [1 if (actualCategories[i] ==
                                      classifiedCategories[i]) else 0
                                for i in xrange(len(actualCategories))]
  correctClassifications = correctClassificationTrace.count(1)
  classificationRate = 100.0 * correctClassifications / len(actualCategories)

  # Classification results
  print "\n*Results*"
  pprint.pprint("Actual Category {0}".format(actualCategories))
  pprint.pprint("Classification  {0}".format(classifiedCategories))
  pprint.pprint("Class. Correct  {0}".format(
    correctClassificationTrace))
  pprint.pprint("Perturb Type    {0}".format(perturbationTrace))
  numPerturbations = (len(perturbationTrace) -
                      perturbationTrace.count(PerturbationType.none))

  errorDict = {PerturbationType.none: 0,
               PerturbationType.substitution: 0,
               PerturbationType.skip: 0,
               PerturbationType.add: 0,
               PerturbationType.sequenceJump: 0}

  incorrect = 0
  for i in xrange(len(actualCategories)):
    if actualCategories[i] != classifiedCategories[i]:
      errorDict[perturbationTrace[i]] += 1.0
      incorrect += 1

  print "\n*** Correct Classification Rate: {0:.2f}%".format(classificationRate)
  print "*** Correct / Total: \t{0} / {1}".format(correctClassifications,
                                                len(correctClassificationTrace))

  if exactSubstitutions is None:
    actualPerturbationRate = 100.0 * numPerturbations / len(perturbationTrace)
    print "\nActual perturbation rate: {0:.2f}%".format(actualPerturbationRate)

  substitutionErrorRate = (0 if incorrect == 0 else
    100.0 * errorDict[PerturbationType.substitution] / incorrect)
  skipErrorRate = (0 if incorrect == 0 else
    100.0 * errorDict[PerturbationType.skip] / incorrect)
  addErrorRate = (0 if incorrect == 0 else
    100.0 * errorDict[PerturbationType.add] / incorrect)
  sequenceJumpErrorRate = (0 if incorrect == 0 else
    100.0 * errorDict[PerturbationType.sequenceJump] / incorrect)
  noPerturbationErrorRate = (0 if incorrect == 0 else
    100.0 * errorDict[PerturbationType.none] / incorrect)
  print "\nError Rate by Perturbation:"
  print (  "Substitution:    \t{0:.2f}% "
         "\nSkip Pattern:    \t{1:.2f}% "
         "\nAdd Pattern:     \t{2:.2f}% "
         "\nSequence Jump:   \t{3:.2f}% "
         "\nNo Perturbation: \t{4:.2f}%").format(substitutionErrorRate,
                                                 skipErrorRate,
                                                 addErrorRate,
                                                 sequenceJumpErrorRate,
                                                 noPerturbationErrorRate)

  outputFileName = ("testPresentations{0:0>3}_perturbationRate{"
                    "1:0>3}_exactSubstitutions{2:0>3}.txt").format(
    testPresentations, perturbationChance, exactSubstitutions)
  print "\nWriting results to {0}/{1}".format(outputDir, outputFileName)

  elapsedTime = (time.time() - startTime) / 60.0
  print "\nFinished in {0:.2f} minutes.".format(elapsedTime)

  writeClassificationTrace(outputDir, outputFileName, classificationRate)



def writeClassificationTrace(outputDir, outputFileName, mean):
  """
  Write classification trace to output file.
  :param outputDir: dir where output file will be written
  :param outputFileName: filename of output file
  :param mean: Mean classification performance
  """
  if not os.path.exists(outputDir):
    os.makedirs(outputDir)

  filePath = os.path.join(outputDir, outputFileName)
  with open(filePath, "wb") as outputFile:
    csvWriter = csv.writer(outputFile)
    csvWriter.writerow(["Classification Statistics"])
    csvWriter.writerow([mean])
    outputFile.flush()



def _getArgs():
  parser = OptionParser(usage="%prog PARAMS_DIR OUTPUT_DIR [options]"
                              "\n\nRun variation robustness experiment using "
                              "params in PARAMS_DIR (relative to this file) "
                              "and outputting results to OUTPUT_DIR.")
  parser.add_option("-c",
                    "--console",
                    type=int,
                    default=0,
                    dest="consoleVerbosity",
                    help="Console message verbosity: 0 => none")
  (options, args) = parser.parse_args(sys.argv[1:])
  if len(args) < 2:
    parser.print_help(sys.stderr)
    sys.exit()

  absPath = os.path.join(os.path.dirname(__file__), args[0])
  with open(absPath) as paramsFile:
    params = yaml.safe_load(paramsFile)

  return options, args, params



if __name__ == "__main__":
  (_options, _args, _params) = _getArgs()
  run(_params, _args[0], _args[1], _options.consoleVerbosity)
