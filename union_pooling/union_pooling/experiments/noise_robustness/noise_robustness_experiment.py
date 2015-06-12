#!/usr/bin/env python
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

import csv
import numpy
from optparse import OptionParser
import os
import random
import sys
import time
import yaml

from nupic.data.generators.pattern_machine import PatternMachine
from nupic.data.generators.sequence_machine import SequenceMachine
from nupic.research.monitor_mixin.monitor_mixin_base import MonitorMixinBase

from union_pooling.experiments.union_pooler_experiment import (
    UnionPoolerExperiment)

"""
Noise Robustness Experiment

Data: Sequences generated from an alphabet.

Train Phase: Train network on sequences for some number of repetitions

Test phase: Input sequence pattern by pattern. Sequence-to-sequence
progression is randomly selected. At each step there is a chance that the
next pattern in the sequence is not shown. Specifically the following
perturbations may occur:

  1) random Jump to another sequence
  2) substitution of some other pattern for the normal expected pattern
  3) skipping expected pattern and presenting next pattern in sequence
  4) addition of some other pattern putting off expected pattern one time step

Goal: Characterize the noise robustness of the UnionPooler to various
perturbations. Explore trade-off between remaining stable to noise yet still
changing when sequence actually changes.
"""



_SHOW_PROGRESS_INTERVAL = 2000



def runTestPhase(experiment, inputSequences, inputCategories, sequenceCount,
                 sequenceLength, testPresentations, perturbationChance,
                 sequenceJumpPerturbationChance, consoleVerbosity):
  """
  Performs a number of presentations of sequences with resets afterwards.
  Sequence selection is random.
  At each step of sequence presentation there is a chance of a perturbation.
  Specifically the following perturbations may occur:
    1) Substitution of expected pattern with some other random pattern among
    the set of sequences
    2) Skipping of expected pattern and advancing to next pattern in the
    sequence
    3) Insertion of some other random pattern (among the set of sequences)
    that delays the expected pattern one time step

  @return
        actualCategories -    A list of the actual categories of the patterns
                              presented during the test phase
        classifiedCategores - A list of the classifications of the categories
                              of each pattern presented during the test phase
  """
  actualCategories = []
  classifiedCategories = []

  # Compute the bounds for a wheel-of-fortune style roll
  pert1ChanceThreshold = (1 - sequenceJumpPerturbationChance) * (1 / 3.0)
  pert2ChanceThreshold = (1 - sequenceJumpPerturbationChance) * (2 / 3.0)
  pert3ChanceThreshold = (1 - sequenceJumpPerturbationChance)

  presentation = 0
  presentationString = "Presentation 0: "
  pert0Count = 0
  pert1Count = 0
  pert2Count = 0
  pert3Count = 0
  while presentation < testPresentations:

    # Randomly select the next sequence to present
    r = random.randint(0, sequenceCount - 1)
    start = r + r * sequenceLength
    end = r + 1 + (r + 1) * sequenceLength
    selectedSequence = inputSequences[start:end]
    selectedSequenceCatgories = inputCategories[start:end]

    if consoleVerbosity > 0:
      presentationString += "Seq-{0} ".format(r)

    # Present selected sequence to network
    i = 0
    while i < len(selectedSequence):

      # Roll to determine if there will be a perturbation of next pattern
      if (selectedSequence[i] is not None and
          random.random() < perturbationChance):

        # Randomly select a perturbation type with equal probability
        perturbationType = random.random()
        if perturbationType < pert1ChanceThreshold:

          # Substitution with random pattern
          randIdx = getRandomPatternIndex(inputSequences)
          sensorPattern = inputSequences[randIdx]
          actualCategory = inputCategories[randIdx]
          i += 1
          pert0Count += 1
        elif perturbationType < pert2ChanceThreshold:

          # Skip to next pattern in sequence
          i += 1
          if i == len(selectedSequence):
            break;
          sensorPattern = selectedSequence[i]
          actualCategory = selectedSequenceCatgories[i]
          i += 1
          pert1Count += 1
        elif perturbationType < pert3ChanceThreshold:

          # Add in a random pattern
          randIdx = getRandomPatternIndex(inputSequences)
          sensorPattern = inputSequences[randIdx]
          actualCategory = inputCategories[randIdx]
          pert2Count += 1
        else:

          # Random jump to another sequence
          pert3Count += 1
          break

      else:
        sensorPattern = selectedSequence[i]
        actualCategory = selectedSequenceCatgories[i]
        i += 1
      experiment.runNetworkOnPattern(sensorPattern,
                                     tmLearn=False,
                                     upLearn=False)

      if sensorPattern is not None:
        # Store classification
        unionSDR = experiment.up.getUnionSDR()
        denseUnionSDR = numpy.zeros(experiment.up.getNumColumns())
        denseUnionSDR[unionSDR] = 1.0
        winningCategory, _, _, _ = experiment.classifier.infer(denseUnionSDR)
        # print "Classif: {0} for {1}".format(winningCategory, unionSDR)
        # overlaps, categories = experiment.classifier.getOverlaps(unionSDR)
        # print "Overlaps " + str(overlaps)
        # print "Categories " + str(categories)
        # print
      else:
        winningCategory = None
        actualCategory = None

      classifiedCategories.append(winningCategory)
      actualCategories.append(actualCategory)

    # While presenting sequence
    else:

      # Presentation finished; prepare for next one
      if consoleVerbosity > 0:
        print presentationString
      presentation += 1
      presentationString = "Presentation {0}: ".format(presentation)

    # Finished with sequence presentation

  # While running test presentations
  # Sequence perturbation counts
  if consoleVerbosity > 0:
    print ("\nPerturbation Summary: PatternSub: {0} PatternSkip: {1} "
           "PatternAdd: {2} SequenceSkip {3}\n").format(pert0Count,
                                                        pert1Count,
                                                        pert2Count,
                                                        pert3Count)

  return actualCategories, classifiedCategories



def getRandomPatternIndex(patterns):
  rand = random.randint(0, len(patterns)-1)
  while patterns[rand] is None:
    rand = random.randint(0, len(patterns)-1)
  return rand



def run(params, paramDir, outputDir, consoleVerbosity=0):
  """
  Runs the noise robustness experiment.

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
  print "Running Noise robustness experiment...\n"
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
  perturbationChance = params["perturbationChance"]
  sequenceJumpPerturbationChance = params["sequenceJumpPerturbationChance"]
  tmParamOverrides = params["temporalMemoryParams"]
  upParamOverrides = params["unionPoolerParams"]
  classifierOverrides = params["classifierParams"]

  # Generate a sequence list and an associated labeled list (both containing a
  # set of sequences separated by None)
  print "Generating sequences..."
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

  if consoleVerbosity > 2:
    for i in xrange(len(inputSequences)):
      if inputSequences[i] is None:
        print
      else:
        print inputSequences[i]
        print inputCategories[i]

  # Set up the Temporal Memory and Union Pooler network
  print "\nCreating network..."
  experiment = UnionPoolerExperiment(tmOverrides=tmParamOverrides,
                                     upOverrides=upParamOverrides,
                                     classifierOverrides=classifierOverrides,
                                     consoleVerbosity=0)

  # Training only the Temporal Memory on the generated sequences
  print "\nTraining Temporal Memory..."
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

  # With learning off, but TM and UP running, train the classifier.
  print "\nTraining Classifier..."
  classifResString = ""
  for i in xrange(trainingPasses):
    experiment.runNetworkOnSequences(inputSequences,
                                     inputCategories,
                                     tmLearn=False,
                                     upLearn=False,
                                     classifierLearn=True,
                                     verbosity=consoleVerbosity,
                                     progressInterval=_SHOW_PROGRESS_INTERVAL)

    classifResString +=  "{0}\t\t{1}\t\t{2}\n".format(i,
      experiment.classifier._numPatterns, numberOfSequences)
    experiment.tm.mmClearHistory()
    experiment.up.mmClearHistory()

  if consoleVerbosity > 1:
    print "Pass\tClassifier Patterns\tUnique Sequences"
    print classifResString

  print "\nRunning test phase..."
  actualCategories, classifiedCategories = runTestPhase(experiment,
                                                        inputSequences,
                                                        inputCategories,
                                                        numberOfSequences,
                                                        sequenceLength,
                                                        testPresentations,
                                                        perturbationChance,
                                                        sequenceJumpPerturbationChance,
                                                        consoleVerbosity)

  # Classification stats
  correctClassifications = 0.0
  numberClassifications = 0.0
  classificationVector = []
  for i in xrange(len(actualCategories)):
    if actualCategories[i] is not None:
      numberClassifications += 1
      if actualCategories[i] == classifiedCategories[i]:
        correctClassifications += 1
        classificationVector.append(1)
      else:
        classificationVector.append(0)
    else:
      classificationVector.append("*")
  classificationRate = 100.0 * correctClassifications / numberClassifications

  # Output
  print "\n>>> Correct Classification Rate: {0:.2f}%".format(classificationRate)

  outputFileName = "testPres{0}_perturbationRate{1}_jumpRate{2}.txt".format(
    testPresentations, perturbationChance, sequenceJumpPerturbationChance)
  print "\nWriting results to {0}/{1}".format(outputDir, outputFileName)

  elapsedTime = (time.time() - startTime) / 60.0
  print "\nFinished in {0:.2f} minutes.".format(elapsedTime)

  writeClassificationTrace(actualCategories, classifiedCategories,
                           classificationVector, [classificationRate],
                           [elapsedTime], outputDir, outputFileName)



def writeClassificationTrace(actualCategories, classifiedCategories,
                             classificationVector, classificationStats,
                             elapsedTime, outputDir, outputFileName):
  """
  Write classification trace to output file.
  :param actualCategories: True categories
  :param classifiedCategories: Classified categories
  :param classificationStats: List of stats of classification performance
  :param outputDir: dir where output file will be written
  :param outputFileName: filename of output file
  """
  if not os.path.exists(outputDir):
    os.makedirs(outputDir)

  filePath = os.path.join(outputDir, outputFileName)
  with open(filePath, "wb") as outputFile:
    csvWriter = csv.writer(outputFile)
    csvWriter.writerow(["Actual Categories"])
    csvWriter.writerow(actualCategories)
    csvWriter.writerow(["Classified Categories"])
    csvWriter.writerow(classifiedCategories)
    csvWriter.writerow(["Correct Classifications"])
    csvWriter.writerow(classificationVector)
    csvWriter.writerow(["Classification Statistics"])
    csvWriter.writerow(classificationStats)
    csvWriter.writerow(["Elapsed Time"])
    csvWriter.writerow(elapsedTime)
    outputFile.flush()



def _getArgs():
  parser = OptionParser(usage="%prog PARAMS_DIR OUTPUT_DIR [options]"
                              "\n\nRun noise robustness experiment using "
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
