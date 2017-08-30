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
import sys
import time
import os
import yaml
from optparse import OptionParser

import numpy
from pylab import rcParams

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
plt.ion()

from nupic.data.generators.pattern_machine import PatternMachine
from nupic.data.generators.sequence_machine import SequenceMachine
from nupic.algorithms.monitor_mixin.monitor_mixin_base import MonitorMixinBase

from htmresearch.frameworks.union_temporal_pooling.union_temporal_pooler_experiment import (
    UnionTemporalPoolerExperiment)

"""
Experiment 2
Runs UnionTemporalPooler on input from a Temporal Memory while TM learns the sequence
"""

def experiment2():
  paramDir = 'params/1024_baseline/5_trainingPasses.yaml'
  outputDir = 'results/'
  params = yaml.safe_load(open(paramDir, 'r'))
  options = {'plotVerbosity': 2, 'consoleVerbosity': 2}
  plotVerbosity = 2
  consoleVerbosity = 1


  print "Running SDR overlap experiment...\n"
  print "Params dir: {0}".format(paramDir)
  print "Output dir: {0}\n".format(outputDir)

  # Dimensionality of sequence patterns
  patternDimensionality = params["patternDimensionality"]

  # Cardinality (ON / true bits) of sequence patterns
  patternCardinality = params["patternCardinality"]

  # TODO If this parameter is to be supported, the sequence generation code
  # below must change
  # Number of unique patterns from which sequences are built
  # patternAlphabetSize = params["patternAlphabetSize"]

  # Length of sequences shown to network
  sequenceLength = params["sequenceLength"]

  # Number of sequences used. Sequences may share common elements.
  numberOfSequences = params["numberOfSequences"]

  # Number of sequence passes for training the TM. Zero => no training.
  trainingPasses = params["trainingPasses"]

  tmParamOverrides = params["temporalMemoryParams"]
  upParamOverrides = params["unionPoolerParams"]

  # Generate a sequence list and an associated labeled list (both containing a
  # set of sequences separated by None)
  start = time.time()
  print "\nGenerating sequences..."
  patternAlphabetSize = sequenceLength * numberOfSequences
  patternMachine = PatternMachine(patternDimensionality, patternCardinality,
                                  patternAlphabetSize)
  sequenceMachine = SequenceMachine(patternMachine)

  numbers = sequenceMachine.generateNumbers(numberOfSequences, sequenceLength)
  generatedSequences = sequenceMachine.generateFromNumbers(numbers)
  sequenceLabels = [str(numbers[i + i*sequenceLength: i + (i+1)*sequenceLength])
                    for i in xrange(numberOfSequences)]
  labeledSequences = []
  for label in sequenceLabels:
    for _ in xrange(sequenceLength):
      labeledSequences.append(label)
    labeledSequences.append(None)

  # Set up the Temporal Memory and Union Pooler network
  print "\nCreating network..."
  experiment = UnionTemporalPoolerExperiment(tmParamOverrides, upParamOverrides)

  # Train only the Temporal Memory on the generated sequences
  # if trainingPasses > 0:
  #
  #   print "\nTraining Temporal Memory..."
  #   if consoleVerbosity > 0:
  #     print "\nPass\tBursting Columns Mean\tStdDev\tMax"
  #
  #   for i in xrange(trainingPasses):
  #     experiment.runNetworkOnSequences(generatedSequences,
  #                                      labeledSequences,
  #                                      tmLearn=True,
  #                                      upLearn=None,
  #                                      verbosity=consoleVerbosity,
  #                                      progressInterval=_SHOW_PROGRESS_INTERVAL)
  #
  #     if consoleVerbosity > 0:
  #       stats = experiment.getBurstingColumnsStats()
  #       print "{0}\t{1}\t{2}\t{3}".format(i, stats[0], stats[1], stats[2])
  #
  #     # Reset the TM monitor mixin's records accrued during this training pass
  #     # experiment.tm.mmClearHistory()
  #
  #   print
  #   print MonitorMixinBase.mmPrettyPrintMetrics(
  #     experiment.tm.mmGetDefaultMetrics())
  #   print
  #
  #   if plotVerbosity >= 2:
  #     plotNetworkState(experiment, plotVerbosity, trainingPasses, phase="Training")
  #
  # experiment.tm.mmClearHistory()
  # experiment.up.mmClearHistory()


  print "\nRunning test phase..."

  inputSequences = generatedSequences
  inputCategories = labeledSequences
  tmLearn = True
  upLearn = False
  classifierLearn = False
  currentTime = time.time()

  experiment.tm.reset()
  experiment.up.reset()

  poolingActivationTrace = numpy.zeros((experiment.up._numColumns, 1))
  activeCellsTrace = numpy.zeros((experiment.up._numColumns, 1))
  activeSPTrace = numpy.zeros((experiment.up._numColumns, 1))

  for _ in xrange(trainingPasses):
    for i in xrange(len(inputSequences)):
      sensorPattern = inputSequences[i]
      inputCategory = inputCategories[i]
      if sensorPattern is None:
        pass
      else:
        experiment.tm.compute(sensorPattern,
                        learn=tmLearn,
                        sequenceLabel=inputCategory)

        if upLearn is not None:
          activeCells, predActiveCells, burstingCols, = experiment.getUnionTemporalPoolerInput()
          experiment.up.compute(activeCells,
                          predActiveCells,
                          learn=upLearn,
                          sequenceLabel=inputCategory)

          currentPoolingActivation = experiment.up._poolingActivation

          currentPoolingActivation = experiment.up._poolingActivation.reshape((experiment.up._numColumns, 1))
          poolingActivationTrace = numpy.concatenate((poolingActivationTrace, currentPoolingActivation), 1)

          currentUnionSDR = numpy.zeros((experiment.up._numColumns, 1))
          currentUnionSDR[experiment.up._unionSDR] = 1
          activeCellsTrace = numpy.concatenate((activeCellsTrace, currentUnionSDR), 1)

          currentSPSDR = numpy.zeros((experiment.up._numColumns, 1))
          currentSPSDR[experiment.up._activeCells] = 1
          activeSPTrace = numpy.concatenate((activeSPTrace, currentSPSDR), 1)

    print "\nPass\tBursting Columns Mean\tStdDev\tMax"
    stats = experiment.getBurstingColumnsStats()
    print "{0}\t{1}\t{2}\t{3}".format(0, stats[0], stats[1], stats[2])
    print
    print MonitorMixinBase.mmPrettyPrintMetrics(\
        experiment.tm.mmGetDefaultMetrics() + experiment.up.mmGetDefaultMetrics())
    print
    experiment.tm.mmClearHistory()


  # estimate fraction of shared bits across adjacent time point
  unionSDRshared = experiment.up._mmComputeUnionSDRdiff()

  bitLifeList = experiment.up._mmComputeBitLifeStats()
  bitLife = numpy.array(bitLifeList)


  # Plot SP outputs, UP persistence and UP outputs in testing phase
  def showSequenceStartLine(ax, trainingPasses, sequenceLength):
    for i in xrange(trainingPasses):
      ax.vlines(i*sequenceLength, 0, 100, linestyles='--')

  plt.figure()
  ncolShow = 100
  f, (ax1, ax2, ax3) = plt.subplots(nrows=1,ncols=3)
  ax1.imshow(activeSPTrace[1:ncolShow,:], cmap=cm.Greys,interpolation="nearest",aspect='auto')
  showSequenceStartLine(ax1, trainingPasses, sequenceLength)
  ax1.set_title('SP SDR')
  ax1.set_ylabel('Columns')
  ax2.imshow(poolingActivationTrace[1:100,:], cmap=cm.Greys, interpolation="nearest",aspect='auto')
  showSequenceStartLine(ax2, trainingPasses, sequenceLength)
  ax2.set_title('Persistence')
  ax3.imshow(activeCellsTrace[1:ncolShow,:], cmap=cm.Greys, interpolation="nearest",aspect='auto')
  showSequenceStartLine(ax3, trainingPasses, sequenceLength)
  plt.title('Union SDR')

  ax2.set_xlabel('Time (steps)')

  pp = PdfPages('results/UnionPoolingDuringTMlearning_Experiment2.pdf')
  pp.savefig()
  pp.close()

  f, (ax1, ax2, ax3) = plt.subplots(nrows=3,ncols=1)
  ax1.plot((sum(activeCellsTrace))/experiment.up._numColumns*100)
  ax1.set_ylabel('Union SDR size (%)')
  ax1.set_xlabel('Time (steps)')
  ax1.set_ylim(0,25)

  ax2.plot(unionSDRshared)
  ax2.set_ylabel('Shared Bits')
  ax2.set_xlabel('Time (steps)')

  ax3.hist(bitLife)
  ax3.set_xlabel('Life duration for each bit')
  pp = PdfPages('results/UnionSDRproperty_Experiment2.pdf')
  pp.savefig()
  pp.close()

if __name__ == "__main__":
  experiment2()
