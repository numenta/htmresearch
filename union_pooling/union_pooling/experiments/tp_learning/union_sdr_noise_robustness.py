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
import sys
import time
import os
import yaml
from optparse import OptionParser

import numpy
from pylab import rcParams
import matplotlib.pyplot as plt

from nupic.data.generators.pattern_machine import PatternMachine
from nupic.data.generators.sequence_machine import SequenceMachine
from nupic.research.monitor_mixin.monitor_mixin_base import MonitorMixinBase

from experiments.capacity import data_utils
from union_pooling.experiments.union_pooler_experiment import (
    UnionPoolerExperiment)

_SHOW_PROGRESS_INTERVAL = 200

"""
Experiment 3
Test UP stability w.r.t spatial noise in the sequence

Compare two scenarios, with/without bias toward predicted cells
"""

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
inputSequences = sequenceMachine.generateFromNumbers(numbers)
sequenceLabels = [str(numbers[i + i*sequenceLength: i + (i+1)*sequenceLength])
                  for i in xrange(numberOfSequences)]
inputCategories = []
for label in sequenceLabels:
  for _ in xrange(sequenceLength):
    inputCategories.append(label)
  inputCategories.append(None)

# Set up the Temporal Memory and Union Pooler network
print "\nCreating network..."
experiment = UnionPoolerExperiment(tmParamOverrides, upParamOverrides)

# Train only the Temporal Memory on the generated sequences
if trainingPasses > 0:

  print "\nTraining Temporal Memory..."
  if consoleVerbosity > 0:
    print "\nPass\tBursting Columns Mean\tStdDev\tMax"

  for i in xrange(trainingPasses):
    experiment.runNetworkOnSequences(inputSequences,
                                     inputCategories,
                                     tmLearn=True,
                                     upLearn=None,
                                     verbosity=consoleVerbosity,
                                     progressInterval=_SHOW_PROGRESS_INTERVAL)

    if consoleVerbosity > 0:
      stats = experiment.getBurstingColumnsStats()
      print "{0}\t{1}\t{2}\t{3}".format(i, stats[0], stats[1], stats[2])

    # Reset the TM monitor mixin's records accrued during this training pass
    experiment.tm.mmClearHistory()

  print
  print MonitorMixinBase.mmPrettyPrintMetrics(
    experiment.tm.mmGetDefaultMetrics())
  print

experiment.tm.mmClearHistory()
experiment.up.mmClearHistory()

print "\nRunning test phase on noise-free sequence to establish a baseline..."

experiment.up._exciteFunction._minValue = 10
experiment.up._predictedActiveOverlapWeight = 20

plt.figure()

for predPriority in [True, False]:
  if predPriority:
    experiment.up._exciteFunction._maxValue = 100
  else:
    experiment.up._exciteFunction._maxValue = 10
  experiment.tm.reset()
  experiment.up.reset()
  experiment.tm.mmClearHistory()
  experiment.up.mmClearHistory()

  experiment.runNetworkOnSequences(inputSequences,
                                   inputCategories,
                                   tmLearn=False,
                                   upLearn=False,
                                   verbosity=consoleVerbosity,
                                   progressInterval=_SHOW_PROGRESS_INTERVAL)

  unionSDRbase = experiment.up._mmTraces['activeCells'].data[sequenceLength-1]

  nRepeats = 50
  # probability of turning an active bit into a different bit
  noiseLevelList = numpy.linspace(0, 0.1, 11)

  sharedBitsToBaseLineUSDR = numpy.zeros(noiseLevelList.shape)
  predictedInputsRatio = numpy.zeros(noiseLevelList.shape)
  for i in xrange(len(noiseLevelList)):
    noiseLevel = noiseLevelList[i]
    sharedBitsToBaseLine = numpy.zeros((nRepeats))
    burstingColsRatio = numpy.zeros((nRepeats))
    for rpt in xrange(nRepeats):
      noisySequence = sequenceMachine.addSpatialNoise(inputSequences, noiseLevel)
      experiment.tm.reset()
      experiment.up.reset()
      experiment.tm.mmClearHistory()
      experiment.up.mmClearHistory()

      experiment.runNetworkOnSequences(noisySequence,
                                     inputCategories,
                                     tmLearn=False,
                                     upLearn=False,
                                     verbosity=consoleVerbosity,
                                     progressInterval=_SHOW_PROGRESS_INTERVAL)

      unionSDR = experiment.up._mmTraces['activeCells'].data[sequenceLength-1]
      sharedBitsToBaseLine[rpt] = len(unionSDR & unionSDRbase) / float(len(unionSDRbase))

      defaultMetrics = experiment.tm.mmGetDefaultMetrics()
      burstingColsRatio = defaultMetrics[3].mean / defaultMetrics[0].mean

    sharedBitsToBaseLineUSDR[i] = numpy.mean(sharedBitsToBaseLine)
    predictedInputsRatio[i] = 1 - numpy.mean(burstingColsRatio)
    print noiseLevel, numpy.mean(sharedBitsToBaseLine), 1-numpy.mean(burstingColsRatio)

  plt.plot(noiseLevelList, sharedBitsToBaseLineUSDR)




tmLearn = False
upLearn = False
classifierLearn = False
currentTime = time.time()

noiseLevel = 0.01
noisySequence = sequenceMachine.addSpatialNoise(inputSequences, noiseLevel)

experiment.tm.reset()
experiment.up.reset()
experiment.tm.mmClearHistory()
experiment.up.mmClearHistory()

poolingActivationTrace = numpy.zeros((experiment.up._numColumns, 1))
activeCellsTrace = numpy.zeros((experiment.up._numColumns, 1))
activeSPTrace = numpy.zeros((experiment.up._numColumns, 1))


for i in xrange(len(noisySequence)):
  sensorPattern = noisySequence[i]
  inputCategory = inputCategories[i]
  if sensorPattern is None:
    pass
  else:
    experiment.tm.compute(sensorPattern,
                    formInternalConnections=True,
                    learn=tmLearn,
                    sequenceLabel=inputCategory)

    if upLearn is not None:
      activeCells, predActiveCells, burstingCols, = experiment.getUnionPoolerInput()
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
if trainingPasses > 0 and stats[0] > 0:
  print "***WARNING! MEAN BURSTING COLUMNS IN TEST PHASE IS GREATER THAN 0***"

print
print MonitorMixinBase.mmPrettyPrintMetrics(\
    experiment.tm.mmGetDefaultMetrics() + experiment.up.mmGetDefaultMetrics())
print

      
# estimate fraction of shared bits across adjacent time point      
unionSDRshared = experiment.up._mmComputeUnionSDRdiff()

bitLifeList = experiment.up._mmComputeBitLifeStats()
bitLife = numpy.array(bitLifeList)


# Plot SP outputs, UP persistence and UP outputs in testing phase
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
plt.ion()

ncolShow = 100
f, (ax1, ax2, ax3) = plt.subplots(nrows=1,ncols=3)
ax1.imshow(activeSPTrace[1:ncolShow,:], cmap=cm.Greys,interpolation="nearest",aspect='auto')
# ax1.set_xticklabels([])
ax1.set_title('SP SDR, noise: '+str(noiseLevel))
ax1.set_ylabel('Columns')
ax2.imshow(poolingActivationTrace[1:100,:], cmap=cm.Greys, interpolation="nearest",aspect='auto')
# ax2.set_yticklabels([])
ax2.set_title('Persistence')
ax3.imshow(activeCellsTrace[1:ncolShow,:], cmap=cm.Greys, interpolation="nearest",aspect='auto')
plt.title('Union SDR')

ax2.set_xlabel('Time (steps)')


# pp = PdfPages('results/UnionSDRexample.pdf')
# pp.savefig()
# pp.close()

plt.figure()
plt.subplot(2,2,1)
plt.plot(sum(activeCellsTrace))
plt.ylabel('Union SDR size')
plt.xlabel('Time (steps)')

plt.subplot(2,2,2)
plt.plot(unionSDRshared)
plt.ylabel('Shared Bits with previous union SDRs')
plt.xlabel('Time (steps)')
plt.subplot(2,2,3)
plt.hist(bitLife)
plt.xlabel('Life duration for each bit')
# pp = PdfPages('results/UnionSDRproperty.pdf')
# pp.savefig()
# pp.close()

