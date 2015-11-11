from optparse import OptionParser
import os
import sys
import time
import yaml
import cPickle
import copy

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm



from nupic.data.generators.pattern_machine import PatternMachine
from nupic.data.generators.sequence_machine import SequenceMachine
from nupic.research.monitor_mixin.monitor_mixin_base import MonitorMixinBase

from htmresearch.frameworks.union_temporal_pooling.spatiotemporal_pooling_experiment import (
    SpatiotemporalPoolerExperiment)

import common

from htmresearch.algorithms.spatiotemporal_pooling_experiment import SpatiotemporalPooler

def generateSequences(patternDimensionality, patternCardinality, sequenceLength,
                      sequenceCount):
  # Generate a sequence list and an associated labeled list
  # (both containing a set of sequences separated by None)
  print "Generating sequences..."
  patternAlphabetSize = sequenceLength * sequenceCount
  patternMachine = PatternMachine(patternDimensionality, patternCardinality,
                                  patternAlphabetSize)
  sequenceMachine = SequenceMachine(patternMachine)
  numbers = sequenceMachine.generateNumbers(sequenceCount, sequenceLength)
  generatedSequences = sequenceMachine.generateFromNumbers(numbers)
  sequenceLabels = [
    str(numbers[i + i * sequenceLength: i + (i + 1) * sequenceLength])
    for i in xrange(sequenceCount)]
  labeledSequences = []
  for label in sequenceLabels:
    for _ in xrange(sequenceLength):
      labeledSequences.append(label)
    labeledSequences.append(None)

  return generatedSequences, labeledSequences



sequenceLength = 50

def main():
  inputSequences, inputCategories = generateSequences(1024,20,sequenceLength,1)
  
  print inputCategories
    
  stpe = SpatiotemporalPoolerExperiment()
  
  columnActivations, unionedInput = stpe.runNetworkOnSequences(inputSequences, inputCategories)
  
  aspect = columnActivations.shape[0] / float(columnActivations.shape[1])
  
  # create raw input raster
  rawInput = np.zeros((len(inputSequences), stpe.stp._numInputs))  
  for i in xrange(len(inputSequences)):
    if inputSequences[i] is not None:
      rawInput[i][list(inputSequences[i])] = 1
  
  plt.figure()
  plt.subplot(121)
  plt.imshow(1-rawInput.T, aspect=aspect, interpolation="none", cmap = cm.Greys_r)
  plt.title('Original Input')
  plt.xlabel('Time step')
  plt.ylabel('Input')
  
  plt.subplot(122)
  plt.imshow(1-unionedInput.T, aspect=aspect, interpolation="none", cmap = cm.Greys_r)
  plt.title('Unioned Input')
  plt.xlabel('Time step')
  # plt.ylabel('Input')
  plt.savefig("results/input.png")
  
  # Collect input stats
  # turn unioned into set
  unionedSetList = []
  for row in unionedInput:
    unionedSetList.append(set(row.nonzero()[0]))

  (similarityMatrixBefore, similarityMatrixAfter, similarityMatrixBeforeAfter) = \
    common.calculateSimilarityMatrix(inputSequences, unionedSetList, sequenceLength)

  f, (ax1, ax2) = plt.subplots(nrows=1,ncols=2)
  im = ax1.imshow(similarityMatrixBefore[0:sequenceLength, 0:sequenceLength],interpolation="nearest")
  ax1.set_xlabel('Time (steps)')
  ax1.set_ylabel('Time (steps)')
  ax1.set_title('Overlap - Raw Input')
  
  im = ax2.imshow(similarityMatrixAfter[0:sequenceLength, 0:sequenceLength],interpolation="nearest")
  ax2.set_xlabel('Time (steps)')
  ax2.set_ylabel('Time (steps)')
  ax2.set_title('Overlap - Unioned Input')
  
  f.savefig('results/UnionSDRoverlapInput.png')
    
  preColumnActivations = columnActivations.copy()

  bitLifeListPretraining = np.array(stpe.stp._mmComputeBitLifeStats())
  
  data = {"before": dict(), "after": dict()}
    
  data["before"]["activeDutyCycle"] = np.zeros(stpe.stp.getColumnDimensions(), dtype=np.float32)
  stpe.stp.getActiveDutyCycles(data["before"]["activeDutyCycle"])
  
  data["before"]["overlapDutyCycle"] = np.zeros(stpe.stp.getColumnDimensions(), dtype=np.float32)
  stpe.stp.getActiveDutyCycles(data["before"]["overlapDutyCycle"])
  
  stpBefore = copy.deepcopy(stpe.stp)
  
  stpe.stp.mmClearHistory()
  
  # Perform iterations with learning on
  totalIterations = 500
  import time
  startTime = time.time()
  for i in xrange(totalIterations):
    _, _ = stpe.runNetworkOnSequences(inputSequences, inputCategories, stpLearn=True)
    if i % 5 == 0 and i != 0:
      now = time.time()
      elapsed = (now - startTime) / 60.0
      remaining = ((totalIterations - i) / (i / elapsed))
      print "%d / %d: elapsed %.1f min, remaining %.1f min" % (i, totalIterations, elapsed, remaining)
    
  
  data["after"]["activeDutyCycle"] = np.zeros(stpe.stp.getColumnDimensions(), dtype=np.float32)
  stpe.stp.getActiveDutyCycles(data["before"]["activeDutyCycle"])
  
  data["after"]["overlapDutyCycle"] = np.zeros(stpe.stp.getColumnDimensions(), dtype=np.float32)
  stpe.stp.getActiveDutyCycles(data["before"]["overlapDutyCycle"])

  stpDuring = copy.deepcopy(stpe.stp)
  
  stpe.stp.mmClearHistory()
  
  # Run one final test run after training to collect stats
  columnActivations, unionedInput = stpe.runNetworkOnSequences(inputSequences, inputCategories)  
  
  stpAfter = copy.deepcopy(stpe.stp)
  
  # Create heatmap
  common.plotSummaryResults(
    stpBefore,
    stpDuring,
    stpAfter,
    totalIterations,
    sequenceLength=sequenceLength,
    numColumns=stpe.stp._numColumns)
  
  
  bitLifeListPosttraining = np.array(stpe.stp._mmComputeBitLifeStats())
  
  # Column raster
  plt.figure()
  plt.subplot(121)
  plt.imshow(1-preColumnActivations.T, aspect=aspect, interpolation="none", cmap = cm.Greys_r)
  plt.title('Pre-training')
  plt.xlabel('Time step')
  plt.ylabel('SP Column')

  plt.subplot(122)
  plt.imshow(1-columnActivations.T, aspect=aspect, interpolation="none", cmap = cm.Greys_r)
  plt.title('Post-training')
  plt.xlabel('Time step')
  plt.savefig("results/columns.png")
  
  
  # Creat bit life stats plot
  preMean = bitLifeListPretraining.mean() 
  postMean = bitLifeListPosttraining.mean() 
  
  preStd = bitLifeListPretraining.std() 
  postStd = bitLifeListPosttraining.std() 
  
  
  # Bitlife histogram
  bins = np.linspace(0, 30, 30)
  plt.figure()
  plt.subplot(121)
  plt.hist(bitLifeListPretraining, bins)
  ylim = plt.ylim()
  plt.xlim((0,30))
  plt.ylabel("Cell count")
  plt.xlabel("Time steps")
  plt.title("Pre-training")
  plt.annotate("Mean: %.2f\nStd: %.2f" % (preMean, preStd), xy=(.6,.9), xycoords="axes fraction")
  
  plt.subplot(122)
  plt.hist(bitLifeListPosttraining, bins)
  plt.ylim(ylim)
  plt.xlim((0,30))
  plt.title("Post-training")
  plt.xlabel("Time steps")
  plt.annotate("Mean: %.2f\nStd: %.2f" % (postMean, postStd), xy=(.6,.9), xycoords="axes fraction")
   
  plt.savefig("results/bitlife.png")
  
  
  
  
  # stpe.stp.mmGetCellActivityPlot()
  # plt.savefig("results/test_plot.png")
  
  
  
  import time
  # time.sleep(5)
  
  cPickle.dump(data, open('results/data.pkl', 'wb'))
  
  print "DONE"
  


if __name__ == "__main__":
  main()