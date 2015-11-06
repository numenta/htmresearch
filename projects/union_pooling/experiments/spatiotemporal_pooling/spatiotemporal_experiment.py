from optparse import OptionParser
import os
import sys
import time
import yaml

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from nupic.data.generators.pattern_machine import PatternMachine
from nupic.data.generators.sequence_machine import SequenceMachine
from nupic.research.monitor_mixin.monitor_mixin_base import MonitorMixinBase

from htmresearch.frameworks.union_temporal_pooling.spatiotemporal_pooling_experiment import (
    SpatiotemporalPoolerExperiment)

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



  

def main():
  inputSequences, inputCategories = generateSequences(1024,10,125,2)
  
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
  
  
  plt.figure()
  plt.subplot(121)
  plt.imshow(1-columnActivations.T, aspect=aspect, interpolation="none", cmap = cm.Greys_r)
  plt.title('Pre-training')
  plt.xlabel('Time step')
  plt.ylabel('SP Column')
  # plt.savefig("results/pretain.png")
  
  # plt.draw()
  
  # return

  for _ in xrange(50):
    columnActivations, _ = stpe.runNetworkOnSequences(inputSequences, inputCategories, stpLearn=True)
  
  plt.subplot(122)
  plt.imshow(1-columnActivations.T, aspect=aspect, interpolation="none", cmap = cm.Greys_r)
  plt.title('Post-training')
  plt.xlabel('Time step')
  # plt.ylabel('SP Column')
  plt.savefig("results/columns.png")
  
  
  
  import time
  # time.sleep(5)
  
  print "DONE"
  


if __name__ == "__main__":
  main()