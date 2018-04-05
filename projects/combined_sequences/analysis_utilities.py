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
This file runs a combined HTM network that includes the sensorimotor layers from
the Layers and Columns paper as well as a pure sequence layer.
"""

import cPickle
import os
import pprint
import random
import time
from collections import defaultdict
from texttable import Texttable
from copy import deepcopy

import numpy

def printDiagnostics(exp, sequences, objects, args, verbosity=0):
  """Useful diagnostics for debugging."""
  print "Experiment start time:", time.ctime()
  print "\nExperiment arguments:"
  pprint.pprint(args)

  r = sequences.objectConfusion()
  print "Average common pairs in sequences=", r[0],
  print ", features=",r[2]

  r = objects.objectConfusion()
  print "Average common pairs in objects=", r[0],
  print ", locations=",r[1],
  print ", features=",r[2]

  # For detailed debugging
  if verbosity > 0:
    print "\nObjects are:"
    for o in objects:
      pairs = objects[o]
      pairs.sort()
      print str(o) + ": " + str(pairs)
    print "\nSequences:"
    for i in sequences:
      print i,sequences[i]

  print "\nNetwork parameters:"
  pprint.pprint(exp.config)


def printDiagnosticsAfterTraining(exp, verbosity=0):
  """Useful diagnostics a trained system for debugging."""
  print "Number of connected synapses per cell"
  l2 = exp.getAlgorithmInstance("L2")

  numConnectedCells = 0
  connectedSynapses = 0
  for c in range(4096):
    cp = l2.numberOfConnectedProximalSynapses([c])
    if cp>0:
      # print c, ":", cp
      numConnectedCells += 1
      connectedSynapses += cp

  print "Num connected cells:", numConnectedCells
  print "Avg per connected cell:", float(connectedSynapses) / numConnectedCells

  print


def analyzeExperiment(dirName):
  """
  """
  # Results are put into a pkl file which can be used to generate the plots.
  # dirName is the absolute path where the pkl file will be placed.
  resultsFilename = os.path.join(dirName, "pure_sequences_example.pkl")

  with open(resultsFilename, "rb") as f:
    results = cPickle.load(f)

    for i,r in enumerate(results):
      print "\nResult:",i
      r.pop("objects", None)
      r.pop("sequences", None)
      stat = r.pop("statistics")
      if ( (r["numFeatures"] == 500) and (r["sequenceAccuracyL2"] <= 0.2) and
           (r["objectAccuracyL2"] >= 0.9) ):
        pprint.pprint(r)
      sObject = 0
      sSequence = 0
      for i in range(0, 50):
        sObject += sum(stat[i]['L4 PredictedActive C0'])
      for i in range(50, 100):
        sSequence += sum(stat[i]['L4 PredictedActive C0'])
      print sObject, sSequence


def stripPickleFile(dirName):
  """
  """
  # Results are put into a pkl file which can be used to generate the plots.
  # dirName is the absolute path where the pkl file will be placed.
  resultsFilename = os.path.join(dirName, "superimposed_pool_increments.pkl")

  with open(resultsFilename, "rb") as f:
    results = cPickle.load(f)

    for i,r in enumerate(results):
      print "\nResult:", i
      r.pop("objects", None)
      r.pop("sequences", None)
      r.pop("statistics")

  resultsFilename = os.path.join(dirName, "superimposed_pool_increments_stripped.pkl")
  with open(resultsFilename, "wb") as f:
    cPickle.dump(results, f)


def joinPickleFiles(dirName, basenames, outfileName):
  """
  """
  allResults = []
  for fn in basenames:
    resultsFile = os.path.join(dirName, fn)
    with open(resultsFile, "rb") as f:
      print "\n\nFile: ", fn
      results = cPickle.load(f)

      for i,r in enumerate(results):
        print "Result:", i
        r.pop("objects", None)
        r.pop("sequences", None)
        allResults.append(r)

  resultsFilename = os.path.join(dirName, outfileName)
  with open(resultsFilename, "wb") as f:
    cPickle.dump(allResults, f)


def createArgs(**kwargs):
  """
  Each kwarg is a list. Return a list of dicts representing all possible
  combinations of the kwargs.
  """
  if len(kwargs) == 0: return [{}]
  kargs = deepcopy(kwargs)
  k1 = kargs.keys()[0]
  values = kargs.pop(k1)
  args = []

  # Get all other combinations
  otherArgs = createArgs(**kargs)

  # Create combinations for values associated with k1
  for v in values:
    newArgs = deepcopy(otherArgs)
    arg = {k1: v}
    for newArg in newArgs:
      newArg.update(arg)
      args.append(newArg)

  return args


def summarizeExperiment(dirName):
  """
  """
  # Results are put into a pkl file which can be used to generate the plots.
  # dirName is the absolute path where the pkl file will be placed.
  table = Texttable()
  table.header(["Locations", "Features", "Objects", "Sequences",
                "objectAccuracyL2", "sequenceAccuracyL2",
                "objectCorrectSparsityTM", "sequenceCorrectClassificationsTM"])

  resultsFilename = os.path.join(dirName, "sensorimotor_batch_results_more_features.pkl")

  accuraciesL2 = defaultdict(list)
  accuraciesTM = defaultdict(list)
  sparsitiesTM = defaultdict(list)
  with open(resultsFilename, "rb") as f:
    results = cPickle.load(f)

    # Print first one
    r = results[0]
    r.pop("objects", None)
    r.pop("sequences", None)
    r.pop("statistics", None)
    pprint.pprint(r)

    for i,r in enumerate(results):
      table.add_row([r['numLocations'], r['numFeatures'],
                     r['numObjects'], r['numSequences'],
                     r["objectAccuracyL2"], r['sequenceAccuracyL2'],
                     r["objectCorrectSparsityTM"],
                     r["sequenceCorrectClassificationsTM"]])
      # print r["sequenceCorrectClassificationsTM"],r["sequenceCorrectSparsityTM"],r["objectAccuracyL2"],r["sequenceAccuracyL2"]
      # bpsd = r['basalPredictedSegmentDecrement']
      # accuraciesL2[bpsd].append(r["objectAccuracyL2"])
      # sparsitiesTM[bpsd].append(r["sequenceCorrectSparsityTM"])
      # accuraciesTM[bpsd].append(r["sequenceCorrectClassificationsTM"])
      # pprint.pprint(r)


  print table.draw() + "\n"

  # for metric in (accuraciesL2, accuraciesTM, sparsitiesTM):
  #   for k,v in metric.items():
  #     print k,v
  #     a = numpy.array(v)
  #     print "mean/stdev:", a.mean(), a.std()
  #     print
  #   print

if __name__ == "__main__":

  dirname = os.path.dirname(os.path.realpath(__file__))

  # numTrials = 10
  # featureRange = [50, 100]
  # seqRange = [50, 51]
  # locationRange = [10, 100]
  #
  # args = createArgs(
  #   numSequences=seqRange,
  #   numFeatures=featureRange,
  #   numLocations=locationRange,
  #   numObjects=[0],
  #   seqLength=[10],
  #   nTrials=[numTrials]
  # )
  # pprint.pprint(args)

  analyzeExperiment(dirname)
  # summarizeExperiment(dirname)
  # stripPickleFile(dirname)
  # joinPickleFiles(dirname,
  #                 [
  #                   "superimposed_1000f_1024mcs.pkl",
  #                   "superimposed_smaller_mcs.pkl",
  #                   "superimposed_128mcs.pkl",
  #                 ],
  #                 "superimposed_range_of_mcs.pkl"
  #                 )
