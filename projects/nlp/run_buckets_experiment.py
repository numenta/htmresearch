#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
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
Script to run "buckets" classification experiments.
"""

import argparse
import os
import pprint
import time

from htmresearch.frameworks.nlp.bucket_runner import BucketRunner
from htmresearch.frameworks.nlp.bucket_htm_runner import BucketHTMRunner

try:
  import simplejson as json
except ImportError:
  import json



def checkInputs(args):
  """Function that displays a set of arguments and asks to proceed."""
  pprint.pprint(vars(args))
  userIn = raw_input("Proceed? (y/n): ")

  if userIn == 'y':
    return True

  if userIn == 'n':
    return False

  print "Incorrect input given\n"
  return checkInputs(args)



def run(args):
  start = time.time()

  root = os.path.dirname(os.path.realpath(__file__))
  resultsDir = os.path.join(root, args.resultsDir)

  metricsDicts = []
  for trial in xrange(args.trials):
    print "Running buckets experiment trial {}...".format(trial)
    if args.modelName == "HTMNetwork":
      runner = BucketHTMRunner(dataPath=args.dataPath,
                               resultsDir=resultsDir,
                               experimentName=args.experimentName,
                               experimentType="buckets",
                               modelName=args.modelName,
                               retinaScaling=args.retinaScaling,
                               retina=args.retina,
                               apiKey=args.apiKey,
                               loadPath=args.loadPath,
                               plots=args.plots,
                               orderedSplit=args.orderedSplit,
                               verbosity=args.verbosity,
                               generateData=args.generateData,
                               classificationFile=args.classificationFile,
                               networkConfigPath=args.networkConfigPath,
                               trainingReps=args.trainingReps,
                               seed=args.seed,
                               concatenationMethod=args.combineMethod,
                               numClasses=0)
      runner.initModel(0)
    else:
      runner = BucketRunner(dataPath=args.dataPath,
                            resultsDir=resultsDir,
                            experimentName=args.experimentName,
                            experimentType="buckets",
                            modelName=args.modelName,
                            retinaScaling=args.retinaScaling,
                            retina=args.retina,
                            apiKey=args.apiKey,
                            loadPath=args.loadPath,
                            plots=args.plots,
                            orderedSplit=args.orderedSplit,
                            verbosity=args.verbosity,
                            concatenationMethod=args.combineMethod,
                            numClasses=1)
      runner.initModel(args.modelName)

    print "Reading in data, preprocessing, encoding, and bucketing."
    dataTime = time.time()

    runner.setupData(args.textPreprocess)

    runner.encodeSamples(args.writeEncodings)

    runner.bucketData()

    print ("Data setup complete; elapsed time is {0:.2f} seconds.\nNow running "
           "the buckets experiment.".format(time.time() - dataTime))

    runner.partitionIndices(args.seed, args.numInference)

    runner.train()

    metricsDicts.append(runner.runExperiment(args.numInference))

    print "Trial complete, now saving the model."
    runner.saveModel(trial)

    args.seed += 1

  metricsFilePath = os.path.join(resultsDir, args.experimentName, "metrics.json")
  print "Dumping results to {}".format(metricsFilePath)
  with open(metricsFilePath, "w") as f:
    json.dump(metricsDicts, f, sort_keys=True, indent=2, separators=(",", ": "))

  resultCalcs = runner.evaluateResults(
    metricsDicts, args.numInference, args.modelName)

  print "Experiment complete in {0:.2f} seconds.".format(time.time() - start)



if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument("dataPath",
                      help="Path to data CSV or folder of CSVs.",
                      type=str)
  parser.add_argument("-n", "--experimentName",
                      default="buckets",
                      type=str,
                      help="Experiment name.")
  parser.add_argument("-m", "--modelName",
                      default="CioWordFingerprint",
                      type=str,
                      help="Name of model class. Also used for model results "
                           "directory and pickle checkpoint.")
  parser.add_argument("--retinaScaling",
                      default=1.0,
                      type=float,
                      help="Factor by which to scale the Cortical.io retina.")
  parser.add_argument("--retina",
                      default="en_associative",
                      type=str,
                      help="Name of Cortical.io retina.")
  parser.add_argument("--apiKey",
                      default=None,
                      type=str,
                      help="Key for Cortical.io API.")
  parser.add_argument("--resultsDir",
                      default="results",
                      help="This will hold the experiment results.")
  parser.add_argument("--textPreprocess",
                      action="store_true",
                      default=False,
                      help="Whether or not to use text preprocessing.")
  parser.add_argument("--numInference",
                      help="Number of samples to use for inference per bucket. "
                           "A bucket with too few samples will be skipped.",
                      type=int,
                      default=10)
  parser.add_argument("--trials",
                      help="Number of experiment trials to run, where the "
                           "random selection of inference data samples will "
                           "change each trial.",
                      type=int,
                      default=1)
  parser.add_argument("--loadPath",
                      help="Path from which to load the serialized model.",
                      type=str,
                      default=None)
  parser.add_argument("--plots",
                      default=1,
                      type=int,
                      help="0 for no evaluation plots, 1 for classification "
                           "accuracy plots, 2 includes the confusion matrix.")
  parser.add_argument("--orderedSplit",
                      default=False,
                      action="store_true",
                      help="To split the data sets, False will split "
                           "the samples randomly, True will allocate the "
                           "first n samples to querying with the remainder "
                           "for ranking.")
  parser.add_argument("--seed",
                      default=42,
                      type=int,
                      help="Random seed, used in partitioning the data.")
  parser.add_argument("--writeEncodings",
                      default=False,
                      action="store_true",
                      help="Write encoded patterns to a JSON.")
  parser.add_argument("-v", "--verbosity",
                      default=1,
                      type=int,
                      help="verbosity 0 will print out experiment steps, "
                           "verbosity 1 will include results, and verbosity > "
                           "1 will print out preprocessed tokens and kNN "
                           "inference metrics.")
  parser.add_argument("--skipConfirmation",
                      help="If specified will skip the user confirmation step.",
                      default=False,
                      action="store_true")
  parser.add_argument("--networkConfigPath",
                      default="htm_network_config.json",
                      help="Path to JSON specifying the network params.",
                      type=str)
  parser.add_argument("--generateData",
                      default=False,
                      action="store_true",
                      help="Whether or not to generate network data files.")
  parser.add_argument("--classificationFile",
                      default="",
                      help="JSON file mapping string labels to ids.")
  parser.add_argument("--trainingReps",
                      help="How many times to repeat training an HTM network.",
                      type=int,
                      default=1)
  parser.add_argument("--combineMethod",
                      help="Method for combining KNN distances over multiple "
                           "inference steps. Acceptable values are 'min' and "
                           "'mean'.",
                      type=str,
                      default="min")

  args = parser.parse_args()

  if args.skipConfirmation or checkInputs(args):
    run(args)
