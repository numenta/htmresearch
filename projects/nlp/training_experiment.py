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
Script to run "incremental training" classification experiment.
"""

import argparse
import os
import pprint
import time

from htmresearch.frameworks.nlp.runner import Runner
from htmresearch.frameworks.nlp.multi_runner import MultiRunner
from htmresearch.frameworks.nlp.htm_runner import HTMRunner



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

  if os.path.isdir(args.dataPath):
    runner = MultiRunner(dataPath=args.dataPath,
                         resultsDir=resultsDir,
                         experimentName=args.experimentName,
                         loadPath=args.loadPath,
                         modelName=args.modelName,
                         numClasses=args.numClasses,
                         plots=args.plots,
                         orderedSplit=args.orderedSplit,
                         trainSizes=args.trainSizes,
                         verbosity=args.verbosity,
                         test=args.test)
  elif args.modelName == "HTMNetwork":
    runner = HTMRunner(dataPath=args.dataPath,
                       networkConfigPath=args.networkConfigPath,
                       resultsDir=resultsDir,
                       experimentName=args.experimentName,
                       loadPath=args.loadPath,
                       modelName=args.modelName,
                       numClasses=args.numClasses,
                       plots=args.plots,
                       orderedSplit=args.orderedSplit,
                       trainSizes=args.trainSizes,
                       verbosity=args.verbosity,
                       generateData=args.generateData,
                       votingMethod=args.votingMethod,
                       classificationFile=args.classificationFile)
  else:
    runner = Runner(dataPath=args.dataPath,
                    resultsDir=resultsDir,
                    experimentName=args.experimentName,
                    loadPath=args.loadPath,
                    modelName=args.modelName,
                    numClasses=args.numClasses,
                    plots=args.plots,
                    orderedSplit=args.orderedSplit,
                    trainSizes=args.trainSizes,
                    verbosity=args.verbosity)

  if args.modelName == "HTMNetwork":
    runner.initModel(0)
  else:
    runner.initModel(args.modelName)

  print "Reading in data and preprocessing."
  dataTime = time.time()
  runner.setupData(args.textPreprocess)
  print ("Data setup complete; elapsed time is {0:.2f} seconds.\nNow encoding "
         "the data".format(time.time() - dataTime))

  encodeTime = time.time()
  runner.encodeSamples()
  print ("Encoding complete; elapsed time is {0:.2f} seconds.\nNow running the "
         "experiment.".format(time.time() - encodeTime))

  runner.runExperiment()

  runner.writeOutClassifications()

  runner.calculateResults()

  print "Saving..."
  runner.saveModel()

  print "Experiment complete in {0:.2f} seconds.".format(time.time() - start)

  if args.validation:
    print "Validating experiment against expected classifications..."
    print runner.validateExperiment(args.validation)


if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument("dataPath",
                      help="Path to data CSV or folder of CSVs.",
                      type=str)
  parser.add_argument("--networkConfigPath",
                      default="htm_network_config.json",
                      help="Path to JSON specifying the network params.",
                      type=str)
  parser.add_argument("--test",
                      default=None,
                      help="Path to data CSV to use for testing if provided. "
                           "Otherwise will test on \'dataPath\'.")
  parser.add_argument("-e", "--experimentName",
                      default="incremental_training",
                      type=str,
                      help="Experiment name.")
  parser.add_argument("-m", "--modelName",
                      default="Keywords",
                      type=str,
                      help="Name of model class. Also used for model results "
                           "directory and pickle checkpoint.")
  parser.add_argument("--resultsDir",
                      default="results",
                      help="This will hold the experiment results.")
  parser.add_argument("--textPreprocess",
                      action="store_true",
                      default=False,
                      help="Whether or not to use text preprocessing.")
  parser.add_argument("--loadPath",
                      help="Path from which to load the serialized model.",
                      type=str,
                      default=None)
  parser.add_argument("--numClasses",
                      help="Specifies the number of classes per sample.",
                      type=int,
                      default=3)
  parser.add_argument("--plots",
                      default=1,
                      type=int,
                      help="0 for no evaluation plots, 1 for classification "
                           "accuracy plots, 2 includes the confusion matrix.")
  parser.add_argument("--orderedSplit",
                      default=False,
                      action="store_true",
                      help="To split the train and test sets, False will split "
                           "the samples randomly, True will allocate the "
                           "first n samples to training with the remainder "
                           "for testing.")
  parser.add_argument("--trainSizes",
                      default=[7, 7, 7, 13, 13, 13],
                      nargs="+",
                      type=int,
                      help="Number of samples to use in training. Separate "
                           "with spaces for multiple trials.")
  parser.add_argument("-v", "--verbosity",
                      default=1,
                      type=int,
                      help="verbosity 0 will print out experiment steps, "
                           "verbosity 1 will include results, and verbosity > "
                           "1 will print out preprocessed tokens and kNN "
                           "inference metrics.")
  parser.add_argument("--validation",
                      default="",
                      help="Path to file of expected classifications.")
  parser.add_argument("--skipConfirmation",
                      help="If specified will skip the user confirmation step.",
                      default=False,
                      action="store_true")
  parser.add_argument("--generateData",
                      default=False,
                      action="store_true",
                      help="Whether or not to generate network data files.")
  parser.add_argument("--votingMethod",
                      default="last",
                      choices=["last", "most"],
                      help="Method to use when picking final classifications.")
  parser.add_argument("--classificationFile",
                      default="",
                      help="JSON file mapping string labels to ids.")

  args = parser.parse_args()

  if args.skipConfirmation or checkInputs(args):
    run(args)
