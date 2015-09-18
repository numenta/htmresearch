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
Runs k-fold cross validation experiment for classification of text samples.

EXAMPLE: from the nupic.fluent directory run...
  python experiments/baseline_experiment.py data/sample_reviews/sample_reviews.csv

Notes:
- The above example runs the ClassificationModelRandomSDR subclass of Model. To
  use a different model, use cmd line args modelName and modelModuleName.
- k-fold cross validation: the training dataset is split
  differently for each of the k trials. The majority of the dataset is used for
  training, and a small portion (1/k) is held out for evaluation; this
  evaluation data is different from the test data.
- classification and label are used interchangeably
"""


import argparse
import os
import pprint
import time

from fluent.experiments.runner import Runner
#from fluent.experiments.multi_runner import MultiRunner
from fluent.experiments.htm_runner import HTMRunner
from fluent.utils.data_split import KFolds



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

  if (not isinstance(args.kFolds, int)) or (args.kFolds < 1):
    raise ValueError("Invalid value for number of cross-validation folds.")

  root = os.path.dirname(os.path.realpath(__file__))
  resultsDir = os.path.join(root, args.resultsDir)

  if args.modelName == "HTMNetwork":
    runner = HTMRunner(dataPath=args.dataPath,
                       networkConfigPath=args.networkConfigPath,
                       resultsDir=resultsDir,
                       experimentName=args.experimentName,
                       loadPath=args.loadPath,
                       modelName=args.modelName,
                       numClasses=args.numClasses,
                       plots=args.plots,
                       orderedSplit=args.orderedSplit,
                       trainSizes=[],
                       verbosity=args.verbosity,
                       generateData=args.generateData,
                       votingMethod=args.votingMethod,
                       classificationFile=args.classificationFile,
                       classifierType=args.classifierType)
  else:
    runner = Runner(dataPath=args.dataPath,
                    resultsDir=resultsDir,
                    experimentName=args.experimentName,
                    loadPath=args.loadPath,
                    modelName=args.modelName,
                    numClasses=args.numClasses,
                    plots=args.plots,
                    orderedSplit=args.orderedSplit,
                    trainSizes=[],
                    verbosity=args.verbosity)

    # HTM network data isn't ready yet to initialize the model
    runner.initModel(args.modelName)

  print "Reading in data and preprocessing."
  dataTime = time.time()
  runner.setupData(args.textPreprocess)

  # TODO: move kfolds splitting to Runner
  random = False if args.orderedSplit else True
  runner.partitions = KFolds(args.kFolds).split(
    range(len(runner.samples)), randomize=random)
  runner.trainSizes = [len(x[0]) for x in runner.partitions]
  print ("Data setup complete; elapsed time is {0:.2f} seconds.\nNow encoding "
         "the data".format(time.time() - dataTime))

  encodeTime = time.time()
  runner.encodeSamples()
  print ("Encoding complete; elapsed time is {0:.2f} seconds.\nNow running the "
         "experiment.".format(time.time() - encodeTime))

  runner.runExperiment()
  print "Experiment complete in {0:.2f} seconds.".format(time.time() - start)

  resultCalcs = runner.calculateResults()
  _ = runner.evaluateCumulativeResults(resultCalcs)

  print "Saving..."
  runner.saveModel()

  if args.validation:
    print "Validating experiment against expected classifications..."
    print runner.validateExperiment(args.validation)

  ## TODO:
  # print "Calculating random classifier results for comparison."
  # print model.classifyRandomly(labels)


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("dataPath",
                      help="Path to data CSV.",
                      type=str)
  parser.add_argument("--networkConfigPath",
                      default=os.path.abspath("data/network_configs/sp_tm_knn.json"),
                      help="Path to JSON specifying the network params.",
                      type=str)
  parser.add_argument("-k", "--kFolds",
                      default=5,
                      type=int,
                      help="Number of folds for cross validation; k=1 will "
                      "run no cross-validation.")
  parser.add_argument("-e", "--experimentName",
                      default="k_folds",
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
                      default=0,
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
  parser.add_argument("--classifier",
                      default="KNN",
                      choices=["KNN", "CLA"],
                      help="Type of classifier to use for the HTM")
  parser.add_argument("-v", "--verbosity",
                      default=1,
                      type=int,
                      help="verbosity 0 will print out experiment steps, "
                           "verbosity 1 will include results, and verbosity > "
                           "1 will print out preprocessed tokens and kNN "
                           "inference metrics.")
  parser.add_argument("--contrCSV",
                      default="",
                      help="Path to contraction csv")
  parser.add_argument("--abbrCSV",
                      default="",
                      help="Path to abbreviation csv")
  parser.add_argument("--batch",
                      help="Train the model with all the data at one time",
                      action="store_true")
  parser.add_argument("--validation",
                      default="",
                      help="Path to file of expected classifications.")
  parser.add_argument("--skipConfirmation",
                      help="If specified will skip the user confirmation step",
                      default=False,
                      action="store_true")
  parser.add_argument("--generateData",
                      default=False,
                      action="store_true",
                      help="Whether or not to generate network data files. "
                           "This only applies to HTM models.")
  parser.add_argument("--votingMethod",
                      default="last",
                      choices=["last", "most"],
                      help="Method to use when picking final classifications. "
                           "This only applies to HTM models.")
  parser.add_argument("--classificationFile",
                      default="",
                      help="Json file mapping labels strings to their IDs. "
                           "This only applies to HTM models.")
  parser.add_argument("--classifierType",
                      default="KNN",
                      choices=["KNN", "CLA"],
                      help="Type of classifier to use for the HTM")

  args = parser.parse_args()

  if args.skipConfirmation or checkInputs(args):
    run(args)
