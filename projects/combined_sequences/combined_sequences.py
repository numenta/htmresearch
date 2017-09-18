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
from multiprocessing import Pool, cpu_count
import argparse
from argparse import RawDescriptionHelpFormatter
import os
import pprint
import random
import time

import numpy

from htmresearch.frameworks.layers.combined_sequence_experiment import (
  L4TMExperiment
)
from htmresearch.frameworks.layers.object_machine_factory import (
  createObjectMachine
)


def printDiagnostics(exp, sequences, objects, verbosity=0):
  """Useful diagnostics for debugging."""
  r = sequences.objectConfusion()
  print "Average common pairs in sequences=", r[0],
  print ", features=",r[2]

  r = objects.objectConfusion()
  print "Average common pairs in objects=", r[0],
  print ", locations=",r[1],
  print ", features=",r[2]

  print "Total number of objects created:",len(objects.getObjects())
  print "Total number of sequences created:",len(sequences.getObjects())

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


def trainSequences(sequences, exp):
  """Train the network on all the sequences"""
  for seqName in sequences:
    # Make sure we learn enough times to deal with high order sequences and
    # remove extra predictions.
    iterations = 3*len(sequences[seqName])
    for p in range(iterations):

      # Ensure we generate new random location for each sequence presentation
      objectSDRs = sequences.provideObjectsToLearn([seqName])
      exp.learnObjects(objectSDRs, reset=False)

      # TM needs reset between sequences, but not other regions
      exp.TMColumns[0].reset()

    # L2 needs resets when we switch to new object
    exp.sendReset()


def trainObjects(objects, exp, numRepeatsPerObject, experimentIdOffset):
  """
  Train the network on all the objects by randomly traversing points on
  each object.  We offset the id of each object to avoid confusion with
  any sequences that might have been learned.
  """
  # We want to traverse the features of each object randomly a few times before
  # moving on to the next object. Create the SDRs that we need for this.
  objectsToLearn = objects.provideObjectsToLearn()
  objectTraversals = {}
  for objectId in objectsToLearn:
    objectTraversals[objectId + experimentIdOffset] = objects.randomTraversal(
      objectsToLearn[objectId], numRepeatsPerObject)

  # Train the network on all the SDRs for all the objects
  exp.learnObjects(objectTraversals)


def inferSequence(exp, sequenceId, sequences):
  """Run inference on the given sequence."""
  # Create the (loc, feat) pairs for this sequence for column 0.
  objectSensations = {
    0: [pair for pair in sequences[sequenceId]]
  }

  inferConfig = {
    "object": sequenceId,
    "numSteps": len(objectSensations[0]),
    "pairs": objectSensations,
  }

  inferenceSDRs = sequences.provideObjectToInfer(inferConfig)
  exp.infer(inferenceSDRs, objectName=sequenceId)


def inferObject(exp, objectId, objects, objectName):
  """
  Run inference on the given object.
  objectName is the name of this object in the experiment.
  """

  # Create sequence of random sensations for this object for one column. The
  # total number of sensations is equal to the number of points on the object.
  # No point should be visited more than once.
  objectSensations = {}
  objectSensations[0] = []
  obj = objects[objectId]
  objectCopy = [pair for pair in obj]
  random.shuffle(objectCopy)
  for pair in objectCopy:
    objectSensations[0].append(pair)

  inferConfig = {
    "numSteps": len(objectSensations[0]),
    "pairs": objectSensations,
    "includeRandomLocation": False,
  }

  inferenceSDRs = objects.provideObjectToInfer(inferConfig)

  exp.infer(inferenceSDRs, objectName=objectName)


def runExperiment(args):
  """
  Runs the experiment.  What did you think this does?

  args is a dict representing the various parameters. We do it this way to
  support multiprocessing. The function returns the args dict updated with a
  number of additional keys containing performance metrics.
  """
  numObjects = args.get("numObjects", 10)
  numSequences = args.get("numSequences", 10)
  numFeatures = args.get("numFeatures", 10)
  seqLength = args.get("seqLength", 10)
  numPoints = args.get("numPoints", 10)
  trialNum = args.get("trialNum", 42)
  inputSize = args.get("inputSize", 512)
  numLocations = args.get("numLocations", 100000)
  numInputBits = args.get("inputBits", 20)
  settlingTime = args.get("settlingTime", 3)
  figure6 = args.get("figure6", False)

  random.seed(trialNum)

  #####################################################
  #
  # Create the sequences and objects, and make sure they share the
  # same features and locations.

  sequences = createObjectMachine(
    machineType="sequence",
    numInputBits=numInputBits,
    sensorInputSize=inputSize,
    externalInputSize=1024,
    numCorticalColumns=1,
    numFeatures=numFeatures,
    numLocations=numLocations,
    seed=trialNum
  )
  sequences.createRandomSequences(numSequences, seqLength)

  objects = createObjectMachine(
    machineType="simple",
    numInputBits=numInputBits,
    sensorInputSize=inputSize,
    externalInputSize=1024,
    numCorticalColumns=1,
    numFeatures=numFeatures,
    numLocations=numLocations,
    seed=trialNum
  )

  # Make sure they share the same features and locations
  objects.locations = sequences.locations
  objects.features = sequences.features

  objects.createRandomObjects(numObjects, numPoints=numPoints,
                                    numLocations=numLocations,
                                    numFeatures=numFeatures)

  #####################################################
  #
  # Setup experiment and train the network
  name = "combined_sequences_S%03d_O%03d_F%03d_L%03d_T%03d" % (
    numSequences, numObjects, numFeatures, numLocations, trialNum
  )
  exp = L4TMExperiment(
    name=name,
    numCorticalColumns=1,
    inputSize=inputSize,
    numExternalInputBits=numInputBits,
    externalInputSize=1024,
    numInputBits=numInputBits,
    seed=trialNum,
    L4Overrides={"initialPermanence": 0.41,
                 "activationThreshold": 18,
                 "minThreshold": 18,
                 "basalPredictedSegmentDecrement": 0.0001},
  )

  printDiagnostics(exp, sequences, objects, verbosity=0)

  # Train the network on all the sequences and then all the objects.
  trainSequences(sequences, exp)
  trainObjects(objects, exp, settlingTime, numSequences)

  ##########################################################################
  #
  # Run inference

  print "Running inference"
  if not figure6:
    # By default run inference on every sequence and object.
    for seqId in sequences:
      inferSequence(exp, seqId, sequences)
    for objectId in objects:
      inferObject(exp, objectId, objects, objectId + numSequences)

  else:
    # For figure 6 we want to run sequences and objects in a specific order
    for trial,itemType in enumerate(["sequence", "object", "sequence", "object",
                                     "sequence", "sequence", "object",
                                     "sequence", ]):
      if itemType == "sequence":
        objectId = random.randint(0, numSequences-1)
        inferSequence(exp, objectId, sequences)

      else:
        objectId = random.randint(0, numObjects-1)
        inferObject(exp, objectId, objects, objectId+numSequences)


  ##########################################################################
  #
  # Compute a number of overall inference statistics
  convergencePoint, sensorimotorAccuracy = exp.averageConvergencePoint(
    "L2 Representation", 30, 40, 1)

  sequenceAccuracy = exp.averageSequenceAccuracy(15, 25)

  infStats = exp.getInferenceStats()
  predictedActive = numpy.zeros(len(infStats))
  predicted = numpy.zeros(len(infStats))
  predictedActiveL4 = numpy.zeros(len(infStats))
  predictedL4 = numpy.zeros(len(infStats))
  for i,stat in enumerate(infStats):
    predictedActive[i] = float(sum(stat["TM PredictedActive C0"][2:])) / len(
      stat["TM PredictedActive C0"][2:])
    predicted[i] = float(sum(stat["TM NextPredicted C0"][2:])) / len(
      stat["TM NextPredicted C0"][2:])

    predictedActiveL4[i] = float(sum(stat["L4 PredictedActive C0"])) / len(
      stat["L4 PredictedActive C0"])
    predictedL4[i] = float(sum(stat["L4 Predicted C0"])) / len(
      stat["L4 Predicted C0"])

  print "# Sequences {} # features {} trial # {}\n".format(
    numSequences, numFeatures, trialNum)
  print "Sensorimotor accuracy:", sensorimotorAccuracy
  print "Sequence accuracy:", sequenceAccuracy


  # Return a bunch of metrics we will use in plots
  args.update({"name": exp.name})
  args.update({"objects": sequences.getObjects()})
  args.update({"convergencePoint":convergencePoint})
  args.update({"sensorimotorAccuracyPct": sensorimotorAccuracy})
  args.update({"sequenceAccuracyPct": sequenceAccuracy})
  args.update({"averagePredictions": predicted.mean()})
  args.update({"averagePredictedActive": predictedActive.mean()})
  args.update({"averagePredictionsL4": predictedL4.mean()})
  args.update({"averagePredictedActiveL4": predictedActiveL4.mean()})
  args.update({"statistics": exp.statistics})

  return args


def runExperimentPool(numSequences,
                      numFeatures,
                      numLocations,
                      numObjects,
                      numWorkers=7,
                      nTrials=1,
                      seqLength=10,
                      resultsName="convergence_results.pkl"):
  """
  Run a bunch of experiments using a pool of numWorkers multiple processes. For
  numSequences, numFeatures, and numLocations pass in a list containing valid
  values for that parameter. The cross product of everything is run, and each
  combination is run nTrials times.

  Returns a list of dict containing detailed results from each experiment. Also
  pickles and saves all the results in resultsName for later analysis.

  If numWorkers == 1, the experiments will be run in a single thread. This makes
  it easier to debug.

  Example:
    results = runExperimentPool(
                          numSequences=[10, 20],
                          numFeatures=[5, 13],
                          numWorkers=8,
                          nTrials=5)
  """
  # Create function arguments for every possibility
  args = []

  for o in reversed(numSequences):
    for l in numLocations:
      for f in numFeatures:
        for no in numObjects:
          for t in range(nTrials):
            args.append(
              {"numSequences": o,
               "numFeatures": f,
               "numObjects": no,
               "trialNum": t,
               "seqLength": seqLength,
               "numLocations": l,
               }
            )
  print "{} experiments to run, {} workers".format(len(args), numWorkers)

  # Run the pool
  if numWorkers > 1:
    pool = Pool(processes=numWorkers)
    result = pool.map(runExperiment, args)
  else:
    result = []
    for arg in args:
      result.append(runExperiment(arg))

  # Pickle results for later use
  with open(resultsName,"wb") as f:
    cPickle.dump(result,f)

  return result


def runExperiment4A(dirName):
  """
  This runs the first experiment in the section "Simulations with Pure
  Temporal Sequences".
  """
  # Results are put into a pkl file which can be used to generate the plots.
  # dirName is the absolute path where the pkl file will be placed.
  resultsFilename = os.path.join(dirName, "pure_sequences_example.pkl")

  results = runExperiment(
    {
      "numSequences": 5,
      "seqLength": 10,
      "numFeatures": 10,
      "trialNum": 0,
      "numObjects": 0,
      "numLocations": 100,
    }
  )

  # Pickle results for plotting and possible later debugging
  with open(resultsFilename, "wb") as f:
    cPickle.dump(results, f)


def runExperiment4B(dirName):
  """
  This runs the second experiment in the section "Simulations with Pure Temporal
  Sequences". Here we check accuracy of the L2/L4 networks in classifying the
  sequences. This experiment averages over many parameter combinations and could
  take several minutes.
  """
  # Results are put into a pkl file which can be used to generate the plots.
  # dirName is the absolute path where the pkl file will be placed.
  resultsName = os.path.join(dirName, "sequence_batch_results.pkl")

  numTrials = 10
  featureRange = [5, 10, 100]
  seqRange = [50]
  locationRange = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900,
                   1000, 1100, 1200, 1300, 1400, 1500, 1600]

  # Comment this out if you  are re-running analysis on already saved results.
  # Very useful for debugging the plots
  runExperimentPool(
    numSequences=seqRange,
    numFeatures=featureRange,
    numLocations=locationRange,
    numObjects=[0],
    seqLength=10,
    nTrials=numTrials,
    numWorkers=cpu_count()-1,
    resultsName=resultsName)


def runExperiment5A(dirName):
  """
  This runs the first experiment in the section "Simulations with Sensorimotor
  Sequences", an example sensorimotor sequence.
  """
  # Results are put into a pkl file which can be used to generate the plots.
  # dirName is the absolute path where the pkl file will be placed.
  resultsFilename = os.path.join(dirName, "sensorimotor_sequence_example.pkl")
  results = runExperiment(
    {
      "numSequences": 0,
      "seqLength": 10,
      "numFeatures": 100,
      "trialNum": 4,
      "numObjects": 50,
      "numLocations": 100,
    }
  )

  # Pickle results for plotting and possible later debugging
  with open(resultsFilename, "wb") as f:
    cPickle.dump(results, f)


def runExperiment5B(dirName):
  """
  This runs the second experiment in the section "Simulations with Sensorimotor
  Sequences". It averages over many parameter combinations. This experiment
  could take several hours.  You can run faster versions by reducing the number
  of trials.
  """
  # Results are put into a pkl file which can be used to generate the plots.
  # dirName is the absolute path where the pkl file will be placed.
  resultsName = os.path.join(dirName, "sensorimotor_batch_results.pkl")

  # We run 10 trials for each column number and then analyze results
  numTrials = 10
  featureRange = [5, 10, 50]
  objectRange = [2, 5, 10, 20, 30, 40, 50, 70]
  locationRange = [100]

  # Comment this out if you  are re-running analysis on already saved results.
  # Very useful for debugging the plots
  runExperimentPool(
    numSequences=[0],
    numObjects=objectRange,
    numFeatures=featureRange,
    numLocations=locationRange,
    nTrials=numTrials,
    numWorkers=cpu_count() - 1,
    resultsName=resultsName)


def runExperiment6(dirName):
  """
  This runs the experiment the section "Simulations with Combined Sequences",
  an example stream containing a mixture of temporal and sensorimotor sequences.
  """
  # Results are put into a pkl file which can be used to generate the plots.
  # dirName is the absolute path where the pkl file will be placed.
  resultsFilename = os.path.join(dirName, "combined_results.pkl")
  results = runExperiment(
    {
      "numSequences": 50,
      "seqLength": 10,
      "numObjects": 50,
      "numFeatures": 50,
      "trialNum": 8,
      "numLocations": 50,
      "settlingTime": 3,
      "figure6": True,
    }
  )

  # Pickle results for plotting and possible later debugging
  with open(resultsFilename, "wb") as f:
    cPickle.dump(results, f)


if __name__ == "__main__":

  # Map paper figures to experiment
  generateFigureFunc = {
    "4A": runExperiment4A,
    "4B": runExperiment4B,
    "5A": runExperiment5A,
    "5B": runExperiment5B,
    "6":  runExperiment6,
  }
  figures = generateFigureFunc.keys()
  figures.sort()

  parser = argparse.ArgumentParser(
    description="Use this script to generate the figures and results presented "
                "in (Ahmad & Hawkins, 2017)",
    formatter_class=RawDescriptionHelpFormatter,
    epilog="--------------------------------------------------------------\n"
           "  Subutai Ahmad & Jeff Hawkins (2017) \n"
           "  Untangling Sequences: Behavior vs. External Causes \n"
           "--------------------------------------------------------------\n")

  parser.add_argument(
    "figure",
    metavar="FIGURE",
    nargs='?',
    type=str,
    default=None,
    choices=figures,
    help=(
    "Specify the figure name to generate. Possible values are: %s " % figures)
  )
  parser.add_argument(
    "-l", "--list",
    action='store_true',
    help='List all figures'
  )
  opts = parser.parse_args()

  if opts.list:
    # Generate help by extracting the docstring from each function.
    for fig, func in sorted(generateFigureFunc.iteritems()):
      print fig, func.__doc__
  elif opts.figure is not None:
    startTime = time.time()
    generateFigureFunc[opts.figure](os.path.dirname(os.path.realpath(__file__)))
    print "Actual runtime=", time.time() - startTime
  else:
    parser.print_help()

