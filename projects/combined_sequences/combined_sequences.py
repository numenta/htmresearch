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
import traceback
import functools
import sys

import numpy

from htmresearch.frameworks.layers.combined_sequence_experiment import (
  L4TMExperiment
)
from htmresearch.frameworks.layers.object_machine_factory import (
  createObjectMachine
)


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

  print "Num L2 cells with connected synapses:", numConnectedCells
  if numConnectedCells > 0:
    print "Avg connected synapses per connected cell:", float(connectedSynapses)/numConnectedCells

  print


def trainSequences(sequences, exp, idOffset=0):
  """Train the network on all the sequences"""
  for seqId in sequences:
    # Make sure we learn enough times to deal with high order sequences and
    # remove extra predictions.
    iterations = 3*len(sequences[seqId])
    for p in range(iterations):

      # Ensure we generate new random location for each sequence presentation
      s = sequences.provideObjectsToLearn([seqId])
      objectSDRs = dict()
      objectSDRs[seqId + idOffset] = s[seqId]
      exp.learnObjects(objectSDRs, reset=False)

      # TM needs reset between sequences, but not other regions
      exp.TMColumns[0].reset()

    # L2 needs resets when we switch to new object
    exp.sendReset()


def trainObjects(objects, exp, numRepeatsPerObject, experimentIdOffset=0):
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


def inferSequence(exp, sequenceId, sequences, objectName=0):
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
  exp.infer(inferenceSDRs, objectName=objectName)


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


def createSuperimposedSDRs(inferenceSDRSequence, inferenceSDRObject):
  superimposedSDRs = []
  for sensations in zip(inferenceSDRSequence, inferenceSDRObject):
    print "sequence loc:", sensations[0][0][0]
    print "object loc:  ",sensations[1][0][0]
    print "superimposed:", sensations[0][0][0].union(sensations[1][0][0])
    print
    print "sequence feat:", sensations[0][0][1]
    print "object feat:  ",sensations[1][0][1]
    print "superimposed:", sensations[0][0][1].union(sensations[1][0][1])
    print
    newSensation = {
      0: (sensations[0][0][0].union(sensations[1][0][0]),
          sensations[0][0][1].union(sensations[1][0][1]))
    }
    print newSensation
    superimposedSDRs.append(newSensation)
    print
    print

  return superimposedSDRs


def createSuperimposedSensorySDRs(sequenceSensations, objectSensations):
  """
  Given two lists of sensations, create a new list where the sensory SDRs are
  union of the individual sensory SDRs. Keep the location SDRs from the object.

  A list of sensations has the following format:
  [
    {
      0: (set([1, 5, 10]), set([6, 12, 52]),  # location, feature for CC0
    },
    {
      0: (set([5, 46, 50]), set([8, 10, 11]),  # location, feature for CC0
    },
  ]

  We assume there is only one cortical column, and that the two input lists have
  identical length.

  """
  assert len(sequenceSensations) == len(objectSensations)
  superimposedSensations = []
  for i, objectSensation in enumerate(objectSensations):
    # print "sequence loc:", sequenceSensations[i][0][0]
    # print "object loc:  ",objectSensation[0][0]
    # print
    # print "sequence feat:", sequenceSensations[i][0][1]
    # print "object feat:  ",objectSensation[0][1]
    # print
    newSensation = {
      0: (objectSensation[0][0],
          sequenceSensations[i][0][1].union(objectSensation[0][1]))
    }
    # newSensation = {
    #   0: (objectSensation[0][0],objectSensation[0][1])
    # }
    superimposedSensations.append(newSensation)
    # print newSensation
    # print
    # print

  return superimposedSensations


def inferSuperimposedSequenceObjects(exp, sequenceId, objectId, sequences, objects):
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

  inferenceSDRSequence = sequences.provideObjectToInfer(inferConfig)

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

  inferenceSDRObject = objects.provideObjectToInfer(inferConfig)

  superimposedSDRs = createSuperimposedSDRs(inferenceSDRSequence, inferenceSDRObject)

  # exp.infer(superimposedSDRs, objectName=str(sequenceId) + "+" + str(objectId))
  exp.infer(superimposedSDRs, objectName=sequenceId*len(objects) + objectId)


def trainSuperimposedSequenceObjects(exp, numRepetitions,
                                     sequences, objects):
  """
  Train the network on the given object and sequence simultaneously for N
  repetitions each.  The total number of training inputs is N * seqLength = M
  (Ideally numPoints = seqLength)

  Create a list of M random training presentations of the object (To)
  Create a list of M=N*seqLength training presentations of the sequence (Ts).

  Create a list of M training inputs. The i'th training sensory input is created
  by taking the OR of To and Ts.  The location input comes from the object.

  We only train L4 this way (?).

  The other alternative is to train on one object for N repetitions. For each
  repetition we choose a random sequence and create OR'ed sensory inputs. Insert
  reset between each sequence presentation as before.  We need to ensure that
  each sequence is seen at least seqLength times.
  """
  trainingSensations = {}
  objectSDRs = objects.provideObjectsToLearn()
  sequenceSDRs = sequences.provideObjectsToLearn()

  # Create the order of sequences we will show. Each sequence will be shown
  # exactly numRepetitions times.
  sequenceOrder = range(len(sequences)) * numRepetitions
  random.shuffle(sequenceOrder)

  for objectId,sensations in objectSDRs.iteritems():

    # Create sequence of random sensations for this object, repeated
    # numRepetitions times. The total number of sensations is equal to the
    # number of points on the object multiplied by numRepetitions. Each time
    # an object is visited, we choose a random sequence to show.

    trainingSensations[objectId] = []

    for s in range(numRepetitions):
      # Get sensations for this object and shuffle them
      objectSensations = [sensation for sensation in sensations]
      random.shuffle(objectSensations)

      # Pick a random sequence and get its sensations.
      sequenceId = sequenceOrder.pop()
      sequenceSensations = sequenceSDRs[sequenceId]

      # Create superimposed sensory sensations.  We will only use the location
      # SDRs from the object SDRs.
      trainingSensations[objectId].extend(createSuperimposedSensorySDRs(
        sequenceSensations, objectSensations))

  # Train the network on all the SDRs for all the objects
  exp.learnObjects(trainingSensations)


def get_traceback(f):
  """
  Multiprocessing doesn't forward exception traceback information. This does.
  From: http://pragmaticpython.com/2017/02/19/
  """
  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    try:
      return f(*args, **kwargs)
    except Exception, ex:
      ret = '#' * 60
      ret += "\nException caught:"
      ret += "\n" + '-' * 60
      ret += "\n" + traceback.format_exc()
      ret += "\n" + '-' * 60
      ret += "\n" + "#" * 60
      print sys.stderr, ret
      sys.stderr.flush()
      raise ex

  return wrapper


@get_traceback
def runExperiment(args):
  """
  Runs the experiment.  The code is organized around what we need for specific
  figures in the paper.

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
  inputSize = args.get("inputSize", 1024)
  numLocations = args.get("numLocations", 100000)
  numInputBits = args.get("inputBits", 20)
  settlingTime = args.get("settlingTime", 1)
  numRepetitions = args.get("numRepetitions", 5)
  figure = args.get("figure", False)
  synPermProximalDecL2 = args.get("synPermProximalDecL2", 0.001)
  minThresholdProximalL2 = args.get("minThresholdProximalL2", 10)
  sampleSizeProximalL2 = args.get("sampleSizeProximalL2", 15)
  basalPredictedSegmentDecrement = args.get(
    "basalPredictedSegmentDecrement", 0.0006)
  stripStats = args.get("stripStats", True)


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
    L2Overrides={"synPermProximalDec": synPermProximalDecL2,
           "minThresholdProximal": minThresholdProximalL2,
           "sampleSizeProximal": sampleSizeProximalL2,
           "initialProximalPermanence": 0.45,
           "synPermProximalDec": 0.002,
    },
    TMOverrides={
      "basalPredictedSegmentDecrement": basalPredictedSegmentDecrement
    },
    L4Overrides={"initialPermanence": 0.21,
           "activationThreshold": 18,
           "minThreshold": 18,
           "basalPredictedSegmentDecrement": basalPredictedSegmentDecrement,
    },
  )

  printDiagnostics(exp, sequences, objects, args, verbosity=0)

  # Train the network on all the sequences and then all the objects.
  if figure in ["S", "6", "7"]:
    trainSuperimposedSequenceObjects(exp, numRepetitions, sequences, objects)
  else:
    trainObjects(objects, exp, numRepetitions)
    trainSequences(sequences, exp, numObjects)

  ##########################################################################
  #
  # Run inference

  print "Running inference"
  if figure in ["6"]:
    # We have trained the system on both temporal sequences and
    # objects. We test the system by randomly switching between sequences and
    # objects. To replicate the graph, we want to run sequences and objects in a
    # specific order
    for trial,itemType in enumerate(["sequence", "object", "sequence", "object",
                                     "sequence", "sequence", "object",
                                     "sequence", ]):
      if itemType == "sequence":
        objectId = random.randint(0, numSequences-1)
        inferSequence(exp, objectId, sequences, objectId+numObjects)

      else:
        objectId = random.randint(0, numObjects-1)
        inferObject(exp, objectId, objects, objectId)


  elif figure in ["7"]:
    # For figure 7 we have trained the system on both temporal sequences and
    # objects. We test the system by superimposing randomly chosen sequences and
    # objects.
    for trial in range(10):
      sequenceId = random.randint(0, numSequences - 1)
      objectId = random.randint(0, numObjects - 1)
      inferSuperimposedSequenceObjects(exp, sequenceId=sequenceId,
                         objectId=objectId, sequences=sequences, objects=objects)

  else:
    # By default run inference on every sequence and object in order.
    for objectId in objects:
      inferObject(exp, objectId, objects, objectId)
    for seqId in sequences:
      inferSequence(exp, seqId, sequences, seqId+numObjects)


  ##########################################################################
  #
  # Debugging diagnostics
  printDiagnosticsAfterTraining(exp)

  ##########################################################################
  #
  # Compute a number of overall inference statistics

  print "# Sequences {} # features {} trial # {}\n".format(
    numSequences, numFeatures, trialNum)

  convergencePoint, sequenceAccuracyL2 = exp.averageConvergencePoint(
    "L2 Representation", 30, 40, 1, numObjects)
  print "L2 accuracy for sequences:", sequenceAccuracyL2

  convergencePoint, objectAccuracyL2 = exp.averageConvergencePoint(
    "L2 Representation", 30, 40, 1, 0, numObjects)
  print "L2 accuracy for objects:", objectAccuracyL2

  objectCorrectSparsityTM, _ = exp.averageSequenceAccuracy(15, 25, 0, numObjects)
  print "TM accuracy for objects:", objectCorrectSparsityTM

  sequenceCorrectSparsityTM, sequenceCorrectClassificationsTM = \
    exp.averageSequenceAccuracy(15, 25, numObjects)
  print "TM accuracy for sequences:", sequenceCorrectClassificationsTM

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

  # Return a bunch of metrics we will use in plots
  args.update({"sequences": sequences.getObjects()})
  args.update({"objects": objects.getObjects()})
  args.update({"convergencePoint":convergencePoint})
  args.update({"objectAccuracyL2": objectAccuracyL2})
  args.update({"sequenceAccuracyL2": sequenceAccuracyL2})
  args.update({"sequenceCorrectSparsityTM": sequenceCorrectSparsityTM})
  args.update({"sequenceCorrectClassificationsTM": sequenceCorrectClassificationsTM})
  args.update({"objectCorrectSparsityTM": objectCorrectSparsityTM})
  args.update({"averagePredictions": predicted.mean()})
  args.update({"averagePredictedActive": predictedActive.mean()})
  args.update({"averagePredictionsL4": predictedL4.mean()})
  args.update({"averagePredictedActiveL4": predictedActiveL4.mean()})

  if stripStats:
    exp.stripStats()
  args.update({"name": exp.name})
  args.update({"statistics": exp.statistics})
  args.update({"networkConfig": exp.config})

  return args


def runExperimentPool(numSequences,
                      numFeatures,
                      numLocations,
                      numObjects,
                      numWorkers=7,
                      nTrials=1,
                      seqLength=10,
                      figure="",
                      numRepetitions=1,
                      synPermProximalDecL2=[0.001],
                      minThresholdProximalL2=[10],
                      sampleSizeProximalL2=[15],
                      inputSize=[1024],
                      basalPredictedSegmentDecrement=[0.0006],
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

  for bd in basalPredictedSegmentDecrement:
    for i in inputSize:
      for thresh in minThresholdProximalL2:
        for dec in synPermProximalDecL2:
          for s in sampleSizeProximalL2:
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
                         "sampleSizeProximalL2": s,
                         "synPermProximalDecL2": dec,
                         "minThresholdProximalL2": thresh,
                         "numRepetitions": numRepetitions,
                         "figure": figure,
                         "inputSize": i,
                         "basalPredictedSegmentDecrement": bd,
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

  # results = runExperiment(
  #   {
  #     "numSequences": 50,
  #     "seqLength": 10,
  #     "numFeatures": 100,
  #     "trialNum": 0,
  #     "numObjects": 0,
  #     "numLocations": 100,
  #     "numRepetitions": 10,
  #   }
  # )

  results = runExperiment(
    {
      "numSequences": 50,
      "seqLength": 10,
      "numFeatures": 100,
      "trialNum": 0,
      "numObjects": 0,
      "numLocations": 200,
      "numRepetitions": 30,
      "inputSize": 2048,
      "basalPredictedSegmentDecrement": 0.001,
      "stripStats": False,
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
  resultsName = os.path.join(dirName, "sequence_batch_high_dec_normal_features.pkl")

  numTrials = 10
  featureRange = [10, 50, 100, 200]
  seqRange = [50]
  locationRange = [10, 100, 200, 300, 400, 500]

  runExperimentPool(
    numSequences=seqRange,
    numFeatures=featureRange,
    numLocations=locationRange,
    numObjects=[0],
    seqLength=10,
    nTrials=numTrials,
    numWorkers=cpu_count()-1,
    basalPredictedSegmentDecrement=[0.005],
    resultsName=resultsName)


def runExperiment4C(dirName):
  """
  Similar to 4B but runs multiple sequences
  """
  # Results are put into a pkl file which can be used to generate the plots.
  # dirName is the absolute path where the pkl file will be placed.
  resultsName = os.path.join(dirName, "sequences_range_2048_mcs_500_locs.pkl")

  numTrials = 10
  featureRange = [50, 100, 200]
  seqRange = [10, 25, 50, 75, 100, 150]
  locationRange = [500]

  runExperimentPool(
    numSequences=seqRange,
    numFeatures=featureRange,
    numLocations=locationRange,
    numObjects=[0],
    seqLength=10,
    nTrials=numTrials,
    numWorkers=cpu_count()-1,
    # minThresholdProximalL2=[15],
    # sampleSizeProximalL2=[17],
    basalPredictedSegmentDecrement=[0.0006],
    inputSize=[2048],
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
  resultsName = os.path.join(dirName, "sensorimotor_batch_results_more_objects.pkl")

  # We run 10 trials for each column number and then analyze results
  numTrials = 10
  featureRange = [10, 50, 100, 150, 500]
  objectRange = [110, 130, 200, 300]
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
    numRepetitions=10,
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
      "numFeatures": 500,
      "trialNum": 8,
      "numLocations": 100,
      "settlingTime": 1,
      "figure": "6",
      "numRepetitions": 30,
      "basalPredictedSegmentDecrement": 0.001,
      "stripStats": False,
    }
  )

  # Pickle results for plotting and possible later debugging
  with open(resultsFilename, "wb") as f:
    cPickle.dump(results, f)


def runExperiment7(dirName):
  """
  This runs the experiment the section "Simulations with Combined Sequences",
  an example stream containing a mixture of temporal and sensorimotor sequences.
  For inference we superimpose the objects and sequences.
  """
  # Results are put into a pkl file which can be used to generate the plots.
  # dirName is the absolute path where the pkl file will be placed.
  resultsFilename = os.path.join(dirName, "superimposed_sequence_results.pkl")
  results = runExperiment(
    {
      "numSequences": 50,
      "seqLength": 10,
      "numObjects": 50,
      "numFeatures": 500,
      "trialNum": 8,
      "numLocations": 100,
      "settlingTime": 1,
      "figure": "7",
      "numRepetitions": 30,
      "basalPredictedSegmentDecrement": 0.001,
      "stripStats": False,
    }
  )

  # Pickle results for plotting and possible later debugging
  with open(resultsFilename, "wb") as f:
    cPickle.dump(results, f)


def runExperimentS(dirName):
  """
  This runs an experiment where the network is trained on stream containing a
  mixture of temporal and sensorimotor sequences.
  """
  # Results are put into a pkl file which can be used to generate the plots.
  # dirName is the absolute path where the pkl file will be placed.
  resultsFilename = os.path.join(dirName, "superimposed_training.pkl")
  results = runExperiment(
    {
      "numSequences": 50,
      "numObjects": 50,
      "seqLength": 10,
      "numFeatures": 100,
      "trialNum": 8,
      "numLocations": 100,
      "numRepetitions": 30,
      "sampleSizeProximalL2": 15,
      "minThresholdProximalL2": 10,
      "figure": "S",
      "stripStats": False,
    }
  )

  # Pickle results for plotting and possible later debugging
  with open(resultsFilename, "wb") as f:
    cPickle.dump(results, f)

  # Debugging
  with open(resultsFilename, "rb") as f:
    r = cPickle.load(f)

    r.pop("objects")
    r.pop("sequences")
    stat = r.pop("statistics")
    pprint.pprint(r)
    sObject = 0
    sSequence = 0
    for i in range(0, 50):
      sObject += sum(stat[i]['L4 PredictedActive C0'])
    for i in range(50, 100):
      sSequence += sum(stat[i]['L4 PredictedActive C0'])
    print sObject, sSequence



def runExperimentSP(dirName):
  """
  This runs a pool of experiments where the network is trained on stream
  containing a mixture of temporal and sensorimotor sequences.
  """
  # Results are put into a pkl file which can be used to generate the plots.
  # dirName is the absolute path where the pkl file will be placed.
  resultsFilename = os.path.join(dirName, "superimposed_128mcs.pkl")

  # We run a bunch of trials with these combinations
  numTrials = 10
  featureRange = [1000]
  objectRange = [50]

  # Comment this out if you  are re-running analysis on already saved results.
  runExperimentPool(
    numSequences=objectRange,
    numObjects=objectRange,
    numFeatures=featureRange,
    numLocations=[100],
    nTrials=numTrials,
    numWorkers=cpu_count() - 1,
    resultsName=resultsFilename,
    figure="S",
    numRepetitions=30,
    sampleSizeProximalL2=[15],
    minThresholdProximalL2=[10],
    synPermProximalDecL2=[0.001],
    # basalPredictedSegmentDecrement=[0.0, 0.001, 0.002, 0.003, 0.004, 0.005],
    basalPredictedSegmentDecrement=[0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12],
    inputSize=[128],
  )

  print "Done with experiments"

  # Debugging
  # with open(resultsFilename, "rb") as f:
  #   results = cPickle.load(f)
  #
  #   for i,r in enumerate(results):
  #     print "\nResult:",i
  #     r.pop("objects", None)
  #     r.pop("sequences", None)
  #     stat = r.pop("statistics")
  #     if ( (r["numFeatures"] == 500) and (r["sequenceAccuracyL2"] <= 0.2) and
  #          (r["objectAccuracyL2"] >= 0.9) ):
  #       pprint.pprint(r)
  #     sObject = 0
  #     sSequence = 0
  #     for i in range(0, 50):
  #       sObject += sum(stat[i]['L4 PredictedActive C0'])
  #     for i in range(50, 100):
  #       sSequence += sum(stat[i]['L4 PredictedActive C0'])
  #     print sObject, sSequence


if __name__ == "__main__":

  # Map paper figures to experiment
  generateFigureFunc = {
    "4A": runExperiment4A,
    "4B": runExperiment4B,
    "4C": runExperiment4C,
    "5A": runExperiment5A,
    "5B": runExperiment5B,
    "6":  runExperiment6,
    "7":  runExperiment7,
    "S":  runExperimentS,
    "SP": runExperimentSP,
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
  parser.add_argument(
    "notes",
    metavar="NOTES",
    nargs='?',
    type=str,
    default="",
    help=(
    "Notes")
  )
  opts = parser.parse_args()

  print opts.notes

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

