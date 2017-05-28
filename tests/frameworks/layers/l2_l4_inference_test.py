# ----------------------------------------------------------------------
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

"""Tests for l2_l4_inference module."""

import copy
from mock import patch
import unittest
import random

from htmresearch.frameworks.layers import l2_l4_inference

from htmresearch.frameworks.layers.object_machine_factory import (
  createObjectMachine
)

import numpy

def _randomSDR(numOfBits, size):
  """
  Creates a random SDR for the given cell count and SDR size
  :param numOfBits: Total number of bits
  :param size: Number of active bits desired in the SDR
  :return: list with active bits indexes in SDR
  """
  return random.sample(xrange(numOfBits), size)


class L4L2ExperimentTest(unittest.TestCase):
  """Tests for the L4L2Experiment class.

  The L4L2Experiment class doesn't have much logic in it. It sets up a network
  and the real work is all done inside the network. The tests here make sure
  that the interface works and has some basic sanity checks for the experiment
  statistics. These are intended to make sure that the code works but do not go
  far enough to validate that the experiments are set up correctly and getting
  meaningful experimental results.
  """


  def testSimpleExperiment(self):
    """Simple test of the basic interface for L4L2Experiment."""
    # Set up experiment
    exp = l2_l4_inference.L4L2Experiment(
      name="sample",
      numCorticalColumns=2,
      numInputBits=20,
      numExternalInputBits=20
    )

    # Set up feature and location SDRs for two locations, A and B, for each
    # cortical column, 0 and 1.
    locA0 = list(xrange(0, 20))
    featA0 = list(xrange(0, 20))
    locA1 = list(xrange(20, 40))
    featA1 = list(xrange(20, 40))

    locB0 = list(xrange(40, 60))
    featB0 = list(xrange(40, 60))
    locB1 = list(xrange(60, 80))
    featB1 = list(xrange(60, 80))

    # Learn each location for each column with several repetitions
    objectsToLearn = {"obj1": [
      {0: (locA0, featA0), 1: (locA1, featA1)},
      {0: (locB0, featB0), 1: (locB1, featB1)},
      {0: (locA0, featA0), 1: (locA1, featA1)},
      {0: (locB0, featB0), 1: (locB1, featB1)},
      {0: (locA0, featA0), 1: (locA1, featA1)},
      {0: (locB0, featB0), 1: (locB1, featB1)},
    ]}
    exp.learnObjects(objectsToLearn, reset=True)

    # Do the inference phase
    sensationsToInfer = [
      {0: (locA0, featA0), 1: (locA1, featA1)},
      {0: (locB0, featB0), 1: ([], [])},
      {0: ([], []), 1: (locA1, featA1)},
      {0: (locA0, featA0), 1: (locA1, featA1)},
    ]
    exp.infer(sensationsToInfer, objectName="obj1", reset=False)

    # Check the results
    stats = exp.getInferenceStats()
    self.assertEqual(len(stats), 1)
    self.assertEqual(stats[0]["numSteps"], 4)
    self.assertEqual(stats[0]["object"], "obj1")

    self.assertSequenceEqual(stats[0]["Overlap L2 with object C0"],
                             [40, 40, 40, 40])
    self.assertSequenceEqual(stats[0]["Overlap L2 with object C1"],
                             [40, 40, 40, 40])

    self.assertEqual(len(exp.getL2Representations()[0]),40)
    self.assertEqual(len(exp.getL2Representations()[1]),40)

    self.assertEqual(len(exp.getL4Representations()[0]),20)
    self.assertEqual(len(exp.getL4Representations()[1]),20)


  def testDelayedLateralandApicalInputs(self):
    """Test whether lateral and apical inputs are synchronized across columns"""
    # Set up experiment
    exp = l2_l4_inference.L4L2Experiment(
      name="sample",
      numCorticalColumns=2,
    )

    objects = createObjectMachine(
      machineType="simple",
      numInputBits=20,
      sensorInputSize=1024,
      externalInputSize=1024,
      numCorticalColumns=2,
    )
    objects.addObject([(1, 1), (2, 2)])
    objects.addObject([(1, 1), (3, 2)])

    exp.learnObjects(objects.provideObjectsToLearn())
    exp._sendReset()

    sensationC0 = [(1, 1), (1, 1), (1, 1)]
    sensationC1 = [(3, 2), (3, 2), (3, 2)]

    lateralInputs = []
    apicalInputs = []
    activeCells = []
    for step in range(3):
      inferConfig = {
        "noiseLevel": None,
        "numSteps": 1,
        "pairs": {0: [sensationC0[step]], 1: [sensationC1[step]]}
      }

      exp.infer(objects.provideObjectToInfer(inferConfig),
                objectName=1, reset=False)

      lateralInputs.append(
        {0: copy.copy(
          exp.network.regions['L2Column_0'].getInputData("lateralInput")),
         1: copy.copy(
           exp.network.regions['L2Column_1'].getInputData("lateralInput"))}
      )
      activeCells.append(
        {0: copy.copy(
          exp.network.regions['L2Column_0'].getOutputData("feedForwardOutput")),
         1: copy.copy(
           exp.network.regions['L2Column_1'].getOutputData("feedForwardOutput"))}
      )
      apicalInputs.append((
        {0: copy.copy(
          exp.network.regions['L4Column_0'].getInputData("apicalInput")),
          1: copy.copy(
            exp.network.regions['L4Column_1'].getInputData("apicalInput"))}
      ))

    # no lateral inputs on first iteration
    self.assertEqual(numpy.sum(numpy.abs(lateralInputs[0][0])), 0)
    self.assertEqual(numpy.sum(numpy.abs(lateralInputs[0][1])), 0)

    # no apical inputs to L4 on first iteration
    self.assertEqual(numpy.sum(numpy.abs(apicalInputs[0][0])), 0)
    self.assertEqual(numpy.sum(numpy.abs(apicalInputs[0][1])), 0)

    # lateral inputs of C0 at time t+1 = active cells of C1 at time t
    for step in range(2):
      self.assertEqual(
        numpy.sum(numpy.abs(lateralInputs[step+1][0]-activeCells[step][1])), 0)
      self.assertEqual(
        numpy.sum(numpy.abs(lateralInputs[step+1][1]-activeCells[step][0])), 0)

    # apical inputs of L4_0 at time t+1 = active cells of L2_0 at time t
    for step in range(2):
      self.assertEqual(
        numpy.sum(numpy.abs(apicalInputs[step + 1][0] - activeCells[step][0])),
        0)
      self.assertEqual(
        numpy.sum(numpy.abs(apicalInputs[step + 1][1] - activeCells[step][1])),
        0)


  def testCapacity(self):
    """This test mimmicks the capacity test parameters with smaller numbers.

    See `projects/l2_pooling/capacity_test.py`.
    """
    l2Params = {
        "inputWidth": 50 * 4,
        "cellCount": 100,
        "sdrSize": 10,
        "synPermProximalInc": 0.1,
        "synPermProximalDec": 0.001,
        "initialProximalPermanence": 0.6,
        "minThresholdProximal": 1,
        "sampleSizeProximal": 5,
        "connectedPermanenceProximal": 0.5,
        "synPermDistalInc": 0.1,
        "synPermDistalDec": 0.001,
        "initialDistalPermanence": 0.41,
        "activationThresholdDistal": 3,
        "sampleSizeDistal": 5,
        "connectedPermanenceDistal": 0.5,
        "distalSegmentInhibitionFactor": 1.5,
        "learningMode": True,
    }
    l4Params = {
        "columnCount": 50,
        "cellsPerColumn": 4,
        "learn": True,
        "learnOnOneCell": False,
        "initialPermanence": 0.51,
        "connectedPermanence": 0.6,
        "permanenceIncrement": 0.1,
        "permanenceDecrement": 0.02,
        "minThreshold": 3,
        "predictedSegmentDecrement": 0.002,
        "activationThreshold": 3,
        "sampleSize": 20,
        "implementation": "etm",
    }
    l4ColumnCount = 50
    numCorticalColumns=2
    exp = l2_l4_inference.L4L2Experiment(
        "testCapacity",
        numInputBits=100,
        L2Overrides=l2Params,
        L4Overrides=l4Params,
        inputSize=l4ColumnCount,
        externalInputSize=l4ColumnCount,
        numLearningPoints=4,
        numCorticalColumns=numCorticalColumns)

    # Set up feature and location SDRs for two locations, A and B, for each
    # cortical column, 0 and 1.
    locA0 = list(xrange(0, 5))
    featA0 = list(xrange(0, 5))
    locA1 = list(xrange(5, 10))
    featA1 = list(xrange(5, 10))

    locB0 = list(xrange(10, 15))
    featB0 = list(xrange(10, 15))
    locB1 = list(xrange(15, 20))
    featB1 = list(xrange(15, 20))

    # Learn each location for each column with several repetitions
    objectsToLearn = {"obj1": [
      {0: (locA0, featA0), 1: (locA1, featA1)},
      {0: (locB0, featB0), 1: (locB1, featB1)},
      {0: (locA0, featA0), 1: (locA1, featA1)},
      {0: (locB0, featB0), 1: (locB1, featB1)},
      {0: (locA0, featA0), 1: (locA1, featA1)},
      {0: (locB0, featB0), 1: (locB1, featB1)},
    ]}
    exp.learnObjects(objectsToLearn, reset=True)

    # Do the inference phase
    sensationsToInfer = [
      {0: (locA0, featA0), 1: (locA1, featA1)},
      {0: (locB0, featB0), 1: ([], [])},
      {0: ([], []), 1: (locA1, featA1)},
      {0: (locA0, featA0), 1: (locA1, featA1)},
    ]
    exp.infer(sensationsToInfer, objectName="obj1", reset=False)

    # Check the results
    stats = exp.getInferenceStats()
    self.assertEqual(len(stats), 1)
    self.assertEqual(stats[0]["numSteps"], 4)
    self.assertEqual(stats[0]["object"], "obj1")
    self.assertSequenceEqual(stats[0]["Overlap L2 with object C0"],
                             [10, 10, 10, 10])
    self.assertSequenceEqual(stats[0]["Overlap L2 with object C1"],
                             [10, 10, 10, 10])


  def testConsistency(self):
    """
    Test that L2 and L4 representations are consistent across different
    instantiations with the same seed.
    """
    random.seed(23)
    # Location and feature pool
    features = [_randomSDR(1024, 20) for _ in xrange(4)]
    locations = [_randomSDR(1024, 20) for _ in xrange(15)]

    # Learn 3 different objects (Can, Mug, Box)
    objectsToLearn = dict()

    # Soda can (cylinder) grasp from top.
    # Same feature at different locations, not all columns have input
    objectsToLearn["Can"] = [
      {
         0: (locations[0], features[0]),  # NW - round
         1: (locations[1], features[0]),  # N  - round
         2: (locations[2], features[0]),  # NE - round
         3: (locations[3], features[0]),  # SE - round
         4: ([], []),  # Little Finger has no input
      },
      {
         0: (locations[1], features[0]),  # N  - round
         1: (locations[2], features[0]),  # NE - round
         2: (locations[3], features[0]),  # SE - round
         3: (locations[4], features[0]),  # S  - round
         4: ([], []),  # Little Finger has no input
      },
      {
         0: (locations[2], features[0]),  # NE - round
         1: (locations[3], features[0]),  # SE - round
         2: (locations[4], features[0]),  # S  - round
         3: (locations[0], features[0]),  # NW - round
         4: ([], []),  # Little Finger has no input
      },
    ]

    # Coffee mug grasp from top
    # Same as cylinder with extra feature (handle)
    objectsToLearn["Mug"] = [
      {
         0: (locations[0], features[0]),  # NW - round
         1: (locations[1], features[1]),  # N  - handle
         2: (locations[2], features[0]),  # NE - round
         3: (locations[3], features[0]),  # SE - round
         4: (locations[4], features[0]),  # S  - round
      },
      {
        0: (locations[3], features[0]),  # SE - round
        1: (locations[0], features[0]),  # NW - round
        2: (locations[1], features[1]),  # N  - handle
        3: (locations[2], features[0]),  # NE - round
        4: (locations[4], features[0]),  # S  - round
      },
      {
        0: (locations[4], features[0]),  # S  - round
        1: (locations[3], features[0]),  # SE - round
        2: (locations[0], features[0]),  # NW - round
        3: (locations[1], features[1]),  # N  - handle
        4: (locations[2], features[0]),  # NE - round
      },
    ]

    # Box grasp from top
    # Symetrical features at different locations.
    objectsToLearn["Box"] = [
      {
         # Top/Front of the box
         0: (locations[5], features[2]),  # W1 - flat
         1: (locations[6], features[3]),  # N1 - corner
         2: (locations[7], features[3]),  # N2 - corner
         3: (locations[8], features[2]),  # E1 - flat
         4: (locations[9], features[2]),  # E2 - flat
      },
      {
         # Top/Side of the box
         0: (locations[5], features[2]),   # W1 - flat
         1: (locations[8], features[2]),   # E1 - flat
         2: (locations[9], features[2]),   # E2 - flat
         3: (locations[10], features[2]),  # E3 - flat
         4: (locations[11], features[2]),  # E4 - flat
      },
      {
         # Top/Back of the box
         0: (locations[9], features[2]),   # E2 - flat
         1: (locations[12], features[3]),  # S1 - corner
         2: (locations[13], features[3]),  # S2 - corner
         3: (locations[14], features[2]),  # W3 - flat
         4: (locations[5], features[2]),   # W1 - flat
      },
    ]

    # Create 10 experiment instances, train them on all objects and run
    # inference on one object
    numExps = 10
    exps = []
    for i in range(numExps):
      exps.append(
        l2_l4_inference.L4L2Experiment(
        "testClassification",
        numCorticalColumns=5,
        inputSize=1024,
        numInputBits=20,
        externalInputSize=1024,
        numExternalInputBits=20,
        numLearningPoints=3,
        seed=23,
        )
      )

      exps[i].learnObjects(objectsToLearn)

      # Try to infer "Mug" using first learned grasp
      sensations = [
          objectsToLearn["Mug"][0]
      ]
      exps[i].sendReset()
      exps[i].infer(sensations*2, reset=False)


    # Ensure L2 and L4 representations are consistent across all experiment
    # instantiations, across all columns, and across 2 different repeats
    for i in range(2):
      for c in range(5):
        L20 = set(exps[0].getL2Representations()[c])
        for e in range(1, numExps):
          self.assertSequenceEqual(L20, set(exps[e].getL2Representations()[c]))

        L40 = set(exps[0].getL4Representations()[c])
        for e in range(numExps):
          self.assertSequenceEqual(L40, set(exps[e].getL4Representations()[c]))


  def testObjectClassificationUnit(self):
    """
    Unit Test for multi column object classification
    """
    random.seed(36)
    objectL2SDR = {
      "Can": [set(_randomSDR(4096, 40)) for _ in xrange(5)],
      "Mug": [set(_randomSDR(4096, 40)) for _ in xrange(5)],
      "Box": [set(_randomSDR(4096, 40)) for _ in xrange(5)]
    }

    with patch.object(l2_l4_inference.L4L2Experiment, "getL2Representations")\
        as mock_getL2Representations:

      exp = l2_l4_inference.L4L2Experiment(
        "testClassificationUnit",
        numCorticalColumns=5
      )
      # Replace objects L2 representations to mock learning
      exp.objectL2Representations = objectL2SDR

      # test exact match
      mock_getL2Representations.return_value = objectL2SDR["Can"]
      results = exp.getCurrentClassification()
      self.assertDictEqual(results, {"Box": 0, "Mug": 0, "Can": 1})

      # test no match
      mock_getL2Representations.return_value = [
        set(_randomSDR(4096, 40)) for _ in xrange(5)]
      results = exp.getCurrentClassification()
      self.assertDictEqual(results, {"Box": 0, "Mug": 0, "Can": 0})

      # test no touch
      mock_getL2Representations.return_value = [(), (), (), (), ()]
      results = exp.getCurrentClassification()
      self.assertDictEqual(results, {"Box": 0, "Mug": 0, "Can": 0})

      # test partial match (Mug/Can)
      mock_getL2Representations.return_value = [
        objectL2SDR["Can"][0],
        objectL2SDR["Can"][1],
        objectL2SDR["Mug"][2],
        objectL2SDR["Mug"][3],
        ()
      ]
      results = exp.getCurrentClassification()
      self.assertDictEqual(results, {"Box": 0, "Mug": 0.5, "Can": 0.5})


  def testObjectClassification(self):
    """
    Test multi column object classification
    """
    random.seed(36)
    exp = l2_l4_inference.L4L2Experiment(
        "testClassification",
        numCorticalColumns=5,
        inputSize=1024,
        numInputBits=20,
        externalInputSize=1024,
        numExternalInputBits=20,
        numLearningPoints=3,
        )

    # Location and feature pool
    features = [_randomSDR(1024, 20) for _ in xrange(4)]
    locations = [_randomSDR(1024, 20) for _ in xrange(15)]

    # Learn 3 different objects (Can, Mug, Box)
    objectsToLearn = dict()

    # Soda can (cylinder) grasp from top.
    # Same feature at different locations, not all columns have input
    objectsToLearn["Can"] = [
      {
         0: (locations[0], features[0]),  # NW - round
         1: (locations[1], features[0]),  # N  - round
         2: (locations[2], features[0]),  # NE - round
         3: (locations[3], features[0]),  # SE - round
         4: ([], []),  # Little Finger has no input
      },
      {
         0: (locations[1], features[0]),  # N  - round
         1: (locations[2], features[0]),  # NE - round
         2: (locations[3], features[0]),  # SE - round
         3: (locations[4], features[0]),  # S  - round
         4: ([], []),  # Little Finger has no input
      },
      {
         0: (locations[2], features[0]),  # NE - round
         1: (locations[3], features[0]),  # SE - round
         2: (locations[4], features[0]),  # S  - round
         3: (locations[0], features[0]),  # NW - round
         4: ([], []),  # Little Finger has no input
      },
    ]

    # Coffee mug grasp from top
    # Same as cylinder with extra feature (handle)
    objectsToLearn["Mug"] = [
      {
         0: (locations[0], features[0]),  # NW - round
         1: (locations[1], features[1]),  # N  - handle
         2: (locations[2], features[0]),  # NE - round
         3: (locations[3], features[0]),  # SE - round
         4: (locations[4], features[0]),  # S  - round
      },
      {
        0: (locations[3], features[0]),  # SE - round
        1: (locations[0], features[0]),  # NW - round
        2: (locations[1], features[1]),  # N  - handle
        3: (locations[2], features[0]),  # NE - round
        4: (locations[4], features[0]),  # S  - round
      },
      {
        0: (locations[4], features[0]),  # S  - round
        1: (locations[3], features[0]),  # SE - round
        2: (locations[0], features[0]),  # NW - round
        3: (locations[1], features[1]),  # N  - handle
        4: (locations[2], features[0]),  # NE - round
      },
    ]

    # Box grasp from top
    # Symetrical features at different locations.
    objectsToLearn["Box"] = [
      {
         # Top/Front of the box
         0: (locations[5], features[2]),  # W1 - flat
         1: (locations[6], features[3]),  # N1 - corner
         2: (locations[7], features[3]),  # N2 - corner
         3: (locations[8], features[2]),  # E1 - flat
         4: (locations[9], features[2]),  # E2 - flat
      },
      {
         # Top/Side of the box
         0: (locations[5], features[2]),   # W1 - flat
         1: (locations[8], features[2]),   # E1 - flat
         2: (locations[9], features[2]),   # E2 - flat
         3: (locations[10], features[2]),  # E3 - flat
         4: (locations[11], features[2]),  # E4 - flat
      },
      {
         # Top/Back of the box
         0: (locations[9], features[2]),   # E2 - flat
         1: (locations[12], features[3]),  # S1 - corner
         2: (locations[13], features[3]),  # S2 - corner
         3: (locations[14], features[2]),  # W3 - flat
         4: (locations[5], features[2]),   # W1 - flat
      },
    ]
    exp.learnObjects(objectsToLearn)

    # Try to infer "Mug" using first learned grasp
    sensations = [
        objectsToLearn["Mug"][0],
        objectsToLearn["Mug"][1]
    ]
    exp.sendReset()
    exp.infer(sensations, reset=False)
    results = exp.getCurrentClassification()
    self.assertEquals(results["Mug"], 1)
    self.assertEquals(results["Box"], 0)
    self.assertEquals(results["Can"], 0)

    # Try to infer "Cylinder" using first learned grasp
    sensations = [
        objectsToLearn["Can"][0],
        objectsToLearn["Can"][1]
    ]
    exp.sendReset()
    exp.infer(sensations, reset=False)
    results = exp.getCurrentClassification()
    self.assertEquals(results["Mug"], 0)
    self.assertEquals(results["Box"], 0)
    self.assertEquals(results["Can"], 1)

    # Try to infer "Box" using first learned grasp
    sensations = [
        objectsToLearn["Box"][0],
        objectsToLearn["Box"][1]
    ]
    exp.sendReset()
    exp.infer(sensations, reset=False)
    results = exp.getCurrentClassification()
    self.assertEquals(results["Mug"], 0)
    self.assertEquals(results["Box"], 1)
    self.assertEquals(results["Can"], 0)

    # Try to infer half "Box" half "Mug" to confuse
    sensations = [
      {
        0: objectsToLearn["Box"][0][0],
        1: objectsToLearn["Box"][0][1],
        2: objectsToLearn["Mug"][0][0],
        3: objectsToLearn["Mug"][0][1],
        4: ([], []),
      },
      {
        0: objectsToLearn["Box"][0][0],
        1: objectsToLearn["Box"][0][1],
        2: objectsToLearn["Mug"][0][0],
        3: objectsToLearn["Mug"][0][1],
        4: ([], []),
      }
    ]
    exp.sendReset()
    exp.infer(sensations, reset=False)
    results = exp.getCurrentClassification()
    self.assertEquals(results["Mug"], 0.5)
    self.assertEquals(results["Box"], 0.5)
    self.assertEquals(results["Can"], 0)


  def testDefaultParameters(self):
    """
    Test multi column object classification
    """
    random.seed(36)
    exp = l2_l4_inference.L4L2Experiment(
      "testClassification",
      numCorticalColumns=1,
      inputSize=1024,
      numInputBits=20,
      externalInputSize=512,
      numExternalInputBits=10,
      numLearningPoints=3,
    )
    net = exp.network

    L4Params =  {
      "columnCount": 1024,
      "cellsPerColumn": 16,
      "learn": True,
      "learnOnOneCell": False,
      "initialPermanence": 0.51,
      "connectedPermanence": 0.6,
      "permanenceIncrement": 0.1,
      "permanenceDecrement": 0.02,
      "minThreshold": 8,
      "predictedSegmentDecrement": 0.0,
      "activationThreshold": 8,
      "sampleSize": 15,
    }

    L2Params =  {
      "inputWidth": 1024 * 16,
      "cellCount": 4096,
      "sdrSize": 40,
      "synPermProximalInc": 0.1,
      "synPermProximalDec": 0.001,
      "initialProximalPermanence": 0.6,
      "minThresholdProximal": 6,
      "sampleSizeProximal": 10,
      "connectedPermanenceProximal": 0.5,
      "synPermDistalInc": 0.1,
      "synPermDistalDec": 0.001,
      "initialDistalPermanence": 0.41,
      "activationThresholdDistal": 13,
      "sampleSizeDistal": 20,
      "connectedPermanenceDistal": 0.5,
      "distalSegmentInhibitionFactor": 1.001,
      "learningMode": True,
    }
    L4Column = net.regions["L4Column_0"].getSelf()
    L2Column = net.regions["L2Column_0"].getSelf()

    # check the default parameters are correct in L4
    for param, value in L4Params.iteritems():
      self.assertEqual(L4Column.getParameter(param), value)

    # check the default parameters are correct in L2
    for param, value in L2Params.iteritems():
      self.assertEqual(L2Column.getParameter(param), value)



if __name__ == "__main__":
  unittest.main()
