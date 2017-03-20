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
import unittest

from htmresearch.frameworks.layers import l2_l4_inference

from htmresearch.frameworks.layers.object_machine_factory import (
  createObjectMachine
)

import numpy


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
    )

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
                             [0, 0, 0, 0])
    self.assertSequenceEqual(stats[0]["Overlap L2 with object C1"],
                             [0, 0, 0, 0])

    self.assertSequenceEqual(exp.getL2Representations(),
                             [set(), set()])

    self.assertSequenceEqual(exp.getL4Representations(),
                             [set(xrange(0, 40)), set(xrange(40, 80))])


  def testDelayedLateralInputs(self):
    """Test whether lateral inputs are synchronized across columns"""
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

    # no lateral inputs on first iteration
    self.assertEqual(numpy.sum(numpy.abs(lateralInputs[0][0])), 0)
    self.assertEqual(numpy.sum(numpy.abs(lateralInputs[0][1])), 0)

    # lateral inputs of C0 at time t+1 = active cells of C1 at time t
    for step in range(2):
      self.assertEqual(
        numpy.sum(numpy.abs(lateralInputs[step+1][0]-activeCells[step][1])), 0)
      self.assertEqual(
        numpy.sum(numpy.abs(lateralInputs[step+1][1]-activeCells[step][0])), 0)


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
        "formInternalBasalConnections": True,
        "learningMode": True,
        "inferenceMode": True,
        "learnOnOneCell": False,
        "initialPermanence": 0.51,
        "connectedPermanence": 0.6,
        "permanenceIncrement": 0.1,
        "permanenceDecrement": 0.02,
        "minThreshold": 3,
        "predictedSegmentDecrement": 0.002,
        "activationThreshold": 3,
        "maxNewSynapseCount": 20,
        "implementation": "etm_cpp",
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



if __name__ == "__main__":
  unittest.main()
