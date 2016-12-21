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

import unittest
import numpy

from htmresearch.frameworks.layers import l2456_model


def _randomSDR():
  return list(numpy.random.permutation(2048)[0:40])


class L2456ModelTest(unittest.TestCase):
  """Tests for the L4L2Model class."""


  def setUp(self):
    "Setup consistent random numpy seed for each method."
    numpy.random.seed(42)


  def _getObjects(self):
    """The two objects we use for training/inference"""
    return {
      "simple1": [
        {
          # location, coarse feature, fine feature for column 0, sensation 1
          0: ([0, 1, 2], _randomSDR(), _randomSDR()),
          # location, coarse feature, fine feature for column 1, sensation 1
          1: ([0, 2, 2], _randomSDR(), _randomSDR()),
        },
        {
          # location, coarse feature, fine feature for column 0, sensation 2
          0: ([0, 2, 3], _randomSDR(), _randomSDR()),
          # location, coarse feature, fine feature for column 1, sensation 2
          1: ([1, 1, 2], _randomSDR(), _randomSDR()),
        },
      ],
      "simple2": [
        {
          # location, coarse feature, fine feature for column 0, sensation 1
          0: ([1, 2, 2], _randomSDR(), _randomSDR()),
          # location, coarse feature, fine feature for column 1, sensation 1
          1: ([2, 1, 2], _randomSDR(), _randomSDR()),
        },
        {
          # location, coarse feature, fine feature for column 0, sensation 2
          0: ([3, 1, 2], _randomSDR(), _randomSDR()),
          # location, coarse feature, fine feature for column 1, sensation 2
          1: ([3, 2, 2], _randomSDR(), _randomSDR()),
        },
      ]
    }


  def testModelCreation(self):
    """Simple test of the basic interface for L4L2Experiment. We set custom
    parameters to ensure they are passed through to the region."""
    # Set up experiment
    model = l2456_model.L2456Model(
      name="sample",
      numCorticalColumns=2,
      L4Overrides={"initialPermanence": 0.57},
      L6Overrides={"initialPermanence": 0.67},
      L2Overrides={"minThresholdProximal": 2},
      L5Overrides={"minThresholdProximal": 3},
    )

    # Ensure we have the right number of columns
    mainRegions = model.L4Columns + model.L2Columns + \
                  model.L5Columns + model.L6Columns
    self.assertEqual(len(mainRegions), 8, "Incorrect number of regions created")

    # Ensure each column got its parameter overrides
    for r in model.L4Columns:
      self.assertEqual(r.initialPermanence, 0.57, "L4 parameter override error")
    for r in model.L6Columns:
      self.assertEqual(r.initialPermanence, 0.67, "L6 parameter override error")
    for r in model.L2Columns:
      self.assertEqual(r.minThresholdProximal, 2, "L2 parameter override error")
    for r in model.L5Columns:
      self.assertEqual(r.minThresholdProximal, 3, "L5 parameter override error")


  def testModelLearning(self):
    """Simple test of the basic interface for L4L2Experiment. We set custom
    parameters to ensure they are passed through to the region."""

    # Set up experiment and train on two random objects
    model = l2456_model.L2456Model(
      name="sample",
      numCorticalColumns=2,
      numLearningPoints=4,
    )

    objects = self._getObjects()
    model.learnObjects(objects)

    # Check we have unique object representations in L2 and L5
    s1 = model.objectRepresentationsL2["simple1"]
    s2 = model.objectRepresentationsL2["simple2"]
    for i,sdr in enumerate(s1):
      self.assertLess(len(s1[i] & s2[i]), 5, "Non-unique L2 representations")

    s1 = model.objectRepresentationsL5["simple1"]
    s2 = model.objectRepresentationsL5["simple2"]
    for i, sdr in enumerate(s1):
      self.assertLess(len(s1[i] & s2[i]), 5, "Non-unique L5 representations")


    # Check that number of iterations is correct. It should be:
    # 2 objects * 2 sensations * 4 learningPoints + 2 resets = 18
    # We can check the timer to check this
    r = model.network.regions["locationInput_0"]
    t = r.computeTimer
    self.assertEqual(t.startCount, 18,
                     "Incorrect number of learning iterations")


  def testModelInference(self):
    """Simple test of the basic interface for L2456Experiment."""

    # Set up experiment and train on two random objects
    model = l2456_model.L2456Model(
      name="sample",
      numCorticalColumns=2,
    )

    objects = self._getObjects()
    model.learnObjects(objects)

    model.infer(objects["simple1"], objectName="simple1")

    # Check that L56 are learning
    self.assertEqual(
      sum(model.getInferenceStats(0)['L5 Representation C0']), 80,
      "L5 doesn't have correct object representation")
    self.assertEqual(
      sum(model.getInferenceStats(0)['Overlap L5 with object C0']), 80,
      "L5 doesn't have correct object representation")

    # Inference and learning currently do not fully work
    # TODO: Need to get L42 learning something useful!


if __name__ == "__main__":
  unittest.main()
