#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import unittest

from sensorimotor.one_d_world import OneDWorld
from sensorimotor.one_d_universe import OneDUniverse
from sensorimotor.random_one_d_agent import RandomOneDAgent

from sensorimotor.sensorimotor_experiment_runner import (
  SensorimotorExperimentRunner
)



class TemporalPoolerMonitorMixinTest(unittest.TestCase):

  VERBOSITY = 1


  def setUp(self):
    # Initialize the universe, worlds, and agents
    nElements = 5
    wEncoders = 7
    universe = OneDUniverse(debugSensor=True,
                            debugMotor=True,
                            nSensor=nElements*wEncoders, wSensor=wEncoders,
                            nMotor=wEncoders*7, wMotor=wEncoders)
    self.agents = [
      RandomOneDAgent(OneDWorld(universe, range(nElements), 4),
                      possibleMotorValues=(-1,1), seed=23),
      RandomOneDAgent(OneDWorld(universe, list(reversed(range(nElements))), 4),
                      possibleMotorValues=(-1,1), seed=23)
    ]

    self.experimentRunner = SensorimotorExperimentRunner(
      tmOverrides={
        "columnDimensions": [nElements*wEncoders],
        "minThreshold": wEncoders*2,
        "maxNewSynapseCount": wEncoders*2,
        "activationThreshold": wEncoders*2
      },
      tpOverrides={
        "columnDimensions": [512],
        "numActiveColumnsPerInhArea": 20
      }
    )


  def testGetMatrixSequenceStableRepresentations(self):
    # Train
    sequences = self.experimentRunner.generateSequences(40, self.agents)
    self.experimentRunner.feedLayers(sequences, tmLearn=True, tpLearn=True,
                                     verbosity=self.VERBOSITY)

    if self.VERBOSITY > 1:
      print self.experimentRunner.tp.prettyPrintDataStabilityConfusion()
      print self.experimentRunner.tp.prettyPrintMetrics(
        self.experimentRunner.tp.getDefaultMetrics())

    self.assertEqual(
      self.experimentRunner.tp.getMetricStabilityConfusion().min, 0)
    self.assertEqual(
      self.experimentRunner.tp.getMetricStabilityConfusion().max, 40)
    self.assertTrue(
      self.experimentRunner.tp.getMetricStabilityConfusion().mean > 0)

    # Test
    sequences = self.experimentRunner.generateSequences(10, self.agents)
    self.experimentRunner.feedLayers(sequences, tmLearn=False, tpLearn=False,
                                     verbosity=self.VERBOSITY)

    if self.VERBOSITY > 1:
      print self.experimentRunner.tp.prettyPrintDataStabilityConfusion()
      print self.experimentRunner.tp.prettyPrintMetrics(
        self.experimentRunner.tp.getDefaultMetrics())

    self.assertEqual(
      self.experimentRunner.tp.getMetricStabilityConfusion().sum, 0)



if __name__ == "__main__":
  unittest.main()
