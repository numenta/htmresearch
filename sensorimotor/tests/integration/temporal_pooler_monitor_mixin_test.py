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
    nElements = 4
    nWorlds = 3
    n = 512
    w = 20
    universe = OneDUniverse(debugSensor=True, debugMotor=True,
                            nSensor=n, wSensor=w,
                            nMotor=n, wMotor=w)
    self.agents = [RandomOneDAgent(
                     OneDWorld(
                       universe,
                       range(nElements * world, nElements * (world+1))),
                     0,
                     possibleMotorValues=(-1,1),
                     seed=23) for world in xrange(nWorlds)]

    self.experimentRunner = SensorimotorExperimentRunner(
      tmOverrides={
        "columnDimensions": [n],
        "minThreshold": w*2,
        "maxNewSynapseCount": w*2,
        "activationThreshold": w*2,
        "seed": 42
      },
      tpOverrides={
        "columnDimensions": [n],
        "numActiveColumnsPerInhArea": w,
        "seed": 42
      }
    )


  def testGetConfusionMetrics(self):
    # Train TM
    sequences = self.experimentRunner.generateSequences(20, self.agents)
    self.experimentRunner.feedLayers(sequences, tmLearn=True, tpLearn=False,
                                     verbosity=self.VERBOSITY)

    self._printInformation()

    self.assertEqual(
      self.experimentRunner.tp.mmGetMetricStabilityConfusion().min, 0)
    self.assertEqual(
      self.experimentRunner.tp.mmGetMetricStabilityConfusion().max, 40)
    self.assertTrue(
      self.experimentRunner.tp.mmGetMetricStabilityConfusion().mean > 0)

    self.assertTrue(
      self.experimentRunner.tp.mmGetMetricDistinctnessConfusion().min < 20)
    self.assertTrue(
      self.experimentRunner.tp.mmGetMetricDistinctnessConfusion().max > 15)
    self.assertTrue(
      self.experimentRunner.tp.mmGetMetricDistinctnessConfusion().mean > 10)

    # Train TP
    sequences = self.experimentRunner.generateSequences(10, self.agents)
    self.experimentRunner.feedLayers(sequences, tmLearn=False, tpLearn=True,
                                     verbosity=self.VERBOSITY)

    self._printInformation()

    self.assertEqual(
      self.experimentRunner.tp.mmGetMetricStabilityConfusion().sum, 0)

    # Test
    sequences = self.experimentRunner.generateSequences(5,
                                                        self.agents,
                                                        numSequences=2)
    self.experimentRunner.feedLayers(sequences, tmLearn=False, tpLearn=False,
                                     verbosity=self.VERBOSITY)

    self._printInformation()

    self.assertTrue(
      self.experimentRunner.tp.mmGetMetricStabilityConfusion().max < 7)

    self.assertTrue(
      self.experimentRunner.tp.mmGetMetricDistinctnessConfusion().sum < 12)


  def _printInformation(self):
    if self.VERBOSITY > 1:
      print "Stability"
      print "============"
      print
      print self.experimentRunner.tp.mmPrettyPrintDataStabilityConfusion()
      print
      print "Distinctness"
      print "============"
      print
      print self.experimentRunner.tp.mmPrettyPrintDataDistinctnessConfusion()
      print
      print "Metrics"
      print "============"
      print
      print self.experimentRunner.tp.mmPrettyPrintMetrics(
        self.experimentRunner.tp.mmGetDefaultMetrics())


if __name__ == "__main__":
  unittest.main()
