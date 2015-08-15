#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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

from classify_sensor_data import (configureNetwork, 
                                  runNetwork)
from settings import (NUM_RECORDS,
                      PARTITIONS,
                      SIGNAL_AMPLITUDE,
                      SIGNAL_MEAN,
                      SIGNAL_PERIOD,
                      WHITE_NOISE_AMPLITUDES)



class TestSensorDataClassification(unittest.TestCase):
  """
  Test classification results for sensor data. 
  """

  def testClassificationAccuracy(self):

    """
    Test classification accuracy for sensor data.
    """

    for noiseAmplitude in WHITE_NOISE_AMPLITUDES:
      expParams = ("RUNNING EXPERIMENT WITH PARAMS:\n"
                   " * numRecords=%s\n"
                   " * noiseAmplitude=%s\n"
                   " * signalAmplitude=%s\n"
                   " * signalMean=%s\n"
                   " * signalPeriod=%s\n"
                   ) % (NUM_RECORDS,
                        noiseAmplitude,
                        SIGNAL_AMPLITUDE,
                        SIGNAL_MEAN,
                        SIGNAL_PERIOD)
      print expParams

      network = configureNetwork(noiseAmplitude)
      numCorrect, numTestRecords, predictionAccuracy = runNetwork(network,
                                                                  NUM_RECORDS,
                                                                  PARTITIONS)

      if noiseAmplitude == 0:
        self.assertEqual(predictionAccuracy, 100)
      else:
        self.assertNotEqual(predictionAccuracy, 100)


if __name__ == "__main__":
  unittest.main()
