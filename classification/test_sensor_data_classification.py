#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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
import json
import os
import unittest

from classify_sensor_data import configureNetwork, runNetwork
from generate_sensor_data import generateData
from settings import (NUM_CATEGORIES,
                      SEQUENCE_LENGTH,
                      OUTFILE_NAME,
                      NUM_RECORDS,
                      NETWORK_CONFIGURATIONS,
                      SIGNAL_AMPLITUDE,
                      SIGNAL_MEAN,
                      SIGNAL_PERIOD,
                      WHITE_NOISE_AMPLITUDES,
                      DATA_DIR,
                      SP_REGION_NAME,
                      TM_REGION_NAME,
                      UP_REGION_NAME)



class TestSensorDataClassification(unittest.TestCase):
  """
  Test classification results for sensor data. 
  """


  def testClassificationAccuracy(self):

    """
    Test classification accuracy for sensor data.
    """

    for networkConfiguration in NETWORK_CONFIGURATIONS:
      for noiseAmplitude in WHITE_NOISE_AMPLITUDES:
        expParams = ("RUNNING EXPERIMENT WITH PARAMS:\n"
                     " * numRecords=%s\n"
                     " * signalAmplitude=%s\n"
                     " * signalMean=%s\n"
                     " * signalPeriod=%s\n"
                     " * noiseAmplitude=%s\n"
                     " * networkConfiguration=%s\n"
                     ) % (NUM_RECORDS,
                          SIGNAL_AMPLITUDE,
                          SIGNAL_MEAN,
                          SIGNAL_PERIOD,
                          noiseAmplitude,
                          json.dumps(networkConfiguration, indent=2))
        print expParams

        inputFile = generateData(DATA_DIR,
                                 OUTFILE_NAME,
                                 SIGNAL_MEAN,
                                 SIGNAL_PERIOD,
                                 SEQUENCE_LENGTH,
                                 NUM_RECORDS,
                                 SIGNAL_AMPLITUDE,
                                 NUM_CATEGORIES,
                                 noiseAmplitude)

        network = configureNetwork(inputFile,
                                   networkConfiguration)

        (numCorrect,
         numTestRecords,
         predictionAccuracy) = runNetwork(network,
                                          networkConfiguration,
                                          NUM_RECORDS)

        spEnabled = networkConfiguration[SP_REGION_NAME]["enabled"]
        tmEnabled = networkConfiguration[TM_REGION_NAME]["enabled"]
        upEnabled = networkConfiguration[UP_REGION_NAME]["enabled"]
        if (noiseAmplitude == 0
            and spEnabled
            and tmEnabled
            and not upEnabled):
          self.assertEqual(predictionAccuracy, 100)
        elif (noiseAmplitude == 0
              and spEnabled
              and not tmEnabled
              and not upEnabled):
          self.assertEqual(predictionAccuracy, 98.75)
        elif (noiseAmplitude == 1.0
              and spEnabled
              and tmEnabled
              and not upEnabled):
          self.assertEqual(predictionAccuracy, 88.0)
        elif (noiseAmplitude == 1.0
              and spEnabled
              and not tmEnabled
              and not upEnabled):
          self.assertEqual(predictionAccuracy, 81.875)


  def tearDown(self):
    """
    Remove data files
    """
    for noiseAmplitude in WHITE_NOISE_AMPLITUDES:
      fileToDelete = os.path.join(DATA_DIR, "%s_%s.csv" % (OUTFILE_NAME,
                                                           noiseAmplitude))
      if os.path.exists(fileToDelete):
        os.remove(fileToDelete)



if __name__ == "__main__":
  unittest.main()
