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
import copy
import os
import unittest

from classification_network import configureNetwork, runNetwork
from generate_sensor_data import generateData
from sensor_data_exp_settings import (NUM_CATEGORIES,
                                      SEQUENCE_LENGTH,
                                      OUTFILE_NAME,
                                      NUM_RECORDS,
                                      SIGNAL_AMPLITUDES,
                                      SIGNAL_MEANS,
                                      SIGNAL_PERIODS,
                                      WHITE_NOISE_AMPLITUDES,
                                      DATA_DIR)
from network_params import (SENSOR_REGION_NAME,
                            SP_REGION_NAME,
                            TM_REGION_NAME,
                            UP_REGION_NAME,
                            CLASSIFIER_REGION_NAME,
                            SENSOR_TYPE,
                            CLA_CLASSIFIER_TYPE,
                            CLA_CLASSIFIER_PARAMS,
                            KNN_CLASSIFIER_TYPE,
                            KNN_CLASSIFIER_PARAMS)
from network_params import NETWORK_CONFIGURATION



def _generateNetworkConfigurations():
  """
  Generate a series of network configurations.
  @return networkConfigurations: (list) network configs.
  """

  networkConfigurations = []

  # First config: SP and TM enabled. UP disabled. CLA Classifier.
  baseNetworkConfig = copy.deepcopy(NETWORK_CONFIGURATION)
  baseNetworkConfig[SP_REGION_NAME]["enabled"] = True
  baseNetworkConfig[TM_REGION_NAME]["enabled"] = True
  baseNetworkConfig[UP_REGION_NAME]["enabled"] = False
  baseNetworkConfig[CLASSIFIER_REGION_NAME]["type"] = CLA_CLASSIFIER_TYPE
  baseNetworkConfig[CLASSIFIER_REGION_NAME]["params"] = CLA_CLASSIFIER_PARAMS
  networkConfigurations.append(baseNetworkConfig)

  # Second config: SP enabled. TM and UP disabled. CLA Classifier.
  baseNetworkConfig = copy.deepcopy(NETWORK_CONFIGURATION)
  baseNetworkConfig[SP_REGION_NAME]["enabled"] = True
  baseNetworkConfig[TM_REGION_NAME]["enabled"] = False
  baseNetworkConfig[UP_REGION_NAME]["enabled"] = False
  baseNetworkConfig[CLASSIFIER_REGION_NAME]["type"] = CLA_CLASSIFIER_TYPE
  baseNetworkConfig[CLASSIFIER_REGION_NAME]["params"] = CLA_CLASSIFIER_PARAMS
  networkConfigurations.append(baseNetworkConfig)

  # First config: SP and TM enabled. UP disabled. KNN Classifier.
  baseNetworkConfig = copy.deepcopy(NETWORK_CONFIGURATION)
  baseNetworkConfig[SP_REGION_NAME]["enabled"] = True
  baseNetworkConfig[TM_REGION_NAME]["enabled"] = True
  baseNetworkConfig[UP_REGION_NAME]["enabled"] = False
  baseNetworkConfig[CLASSIFIER_REGION_NAME]["type"] = KNN_CLASSIFIER_TYPE
  baseNetworkConfig[CLASSIFIER_REGION_NAME]["params"] = KNN_CLASSIFIER_PARAMS
  networkConfigurations.append(baseNetworkConfig)

  return networkConfigurations



class TestSensorDataClassification(unittest.TestCase):
  """
  Test classification results for sensor data. 
  """


  def testClassificationAccuracy(self):

    """
    Test classification accuracy for sensor data.
    """

    networkConfigurations = _generateNetworkConfigurations()

    for networkConfig in networkConfigurations:
      for noiseAmplitude in WHITE_NOISE_AMPLITUDES:
        for signalMean in SIGNAL_MEANS:
          for signalAmplitude in SIGNAL_AMPLITUDES:
            for signalPeriod in SIGNAL_PERIODS:

              sensorType = networkConfig[SENSOR_REGION_NAME]["type"]
              spEnabled = networkConfig[SP_REGION_NAME]["enabled"]
              tmEnabled = networkConfig[TM_REGION_NAME]["enabled"]
              upEnabled = networkConfig[UP_REGION_NAME]["enabled"]
              classifierType = networkConfig[CLASSIFIER_REGION_NAME]["type"]

              expParams = ("RUNNING EXPERIMENT WITH PARAMS:\n"
                           " * numRecords=%s\n"
                           " * signalAmplitude=%s\n"
                           " * signalMean=%s\n"
                           " * signalPeriod=%s\n"
                           " * noiseAmplitude=%s\n"
                           " * sensorType=%s\n"
                           " * spEnabled=%s\n"
                           " * tmEnabled=%s\n"
                           " * upEnabled=%s\n"
                           " * classifierType=%s\n"
                           ) % (NUM_RECORDS,
                                signalAmplitude,
                                signalMean,
                                signalPeriod,
                                noiseAmplitude,
                                sensorType.split(".")[1],
                                spEnabled,
                                tmEnabled,
                                upEnabled,
                                classifierType.split(".")[1])
              print expParams

              inputFile = generateData(DATA_DIR,
                                       OUTFILE_NAME,
                                       signalMean,
                                       signalPeriod,
                                       SEQUENCE_LENGTH,
                                       NUM_RECORDS,
                                       signalAmplitude,
                                       NUM_CATEGORIES,
                                       noiseAmplitude)

              network = configureNetwork(inputFile,
                                         networkConfig)

              (numCorrect,
               numTestRecords,
               predictionAccuracy) = runNetwork(network,
                                                networkConfig,
                                                NUM_RECORDS)

              if (noiseAmplitude == 0
                  and signalMean == 1.0
                  and signalAmplitude == 1.0
                  and signalPeriod == 20.0
                  and classifierType == KNN_CLASSIFIER_TYPE
                  and spEnabled
                  and tmEnabled
                  and not upEnabled):
                self.assertEqual(predictionAccuracy, 100)
              elif (noiseAmplitude == 0
                    and signalMean == 1.0
                    and signalAmplitude == 1.0
                    and signalPeriod == 20.0
                    and classifierType == CLA_CLASSIFIER_TYPE
                    and spEnabled
                    and tmEnabled
                    and not upEnabled):
                self.assertEqual(predictionAccuracy, 100)
              elif (noiseAmplitude == 0
                    and signalMean == 1.0
                    and signalAmplitude == 1.0
                    and signalPeriod == 20.0
                    and classifierType == CLA_CLASSIFIER_TYPE
                    and spEnabled
                    and not tmEnabled
                    and not upEnabled):
                self.assertEqual(predictionAccuracy, 98.75)
              elif (noiseAmplitude == 1.0
                    and signalMean == 1.0
                    and signalAmplitude == 1.0
                    and signalPeriod == 20.0
                    and classifierType == CLA_CLASSIFIER_TYPE
                    and spEnabled
                    and tmEnabled
                    and not upEnabled):
                self.assertEqual(predictionAccuracy, 88.0)
              elif (noiseAmplitude == 1.0
                    and signalMean == 1.0
                    and signalAmplitude == 1.0
                    and signalPeriod == 20.0
                    and classifierType == CLA_CLASSIFIER_TYPE
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
