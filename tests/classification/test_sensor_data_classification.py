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
import shutil
import unittest

import simplejson as json

from nupic.data.file_record_stream import FileRecordStream

from htmresearch.frameworks.classification.classification_network import (
  configureNetwork,
  trainNetwork)
from htmresearch.frameworks.classification.utils.sensor_data import (
  generateSensorData)
from htmresearch.frameworks.classification.utils.network_config import (
  generateSampleNetworkConfig,
  generateNetworkPartitions)

# Parameters to generate the artificial sensor data
OUTFILE_NAME = "white_noise"
SEQUENCE_LENGTH = 200
NUM_CATEGORIES = 3
NUM_RECORDS = 2400
WHITE_NOISE_AMPLITUDES = [0.0, 1.0]
SIGNAL_AMPLITUDES = [1.0]
SIGNAL_MEANS = [1.0]
SIGNAL_PERIODS = [20.0]

# Additional parameters to run the classification experiments 
RESULTS_DIR = "results"
MODEL_PARAMS_DIR = 'model_params'
DATA_DIR = "data"

# Classifier types
CLA_CLASSIFIER_TYPE = "py.CLAClassifierRegion"
KNN_CLASSIFIER_TYPE = "py.KNNClassifierRegion"



class TestSensorDataClassification(unittest.TestCase):
  """Test classification results for sensor data."""


  def setUp(self):
    with open("sensor_data_network_config.json", "rb") as jsonFile:
      self.templateNetworkConfig = json.load(jsonFile)


  def testClassificationAccuracy(self):
    """Test classification accuracy for sensor data."""

    networkConfigurations = generateSampleNetworkConfig(
      self.templateNetworkConfig, NUM_CATEGORIES)

    for networkConfig in networkConfigurations:
      for noiseAmplitude in WHITE_NOISE_AMPLITUDES:
        for signalMean in SIGNAL_MEANS:
          for signalAmplitude in SIGNAL_AMPLITUDES:
            for signalPeriod in SIGNAL_PERIODS:
              sensorType = networkConfig["sensorRegionConfig"].get(
                "regionType")
              spEnabled = networkConfig["sensorRegionConfig"].get(
                "regionEnabled")
              tmEnabled = networkConfig["tmRegionConfig"].get(
                "regionEnabled")
              upEnabled = networkConfig["tpRegionConfig"].get(
                "regionEnabled")
              classifierType = networkConfig["classifierRegionConfig"].get(
                "regionType")

              expParams = ("RUNNING EXPERIMENT WITH PARAMS:\n"
                           " * numRecords=%s\n"
                           " * signalAmplitude=%s\n"
                           " * signalMean=%s\n"
                           " * signalPeriod=%s\n"
                           " * noiseAmplitude=%s\n"
                           " * sensorType=%s\n"
                           " * spEnabled=%s\n"
                           " * tmEnabled=%s\n"
                           " * tpEnabled=%s\n"
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

              inputFile = generateSensorData(DATA_DIR,
                                             OUTFILE_NAME,
                                             signalMean,
                                             signalPeriod,
                                             SEQUENCE_LENGTH,
                                             NUM_RECORDS,
                                             signalAmplitude,
                                             NUM_CATEGORIES,
                                             noiseAmplitude)

              dataSource = FileRecordStream(streamID=inputFile)
              network = configureNetwork(dataSource,
                                         networkConfig)
              partitions = generateNetworkPartitions(networkConfig,
                                                     NUM_RECORDS)

              classificationAccuracy = trainNetwork(network, networkConfig,
                                                  partitions, NUM_RECORDS)

              if (noiseAmplitude == 0
                  and signalMean == 1.0
                  and signalAmplitude == 1.0
                  and signalPeriod == 20.0
                  and classifierType == KNN_CLASSIFIER_TYPE
                  and spEnabled
                  and tmEnabled
                  and not upEnabled):
                self.assertEqual(classificationAccuracy, 100.00)
              elif (noiseAmplitude == 0
                    and signalMean == 1.0
                    and signalAmplitude == 1.0
                    and signalPeriod == 20.0
                    and classifierType == CLA_CLASSIFIER_TYPE
                    and spEnabled
                    and tmEnabled
                    and not upEnabled):
                self.assertEqual(classificationAccuracy, 100.00)
              elif (noiseAmplitude == 0
                    and signalMean == 1.0
                    and signalAmplitude == 1.0
                    and signalPeriod == 20.0
                    and classifierType == CLA_CLASSIFIER_TYPE
                    and spEnabled
                    and not tmEnabled
                    and not upEnabled):
                self.assertEqual(classificationAccuracy, 100.00)
              elif (noiseAmplitude == 1.0
                    and signalMean == 1.0
                    and signalAmplitude == 1.0
                    and signalPeriod == 20.0
                    and classifierType == CLA_CLASSIFIER_TYPE
                    and spEnabled
                    and tmEnabled
                    and not upEnabled):
                # using AlmostEqual until the random bug issue is fixed
                self.assertAlmostEqual(classificationAccuracy, 80, delta=5)
              elif (noiseAmplitude == 1.0
                    and signalMean == 1.0
                    and signalAmplitude == 1.0
                    and signalPeriod == 20.0
                    and classifierType == CLA_CLASSIFIER_TYPE
                    and spEnabled
                    and not tmEnabled
                    and not upEnabled):
                # using AlmostEqual until the random bug issue is fixed
                self.assertAlmostEqual(classificationAccuracy, 81, delta=5)


  def tearDown(self):
    shutil.rmtree(DATA_DIR)



if __name__ == "__main__":
  unittest.main()
