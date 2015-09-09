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
try:
  import simplejson as json
except ImportError:
  import json

from collections import namedtuple
from nupic.data.file_record_stream import FileRecordStream

from classification_network import (configureNetwork,
                                    getRegionConfigParam,
                                    runNetwork)
from generate_sensor_data import generateData
from sensor_data_exp_settings import (NUM_CATEGORIES,
                                      TEST_PARTITION_NAME,
                                      SEQUENCE_LENGTH,
                                      OUTFILE_NAME,
                                      NUM_RECORDS,
                                      SIGNAL_AMPLITUDES,
                                      SIGNAL_MEANS,
                                      SIGNAL_PERIODS,
                                      WHITE_NOISE_AMPLITUDES,
                                      DATA_DIR)

# Region names
SENSOR_CONFIG = "sensorRegionConfig"
SP_CONFIG = "spRegionConfig"
TM_CONFIG = "tmRegionConfig"
UP_CONFIG = "upRegionConfig"
CLASSIFIER_CONFIG = "classifierRegionConfig"

# Region types
SENSOR_TYPE = "py.RecordSensor"
SP_REGION_TYPE = "py.SPRegion"
TM_REGION_TYPE = "py.TPRegion"
UP_REGION_TYPE = "py.UPRegion"
CLA_CLASSIFIER_TYPE = "py.CLAClassifierRegion"
KNN_CLASSIFIER_TYPE = "py.KNNClassifierRegion"

# Classifier region params
CLA_CLASSIFIER_PARAMS = {
  "steps": "0",
  "implementation": "cpp",
  "maxCategoryCount": NUM_CATEGORIES,
  "clVerbosity": 0
}

KNN_CLASSIFIER_PARAMS = {
  "k": 1,
  'distThreshold': 0,
  'maxCategoryCount': NUM_CATEGORIES,
}



def _generateExperimentsNetworkParams(templateNetworkConfig):
  """
  Generate a series of network params for each experiment, using a template
  network params dict.

  @param templateNetworkConfig: (dict) template network config based on which
  other network configs are generated.
  @return networkConfigurations: (list) network configs.
  """

  networkConfigurations = []

  # First config: SP and TM enabled. UP disabled. KNN Classifier.
  networkConfig = copy.deepcopy(templateNetworkConfig)
  networkConfig[SP_CONFIG]["regionEnabled"] = True
  networkConfig[TM_CONFIG]["regionEnabled"] = True
  networkConfig[UP_CONFIG]["regionEnabled"] = False
  networkConfig[CLASSIFIER_CONFIG]["regionType"] = KNN_CLASSIFIER_TYPE
  networkConfig[CLASSIFIER_CONFIG]["regionParams"] = KNN_CLASSIFIER_PARAMS
  networkConfigurations.append(networkConfig)

  # First config: SP and TM enabled. UP disabled. CLA Classifier.
  networkConfig = copy.deepcopy(templateNetworkConfig)
  networkConfig[SP_CONFIG]["regionEnabled"] = True
  networkConfig[TM_CONFIG]["regionEnabled"] = True
  networkConfig[UP_CONFIG]["regionEnabled"] = False
  networkConfig[CLASSIFIER_CONFIG]["regionType"] = CLA_CLASSIFIER_TYPE
  networkConfig[CLASSIFIER_CONFIG]["regionParams"] = CLA_CLASSIFIER_PARAMS
  networkConfigurations.append(networkConfig)

  # Second config: SP enabled. TM and UP disabled. CLA Classifier.
  networkConfig = copy.deepcopy(templateNetworkConfig)
  networkConfig[SP_CONFIG]["regionEnabled"] = True
  networkConfig[TM_CONFIG]["regionEnabled"] = False
  networkConfig[UP_CONFIG]["regionEnabled"] = False
  networkConfig[CLASSIFIER_CONFIG]["regionType"] = CLA_CLASSIFIER_TYPE
  networkConfig[CLASSIFIER_CONFIG]["regionParams"] = CLA_CLASSIFIER_PARAMS
  networkConfigurations.append(networkConfig)

  return networkConfigurations



def _findNumberOfPartitions(networkConfig, numRecords):
  """
  Find the number of partitions for the input data based on a specific
  networkConfig.

  @param networkConfig: (dict) the configuration of this network.
  @param numRecords: (int) Number of records of the input dataset.
  @return partitions: (list of namedtuples) Region names and index at which the
    region is to begin learning. The final partition is reserved as a test set.
  """
  Partition = namedtuple("Partition", "partName index")

  # Add regions to partition list in order of learning.
  regionConfigs = ("spRegionConfig", "tmRegionConfig", "upRegionConfig",
    "classifierRegionConfig")
  partitions = []
  maxNumPartitions = 5
  i = 0
  for region in regionConfigs:
    if networkConfig[region].get("regionEnabled"):
      learnIndex = i * numRecords / maxNumPartitions
      partitions.append(Partition(
        partName=networkConfig[region].get("regionName"), index=learnIndex))
      i += 1

  testIndex = numRecords * (maxNumPartitions-1) / maxNumPartitions
  partitions.append(Partition(partName=TEST_PARTITION_NAME, index=testIndex))

  return partitions



class TestSensorDataClassification(unittest.TestCase):
  """Test classification results for sensor data."""


  def setUp(self):
    self.filesToDelete = []
    with open("sensor_data_network_config.json", "rb") as jsonFile:
      self.templateNetworkConfig = json.load(jsonFile)


  def testClassificationAccuracy(self):
    """Test classification accuracy for sensor data."""

    networkConfigurations = _generateExperimentsNetworkParams(
      self.templateNetworkConfig)

    for networkConfig in networkConfigurations:
      for noiseAmplitude in WHITE_NOISE_AMPLITUDES:
        for signalMean in SIGNAL_MEANS:
          for signalAmplitude in SIGNAL_AMPLITUDES:
            for signalPeriod in SIGNAL_PERIODS:

              sensorType = getRegionConfigParam(networkConfig,
                                                "sensorRegionConfig",
                                                "regionType")
              spEnabled = getRegionConfigParam(networkConfig,
                                               "spRegionConfig",
                                               "regionEnabled")
              tmEnabled = getRegionConfigParam(networkConfig,
                                               "tmRegionConfig",
                                               "regionEnabled")
              upEnabled = getRegionConfigParam(networkConfig,
                                               "upRegionConfig",
                                               "regionEnabled")
              classifierType = getRegionConfigParam(networkConfig,
                                                    "classifierRegionConfig",
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

              self.filesToDelete.append(inputFile)
              dataSource = FileRecordStream(streamID=inputFile)
              network = configureNetwork(dataSource,
                                         networkConfig)
              partitions = _findNumberOfPartitions(networkConfig,
                                                   NUM_RECORDS)
              (numCorrect,
               numTestRecords,
               predictionAccuracy) = runNetwork(network,
                                                networkConfig,
                                                partitions,
                                                NUM_RECORDS)

              if (noiseAmplitude == 0
                  and signalMean == 1.0
                  and signalAmplitude == 1.0
                  and signalPeriod == 20.0
                  and classifierType == KNN_CLASSIFIER_TYPE
                  and spEnabled
                  and tmEnabled
                  and not upEnabled):
                self.assertEqual(predictionAccuracy, 100.00)
              elif (noiseAmplitude == 0
                    and signalMean == 1.0
                    and signalAmplitude == 1.0
                    and signalPeriod == 20.0
                    and classifierType == CLA_CLASSIFIER_TYPE
                    and spEnabled
                    and tmEnabled
                    and not upEnabled):
                self.assertEqual(predictionAccuracy, 100.00)
              elif (noiseAmplitude == 0
                    and signalMean == 1.0
                    and signalAmplitude == 1.0
                    and signalPeriod == 20.0
                    and classifierType == CLA_CLASSIFIER_TYPE
                    and spEnabled
                    and not tmEnabled
                    and not upEnabled):
                self.assertEqual(predictionAccuracy, 100.00)
              elif (noiseAmplitude == 1.0
                    and signalMean == 1.0
                    and signalAmplitude == 1.0
                    and signalPeriod == 20.0
                    and classifierType == CLA_CLASSIFIER_TYPE
                    and spEnabled
                    and tmEnabled
                    and not upEnabled):
                self.assertEqual(predictionAccuracy, 79.79)
              elif (noiseAmplitude == 1.0
                    and signalMean == 1.0
                    and signalAmplitude == 1.0
                    and signalPeriod == 20.0
                    and classifierType == CLA_CLASSIFIER_TYPE
                    and spEnabled
                    and not tmEnabled
                    and not upEnabled):
                self.assertEqual(predictionAccuracy, 81.46)


  def tearDown(self):
    for fileToDelete in self.filesToDelete:
      if os.path.exists(fileToDelete):
        os.remove(fileToDelete)



if __name__ == "__main__":
  unittest.main()
