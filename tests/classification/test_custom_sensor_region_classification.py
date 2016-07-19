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
import logging
import shutil
import unittest
import simplejson
import sys

from nupic.data.file_record_stream import FileRecordStream

from htmresearch.frameworks.classification.classification_network import (
  configureNetwork, classifyNextRecord, TEST_PARTITION_NAME)
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

# Logger
_LOGGER = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO,
                    stream=sys.stdout)



class TestCustomSensorRegionClassification(unittest.TestCase):
  """Test CustomRecordSensor region for the classification of sensor data."""


  def setUp(self):
    shutil.rmtree(DATA_DIR, ignore_errors=True)  # Attempt to remove DATA_DIR
    with open("sensor_data_network_config.json", "rb") as jsonFile:
      self.templateNetworkConfig = simplejson.load(jsonFile)


  def _trainAndTestNetwork(self, network, networkConfig, networkPartitions,
                           dataSource):
    """
    Train the network.
  
    :param network: (Network) a Network instance to run.
    :param networkConfig: (dict) params for network regions.
    :param networkPartitions: (list of tuples) Region names and index at 
      which the region is to begin learning, including a test partition (the 
      last entry).
    :param dataSource: (InputStream) The data source used by the network to 
      get the next record.
    :return classificationAccuracy: (float) the error (in %) between the 
      predicted category and the actual category. If the actual category is 
      None, then return -1.
    """

    # Preserve original partitions
    partitions = copy.deepcopy(networkPartitions)
    classifierRegion = network.regions[networkConfig["classifierRegionConfig"]
      .get("regionName")]

    numCorrect = 0
    recordNumber = 0
    numTestRecords = 0
    dataSourceEmpty = False
    trainedRegionNames = []
    while not dataSourceEmpty:

      # Get next input data to feed to the network
      data = dataSource.getNextRecordDict()
      if not data:
        dataSourceEmpty = True
      else:

        # Classify input data and get inferred category
        timestamp = data["x"]
        value = data["y"]
        category = data["label"]
        classificationResults = classifyNextRecord(network, networkConfig,
                                                   timestamp, value, category)
        inferredCategory = classificationResults["bestInference"]

        # Update network learning rules
        if recordNumber == partitions[0][1]:
          currentRegionName = partitions[0][0]
          if currentRegionName == TEST_PARTITION_NAME:
            for regionName in trainedRegionNames:
              region = network.regions[regionName]
              region.setParameter("learningMode", False)
              _LOGGER.info("--> Learning OFF for %s" % regionName)
          else:
            partitions.pop(0)  # We're done with the current region
            trainedRegionNames.append(currentRegionName)
            network.regions[currentRegionName].setParameter("learningMode",
                                                            True)
            _LOGGER.info("--> Learning ON for %s" % currentRegionName)

        # Evaluate the predictions on the test set
        if recordNumber >= partitions[-1][1]:
          classifierRegion.setParameter("inferenceMode", True)

          if category == inferredCategory:
            numCorrect += 1
          _LOGGER.debug("recordNum=%s, actualValue=%s, inferredCategory=%s"
                        % (recordNumber, category, inferredCategory))
          numTestRecords += 1

        recordNumber += 1

    # Compute overall classification accuracy
    classificationAccuracy = round(100.0 * numCorrect / numTestRecords, 2)
    _LOGGER.info("RESULTS: accuracy=%s | %s correctly classified records "
                 "out of %s test records \n" % (classificationAccuracy,
                                                numCorrect,
                                                numTestRecords))

    return classificationAccuracy


  def _logInfo(self, numRecords, signalAmplitude, signalMean, signalPeriod,
               noiseAmplitude, sensorType, spEnabled, tmEnabled, upEnabled,
               classifierType):

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
                 ) % (numRecords,
                      signalAmplitude,
                      signalMean,
                      signalPeriod,
                      noiseAmplitude,
                      sensorType.split(".")[1],
                      spEnabled,
                      tmEnabled,
                      upEnabled,
                      classifierType.split(".")[1])

    _LOGGER.info(expParams)


  def testClassificationAccuracy(self):
    """Test classification accuracy for sensor data."""

    networkConfigurations = generateSampleNetworkConfig(
        self.templateNetworkConfig, NUM_CATEGORIES)

    for config in networkConfigurations:

      # Use custom region. That's what we want to test here.
      config["sensorRegionConfig"]["regionType"] = "py.CustomRecordSensor"

      # Get the region types for this network config.
      sensorType = config["sensorRegionConfig"].get(
          "regionType")
      spEnabled = config["sensorRegionConfig"].get(
          "regionEnabled")
      tmEnabled = config["tmRegionConfig"].get(
          "regionEnabled")
      upEnabled = config["tpRegionConfig"].get(
          "regionEnabled")
      classifierType = config["classifierRegionConfig"].get(
          "regionType")

      for noiseAmplitude in WHITE_NOISE_AMPLITUDES:
        for signalMean in SIGNAL_MEANS:
          for signalAmplitude in SIGNAL_AMPLITUDES:
            for signalPeriod in SIGNAL_PERIODS:

              self._logInfo(NUM_RECORDS, signalAmplitude, signalMean,
                            signalPeriod,
                            noiseAmplitude, sensorType, spEnabled, tmEnabled,
                            upEnabled,
                            classifierType)

              inputFile = generateSensorData(DATA_DIR,
                                             OUTFILE_NAME,
                                             signalMean,
                                             signalPeriod,
                                             SEQUENCE_LENGTH,
                                             NUM_RECORDS,
                                             signalAmplitude,
                                             NUM_CATEGORIES,
                                             noiseAmplitude)

              partitions = generateNetworkPartitions(config, NUM_RECORDS)
              dataSource = FileRecordStream(streamID=inputFile)
              network = configureNetwork(dataSource, config)
              # network.save("test.nta")
              # from nupic.engine import Network
              # network = Network("test.nta")
              classificationAccuracy = self._trainAndTestNetwork(network,
                                                                 config,
                                                                 partitions,
                                                                 dataSource)

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
