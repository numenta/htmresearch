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

import csv
from prettytable import PrettyTable
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

from settings import (OUTFILE_NAME,
                      SEQUENCE_LENGTH,
                      NUM_CATEGORIES,
                      NUM_RECORDS,
                      WHITE_NOISE_AMPLITUDES,
                      SIGNAL_AMPLITUDES,
                      SIGNAL_MEANS,
                      SIGNAL_PERIODS,
                      DATA_DIR,
                      VERBOSITY,
                      USE_JSON_CONFIG)

RESULTS_FILE = 'results/seq_classification_results.csv'



def run():
  """ Run classification network(s) on artificial sensor data """

  if USE_JSON_CONFIG:
    with open('config/network_configs.json', 'rb') as fr:
      networkConfigurations = json.load(fr)
  else:
    with open("config/network_config_template.json", "rb") as jsonFile:
      templateNetworkConfig = json.load(jsonFile)
      networkConfigurations = generateSampleNetworkConfig(templateNetworkConfig,
                                                          NUM_CATEGORIES)

  headers = ['numRecords', 'seqLength', 'numClasses', 'signalAmplitude',
             'signalMean', 'signalPeriod', 'noiseAmplitude', 'spEnabled',
             'tmEnabled', 'tpEnabled', 'classifierType',
             'classificationAccuracy']

  with open(RESULTS_FILE, 'wb') as fw:
    writer = csv.writer(fw)
    writer.writerow(headers)
    t = PrettyTable(headers)
    for networkConfig in networkConfigurations:
      for noiseAmplitude in WHITE_NOISE_AMPLITUDES:
        for signalMean in SIGNAL_MEANS:
          for signalAmplitude in SIGNAL_AMPLITUDES:
            for signalPeriod in SIGNAL_PERIODS:
              spEnabled = networkConfig["sensorRegionConfig"].get(
                "regionEnabled")
              tmEnabled = networkConfig["tmRegionConfig"].get(
                "regionEnabled")
              upEnabled = networkConfig["tpRegionConfig"].get(
                "regionEnabled")
              classifierType = networkConfig["classifierRegionConfig"].get(
                "regionType")
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
                                                    partitions, NUM_RECORDS,
                                                    VERBOSITY)

              results = [NUM_RECORDS,
                         SEQUENCE_LENGTH,
                         NUM_CATEGORIES,
                         signalAmplitude,
                         signalMean,
                         signalPeriod,
                         noiseAmplitude,
                         spEnabled,
                         tmEnabled,
                         upEnabled,
                         classifierType.split(".")[1],
                         classificationAccuracy]

              writer.writerow(results)
              t.add_row(results)
              
  print '%s\n' % t
  print '==> Results saved to %s\n' % RESULTS_FILE



if __name__ == "__main__":
  run()
