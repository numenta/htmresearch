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

from settings import (NUM_CATEGORIES,
                      NUM_PHASES,
                      NUM_REPS,
                      SIGNAL_TYPES,
                      WHITE_NOISE_AMPLITUDES,
                      SIGNAL_AMPLITUDES,
                      SIGNAL_MEANS,
                      DATA_DIR,
                      VERBOSITY,
                      USE_JSON_CONFIG)

RESULTS_FILE = 'results/seq_classification_results.csv'



def print_and_save_results(classificationResults, expSetups):
  """
  Pretty print exp info and results and save them to CSV file
  :param classificationResults: (list of dict) classification results 
  :param expSetups: (list of dict) experiment setups
  """
  # we don't need the file path
  for expSetup in expSetups:
    del expSetup['filePath']

  with open(RESULTS_FILE, 'wb') as fw:
    writer = csv.writer(fw)
    c_headers = classificationResults[0].keys()
    e_headers = expSetups[0].keys()
    headers = e_headers + c_headers
    writer.writerow(headers)
    t = PrettyTable(headers)
    for i in range(len(expSetups)):
      c_row = [expSetups[i][eh] for eh in e_headers]
      e_row = [classificationResults[i][ch] for ch in c_headers]
      row = c_row + e_row
      writer.writerow(row)
      t.add_row(row)

    print '%s\n' % t
    print '==> Results saved to %s\n' % RESULTS_FILE



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

  expSetups = []
  classificationResults = []
  for signalType in SIGNAL_TYPES:
    for networkConfig in networkConfigurations:
      for noiseAmplitude in WHITE_NOISE_AMPLITUDES:
        for signalMean in SIGNAL_MEANS:
          for signalAmplitude in SIGNAL_AMPLITUDES:
            for numCategories in NUM_CATEGORIES:
              for numReps in NUM_REPS:
                for numPhases in NUM_PHASES:
                  spEnabled = networkConfig["sensorRegionConfig"].get(
                    "regionEnabled")
                  tmEnabled = networkConfig["tmRegionConfig"].get(
                    "regionEnabled")
                  upEnabled = networkConfig["tpRegionConfig"].get(
                    "regionEnabled")
                  classifierType = networkConfig["classifierRegionConfig"].get(
                    "regionType")

                  expSetup = generateSensorData(signalType,
                                                DATA_DIR,
                                                numPhases,
                                                numReps,
                                                signalMean,
                                                signalAmplitude,
                                                numCategories,
                                                noiseAmplitude)
                  expSetup['expId'] = len(expSetups)
                  expSetups.append(expSetup)
                  dataSource = FileRecordStream(streamID=expSetup['filePath'])
                  network = configureNetwork(dataSource,
                                             networkConfig)

                  partitions = generateNetworkPartitions(networkConfig,
                                                         expSetup['numPoints'])

                  traces = trainNetwork(network,
                                        networkConfig,
                                        partitions,
                                        expSetup['numPoints'],
                                        VERBOSITY)
                  finalAccuracy = traces['testClassificationAccuracyTrace'][-1]
                  classificationResults.append({
                    'spEnabled': spEnabled,
                    'tmEnabled': tmEnabled,
                    'upEnabled': upEnabled,
                    'classifierType': classifierType.split(".")[1],
                    'classificationAccuracy': finalAccuracy
                  })

  print_and_save_results(classificationResults, expSetups)



if __name__ == "__main__":
  run()
