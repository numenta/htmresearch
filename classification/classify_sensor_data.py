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

from nupic.data.file_record_stream import FileRecordStream

from classification_network import createNetwork
from generate_sensor_data import generateData
from generate_model_params import findMinMax
from settings import (NUM_CATEGORIES,
                      SEQUENCE_LENGTH,
                      OUTFILE_NAME,
                      NUM_RECORDS,
                      NETWORK_CONFIGURATIONS,
                      SIGNAL_AMPLITUDE,
                      SIGNAL_MEAN,
                      SIGNAL_PERIOD,
                      WHITE_NOISE_AMPLITUDES,
                      DATA_DIR)


def runNetwork(net, numRecords, partitions):
  """
  Run the network and write classification results output.
  @param net: a Network instance to run.
  @param partitions: list of indices at which training begins for the SP, TM,
      and classifier regions, respectively, e.g. [100, 200, 300].

  """
  sensorRegion = net.regions["sensor"]
  spatialPoolerRegion = net.regions["SP"]
  temporalMemoryRegion = net.regions["TM"]
  classifierRegion = net.regions["classifier"]

  phaseInfo = ("-> Training SP. Index=0. LEARNING: SP is ON | TM is OFF | "
               "Classifier is OFF \n")

  print phaseInfo

  numCorrect = 0
  numTestRecords = 0
  for i in xrange(numRecords):
    # Run the network for a single iteration
    net.run(1)

    # NOTE: To be able to extract a category, one of the field of the the
    # dataset needs to have the flag C so it can be recognized as a category
    # by the FileRecordStream instance.
    actualValue = sensorRegion.getOutputData("categoryOut")[0]

    # Various info from the network, useful for debugging & monitoring
    anomalyScore = temporalMemoryRegion.getOutputData("anomalyScore")
    debugInfo = ("=> INDEX=%s |  actualValue=%s | anomalyScore=%s \n" % (
      i, actualValue, anomalyScore))
    # print debugInfo

    # SP has been trained. Now start training the TM too.
    if i == partitions[0]:
      temporalMemoryRegion.setParameter("learningMode", True)
      phaseInfo = ("-> Training TM. Index=%s. LEARNING: SP is ON | "
                   "TM is ON | Classifier is OFF \n" % i)
      print phaseInfo

    # Start training the classifier as well.
    elif i == partitions[1]:
      classifierRegion.setParameter("learningMode", True)
      phaseInfo = ("-> Training Classifier. Index=%s. LEARNING: SP is OFF | "
                   "TM is ON | Classifier is ON \n" % i)

      print phaseInfo

    # Stop training.
    elif i == partitions[2]:
      spatialPoolerRegion.setParameter("learningMode", False)
      temporalMemoryRegion.setParameter("learningMode", False)
      classifierRegion.setParameter("learningMode", False)
      phaseInfo = ("-> Test. Index=%s. LEARNING: SP is OFF | TM is OFF | "
                   "Classifier is OFF \n" % i)
      print phaseInfo

    # Evaluate the predictions on the test set.
    if i >= partitions[2]:

      inferredValue = classifierRegion.getOutputData("categoriesOut")[0]
      debugInfo = (" inferredValue=%s \n" % inferredValue)
      # print debugInfo

      if actualValue == inferredValue:
        numCorrect += 1

      numTestRecords += 1

  predictionAccuracy = 100.0 * numCorrect / numTestRecords

  results = ("RESULTS: accuracy=%s | %s correctly predicted records out of %s "
             "test records \n" % (predictionAccuracy,
                                  numCorrect,
                                  numTestRecords))
  print results

  return numCorrect, numTestRecords, predictionAccuracy



def configureNetwork(encoders,
                     networkConfiguration):
  """
  Configure the network for various experiment values.
  """


  # Create and run network on this data.
  #   Input data comes from a CSV file (scalar values, labels). The
  #   RecordSensor region allows us to specify a file record stream as the
  #   input source via the dataSource attribute.
  dataSource = FileRecordStream(streamID=inputFile)
  
  network = createNetwork(dataSource,
                          encoders,
                          networkConfiguration)

  # Need to init the network before it can run.
  network.initialize()
  return network



def findNumberOfPartitions(networkConfiguration, numRecords):
  """
  Find the number of partitions for the input data based on a specific
  networkConfiguration. 
  
  @param networkConfiguration (dict) the configuration of this network. E.g.
    {
      "sensorRegion":
        {
          "type": "py.RecordSensor",
          "params": RECORD_SENSOR_PARAMS
        },
      "spRegion":
        {
          "enabled": True,
          "params": SP_PARAMS,
        },
      "tmRegion":
        {
          "enabled": True,
          "params": TM_PARAMS,
        },
      "upRegion":
        {
          "enabled": False,
          "params": UP_PARAMS,
        },
      "classifierRegion":
        {
          "type": "py.CLAClassifierRegion",
          "params": CLA_CLASSIFIER_PARAMS
            
        }
    }
  
  @numRecords (int) Number of records of the input dataset.
  
  @return partitions (list) list of indices partitioning the input dataset. 
    The last partition is always the test set. All the partitions before are 
    used to train the regions in the network consecutively.
  """

  spEnabled = networkConfiguration["spRegion"]["enabled"]
  tmEnabled = networkConfiguration["tmRegion"]["enabled"]
  upEnabled = networkConfiguration["tmRegion"]["enabled"]

  if spEnabled and tmEnabled and upEnabled:
    numPartitions = 5
  elif spEnabled and tmEnabled:
    numPartitions = 4
  elif spEnabled:
    numPartitions = 3
  else:
    numPartitions = 2  # only the classifier, so just test and train partitions

  return [numRecords * i / numPartitions for i in xrange(numPartitions)]



def generateScalarEncoderParams(inputFile):
  minval, maxval = findMinMax(inputFile)
  scalarEncoderParams = {
    "name": "white_noise",
    "fieldname": "y",
    "type": "ScalarEncoder",
    "n": 256,
    "w": 21,
    "minval": minval,
    "maxval": maxval
  }
  return scalarEncoderParams
  



if __name__ == "__main__":

  for networkConfiguration in NETWORK_CONFIGURATIONS:
    for noiseAmplitude in WHITE_NOISE_AMPLITUDES:
      expParams = ("RUNNING EXPERIMENT WITH PARAMS:\n"
                   " * numRecords=%s\n"
                   " * signalAmplitude=%s\n"
                   " * signalMean=%s\n"
                   " * signalPeriod=%s\n"
                   " * networkConfiguration=%s\n"
                   ) % (NUM_RECORDS,
                        SIGNAL_AMPLITUDE,
                        SIGNAL_MEAN,
                        SIGNAL_PERIOD,
                        json.dumps(networkConfiguration, indent=2))
      print expParams


      # Generate the data, and get the min/max values
      inputFile = generateData(DATA_DIR,
                               OUTFILE_NAME,
                               SIGNAL_MEAN,
                               SIGNAL_PERIOD,
                               SEQUENCE_LENGTH,
                               NUM_RECORDS,
                               SIGNAL_AMPLITUDE,
                               noiseAmplitude)
      
      scalarEncoderParams = generateScalarEncoderParams(inputFile)
      
      encoders = {"sensor_data" : scalarEncoderParams}
      
      network = configureNetwork(encoders,
                                 networkConfiguration)

      partitions = findNumberOfPartitions(networkConfiguration)

      (numCorrect,
       numTestRecords,
       predictionAccuracy) = runNetwork(network,
                                        NUM_RECORDS,
                                        partitions)
