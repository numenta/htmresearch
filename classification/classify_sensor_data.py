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

from nupic.data.file_record_stream import FileRecordStream

from classification_network import createNetwork
from generate_model_params import findMinMax
from settings import (SENSOR_REGION_NAME,
                      SP_REGION_NAME,
                      TM_REGION_NAME,
                      UP_REGION_NAME,
                      CLASSIFIER_REGION_NAME,
                      TEST_PARTITION_NAME,
                      VERBOSITY)



def enableRegionLearning(network,
                         trainedRegionNames,
                         regionName,
                         recordNumber):
  """
  Enable learning for a specific region.
  
  @param network (Network) the network instance 
  @param trainedRegionNames (list) regions that have been trained on the 
  input data.
  @param regionName (str) name of the current region  
  @param recordNumber (int) value of the current record number 
  """

  region = network.regions[regionName]
  region.setParameter("learningMode", True)
  phaseInfo = ("-> Training '%s'. RecordNumber=%s. Learning is ON for %s, "
               "but OFF for the remaining regions." % (regionName,
                                                       recordNumber,
                                                       trainedRegionNames))
  print phaseInfo



def stopLearning(network, trainedRegionNames, recordNumber):
  """
  Disable learning for all trained regions.
  
  @param network (Network) the network instance 
  @param trainedRegionNames (list) regions that have been trained on the 
  input data.
  @param recordNumber (int) value of the current record number 
  """

  for regionName in trainedRegionNames:
    region = network.regions[regionName]
    region.setParameter("learningMode", False)

  phaseInfo = ("-> Test phase. RecordNumber=%s. "
               "Learning is OFF for all regions: %s" % (recordNumber,
                                                        trainedRegionNames))
  print phaseInfo



def runNetwork(network, networkConfiguration, numRecords):
  """
  Run the network and write classification results output.
  
  @param networkConfiguration (dict) the configuration of this network.
  @param numRecords (int) Number of records of the input dataset.
  @param network (Network) a Network instance to run.
  """

  partitions = findNumberOfPartitions(networkConfiguration, numRecords)
  sensorRegion = network.regions[SENSOR_REGION_NAME]
  classifierRegion = network.regions[CLASSIFIER_REGION_NAME]
  testIndex = partitions[-1][1]

  # keep track of the regions that have been trained
  trainedRegionNames = []

  # Enable learning for the first region
  firstRegionName = partitions[0][0]
  enableRegionLearning(network,
                       trainedRegionNames,
                       firstRegionName,
                       0)

  numCorrect = 0
  numTestRecords = 0
  for recordNumber in xrange(numRecords):
    # Run the network for a single iteration
    network.run(1)

    actualValue = sensorRegion.getOutputData("categoryOut")[0]

    partitionName = partitions[0][0]
    partitionIndex = partitions[0][1]

    # train all of the regions
    if partitionIndex == recordNumber:

      if partitionName == TEST_PARTITION_NAME:
        stopLearning(network, trainedRegionNames, recordNumber)

      else:
        partitions.pop(0)
        trainedRegionNames.append(partitionName)
        enableRegionLearning(network,
                             trainedRegionNames,
                             partitionName,
                             recordNumber)

    # Evaluate the predictions on the test set.
    if recordNumber >= testIndex:
      inferredValue = classifierRegion.getOutputData("categoriesOut")[0]
      if actualValue == inferredValue:
        numCorrect += 1
      elif VERBOSITY > 0:
        print ("[DEBUG] recordNum=%s, actualValue=%s, inferredValue=%s"
               % (recordNumber, actualValue, inferredValue))
      numTestRecords += 1

  predictionAccuracy = 100.0 * numCorrect / numTestRecords

  results = ("RESULTS: accuracy=%s | %s correctly predicted records out of %s "
             "test records \n" % (predictionAccuracy,
                                  numCorrect,
                                  numTestRecords))
  print results

  return numCorrect, numTestRecords, predictionAccuracy



def configureNetwork(inputFile,
                     networkConfiguration):
  """
  Configure the network for various experiment values.
  
  @param inputFile (str) file containing the input data that will be fed to the 
  network
  @param networkConfiguration (dict) the configuration of this network.
  """

  # create the network encoders for sensor data.  
  scalarEncoderParams = generateScalarEncoderParams(inputFile)
  encoders = {"sensor_data": scalarEncoderParams}

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
  
  @param networkConfiguration (dict) the configuration of this network.
  @param numRecords (int) Number of records of the input dataset.
  @return partitions (list of tuples) Region names and associated indices 
  partitioning the input dataset to indicate at which recordNumber it should 
  start learning. The remaining of the data (last partition) is used as a test 
  set. 
  """

  spEnabled = networkConfiguration[SP_REGION_NAME]["enabled"]
  tmEnabled = networkConfiguration[TM_REGION_NAME]["enabled"]
  upEnabled = networkConfiguration[UP_REGION_NAME]["enabled"]

  partitionNames = []
  if spEnabled and tmEnabled and upEnabled:
    numPartitions = 5
    partitionNames.extend([SP_REGION_NAME,
                           TM_REGION_NAME,
                           UP_REGION_NAME])
  elif spEnabled and tmEnabled:
    numPartitions = 4
    partitionNames.extend([SP_REGION_NAME,
                           TM_REGION_NAME])
  elif spEnabled:
    numPartitions = 3
    partitionNames.append(SP_REGION_NAME)
  else:
    numPartitions = 2  # only the classifier, so just test and train partitions
  partitionNames.append(CLASSIFIER_REGION_NAME)
  partitionNames.append(TEST_PARTITION_NAME)

  partitionIndices = [numRecords * i / numPartitions
                      for i in range(0, numPartitions)]

  return zip(partitionNames, partitionIndices)



def generateScalarEncoderParams(inputFile):
  """
  Introspect an input file and return the min and max values.
  
  @param inputFile: input file to introspect.
  @return: (dict) scalar encoder params
  """
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
