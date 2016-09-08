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

""" Useful scripts for classification network configs. """

import copy
from collections import namedtuple

from htmresearch.frameworks.classification.network_training import (
  TEST_PARTITION_NAME)

# Region names
SENSOR_CONFIG = "sensorRegionConfig"
SP_CONFIG = "spRegionConfig"
TM_CONFIG = "tmRegionConfig"
TP_CONFIG = "tpRegionConfig"
CLASSIFIER_CONFIG = "classifierRegionConfig"

# Region types
SENSOR_TYPE = "py.RecordSensor"
SP_REGION_TYPE = "py.SPRegion"
TM_REGION_TYPE = "py.TPRegion"
TP_REGION_TYPE = "py.TPRegion"
CLA_CLASSIFIER_TYPE = "py.CLAClassifierRegion"
SDR_CLASSIFIER_TYPE = "py.SDRClassifierRegion"
KNN_CLASSIFIER_TYPE = "py.KNNClassifierRegion"



def generateNetworkPartitions(networkConfig, numRecords):
  """
  Find the number of partitions for the input data based on a specific
  networkConfig. Partition: Region names and index at which the
  region is to begin learning. The final partition is reserved as a test set.

  @param networkConfig: (dict) the configuration of this network.
  @param numRecords: (int) Number of records of the input dataset.
  @return partitions: (list of namedtuples) Paritions. I.e. the region names 
  and index at which the region is to begin learning. The final partition is 
  reserved as a test set.
  """
  Partition = namedtuple("Partition", "partName index")

  # Add regions to partition list in order of learning.
  regionConfigs = ("spRegionConfig", "tmRegionConfig", "tpRegionConfig",
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

  testIndex = numRecords * (maxNumPartitions - 1) / maxNumPartitions
  partitions.append(Partition(partName=TEST_PARTITION_NAME, index=testIndex))

  return partitions



def generateSampleNetworkConfig(templateNetworkConfig, maxCategoryCount):
  """
  Generate a sample network configuration for sensor data experiments, using a 
  template network params dict.

  @param templateNetworkConfig: (dict) template network config based on which
  other network configs are generated.
  @param maxCategoryCount: (int) the maximum number of categories to classify. 
  @return networkConfigurations: (list) network configs.
  """

  networkConfigurations = []

  # Basic classifier params
  CLA_CLASSIFIER_PARAMS = {
    "implementation": "cpp",
    "alpha": 0.01,
    "verbosity": 0,
    "steps": 0,
    "maxCategoryCount": 3
  }

  SDR_CLASSIFIER_PARAMS = {
    "implementation": "py",
    "alpha": 0.01,
    "verbosity": 0,
    "steps": 0,
    "maxCategoryCount": 3
  }

  ## SDR Classifier ##

  # SP and TM enabled. UP disabled. SDR Classifier.
  networkConfig = copy.deepcopy(templateNetworkConfig)
  networkConfig[SP_CONFIG]["regionEnabled"] = True
  networkConfig[TM_CONFIG]["regionEnabled"] = True
  networkConfig[TP_CONFIG]["regionEnabled"] = False
  networkConfig[CLASSIFIER_CONFIG]["regionType"] = SDR_CLASSIFIER_TYPE
  networkConfig[CLASSIFIER_CONFIG]["regionParams"] = SDR_CLASSIFIER_PARAMS
  networkConfigurations.append(networkConfig)

  # SP enabled. TM and UP disabled. SDR Classifier.
  networkConfig = copy.deepcopy(templateNetworkConfig)
  networkConfig[SP_CONFIG]["regionEnabled"] = True
  networkConfig[TM_CONFIG]["regionEnabled"] = False
  networkConfig[TP_CONFIG]["regionEnabled"] = False
  networkConfig[CLASSIFIER_CONFIG]["regionType"] = SDR_CLASSIFIER_TYPE
  networkConfig[CLASSIFIER_CONFIG]["regionParams"] = SDR_CLASSIFIER_PARAMS
  networkConfigurations.append(networkConfig)

  # SP and TM and UP enabled. SDR Classifier.
  networkConfig = copy.deepcopy(templateNetworkConfig)
  networkConfig[SP_CONFIG]["regionEnabled"] = True
  networkConfig[TM_CONFIG]["regionEnabled"] = True
  networkConfig[TP_CONFIG]["regionEnabled"] = True
  networkConfig[CLASSIFIER_CONFIG]["regionType"] = SDR_CLASSIFIER_TYPE
  networkConfig[CLASSIFIER_CONFIG]["regionParams"] = SDR_CLASSIFIER_PARAMS
  networkConfigurations.append(networkConfig)

  ## CLA Classifier ##

  # First config: SP and TM enabled. UP disabled. CLA Classifier.
  networkConfig = copy.deepcopy(templateNetworkConfig)
  networkConfig[SP_CONFIG]["regionEnabled"] = True
  networkConfig[TM_CONFIG]["regionEnabled"] = True
  networkConfig[TP_CONFIG]["regionEnabled"] = False
  networkConfig[CLASSIFIER_CONFIG]["regionType"] = CLA_CLASSIFIER_TYPE
  networkConfig[CLASSIFIER_CONFIG]["regionParams"] = CLA_CLASSIFIER_PARAMS
  networkConfigurations.append(networkConfig)

  # Second config: SP enabled. TM and UP disabled. CLA Classifier.
  networkConfig = copy.deepcopy(templateNetworkConfig)
  networkConfig[SP_CONFIG]["regionEnabled"] = True
  networkConfig[TM_CONFIG]["regionEnabled"] = False
  networkConfig[TP_CONFIG]["regionEnabled"] = False
  networkConfig[CLASSIFIER_CONFIG]["regionType"] = CLA_CLASSIFIER_TYPE
  networkConfig[CLASSIFIER_CONFIG]["regionParams"] = CLA_CLASSIFIER_PARAMS
  networkConfigurations.append(networkConfig)

  return networkConfigurations
