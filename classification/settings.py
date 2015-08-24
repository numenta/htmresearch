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


# Parameters to generate the artificial sensor data
OUTFILE_NAME = "white_noise"
SEQUENCE_LENGTH = 200
NUM_CATEGORIES = 3
NUM_RECORDS = 2400
DEFAULT_WHITE_NOISE_AMPLITUDE = 10.0
WHITE_NOISE_AMPLITUDES = [0.0, 1.0]
SIGNAL_AMPLITUDE = 1.0
SIGNAL_MEAN = 1.0
SIGNAL_PERIOD = 20.0

# Additional parameters to run the classification experiments 
RESULTS_DIR = "results"
MODEL_PARAMS_DIR = 'model_params'
DATA_DIR = "data"

# Classification network parameters
VERBOSITY = 0

CATEGORY_ENCODER_PARAMS = {
  "name": 'label',
  "w": 21,
  "categoryList": range(NUM_CATEGORIES)
}

RECORD_SENSOR_PARAMS = {
  "verbosity": VERBOSITY,
  "numCategories": NUM_CATEGORIES
}

SEQ_CLASSIFIER_PARAMS = {
  "implementation": "py",
  "clVerbosity": VERBOSITY
}

CLA_CLASSIFIER_PARAMS = {
  "steps": "0",
  "implementation": "cpp",
  "maxCategoryCount": NUM_CATEGORIES,
  "clVerbosity": VERBOSITY
}

KNN_CLASSIFIER_PARAMS = {
  "k": 1,
  'distThreshold': 0,
  'maxCategoryCount': NUM_CATEGORIES,
}

SP_PARAMS = {
  "spVerbosity": VERBOSITY,
  "spatialImp": "cpp",
  "globalInhibition": 1,
  "columnCount": 2048,
  "numActiveColumnsPerInhArea": 40,
  "seed": 1956,
  "potentialPct": 0.8,
  "synPermConnected": 0.1,
  "synPermActiveInc": 0.0001,
  "synPermInactiveDec": 0.0005,
  "maxBoost": 1.0,
}

TM_PARAMS = {
  "verbosity": VERBOSITY,
  "columnCount": 2048,
  "cellsPerColumn": 32,
  "seed": 1960,
  "temporalImp": "tm_py",
  "newSynapseCount": 20,
  "maxSynapsesPerSegment": 32,
  "maxSegmentsPerCell": 128,
  "initialPerm": 0.21,
  "permanenceInc": 0.1,
  "permanenceDec": 0.1,
  "globalDecay": 0.0,
  "maxAge": 0,
  "minThreshold": 9,
  "activationThreshold": 12,
  "outputType": "normal",
  "pamLength": 3,
}

UP_PARAMS = {}  # TODO: Don't know what the UP params are yet.

# Region names for the network
SENSOR_REGION_NAME = "sensorRegion"
SP_REGION_NAME = "spRegion"
TM_REGION_NAME = "tmRegion"
UP_REGION_NAME = "upRegion"
CLASSIFIER_REGION_NAME = "classifierRegion"

# Name of the partition used for the test set
TEST_PARTITION_NAME = "test"

# A list of configurations specifying what regions to add to a network
NETWORK_CONFIGURATIONS = [
  {
    SENSOR_REGION_NAME:
      {
        "type": "py.RecordSensor",
        "params": RECORD_SENSOR_PARAMS
      },
    SP_REGION_NAME:
      {
        "enabled": True,
        "params": SP_PARAMS,
      },
    TM_REGION_NAME:
      {
        "enabled": True,
        "params": TM_PARAMS,
      },
    UP_REGION_NAME:
      {
        "enabled": False,
        "params": UP_PARAMS,
      },
    CLASSIFIER_REGION_NAME:
      {
        "type": "py.SequenceClassifierRegion",
        "params": SEQ_CLASSIFIER_PARAMS
      }
  },

  {
    SENSOR_REGION_NAME:
      {
        "type": "py.RecordSensor",
        "params": RECORD_SENSOR_PARAMS
      },
    SP_REGION_NAME:
      {
        "enabled": True,
        "params": SP_PARAMS,
      },
    TM_REGION_NAME:
      {
        "enabled": False,
        "params": TM_PARAMS,
      },
    UP_REGION_NAME:
      {
        "enabled": False,
        "params": UP_PARAMS,
      },
    CLASSIFIER_REGION_NAME:
      {
        "type": "py.SequenceClassifierRegion",
        "params": SEQ_CLASSIFIER_PARAMS
      }
  }
]
