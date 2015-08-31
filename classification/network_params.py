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
from sensor_data_exp_settings import NUM_CATEGORIES

# Regions verbosity
VERBOSITY = 0

# Region names
SENSOR_REGION_NAME = "sensorRegion"
SP_REGION_NAME = "spRegion"
TM_REGION_NAME = "tmRegion"
UP_REGION_NAME = "upRegion"
CLASSIFIER_REGION_NAME = "classifierRegion"

# Region types
SENSOR_TYPE = "py.RecordSensor"
SP_REGION_TYPE = "py.SPRegion"
TM_REGION_TYPE = "py.TPRegion"
UP_REGION_TYPE = "py.UPRegion"
CLA_CLASSIFIER_TYPE = "py.CLAClassifierRegion"
KNN_CLASSIFIER_TYPE = "py.KNNClassifierRegion"

# Encoder names
SCALAR_ENCODER_NAME = "scalarEncoder"
SCALAR_ENCODER_FIELD_NAME = "y"

# Sensor region params
RECORD_SENSOR_PARAMS = {
  "verbosity": VERBOSITY,
  "numCategories": NUM_CATEGORIES
}

SCALAR_ENCODER_PARAMS = {
  "name": SCALAR_ENCODER_NAME,
  "fieldname": SCALAR_ENCODER_FIELD_NAME,
  "type": "ScalarEncoder",
  "n": 256,
  "w": 21,
  "minval": None,
  "maxval": None
}

# Classifier region params
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

# Spatial pooler region params
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

# Temporal Memory region params
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

# Union pooler region params
# TODO: Don't know what the UP params are yet.
UP_PARAMS = {}

# A network configuration specifies what regions to add to a network and what
# parameters to use for each one of these regions.
# Classifier and Sensor regions are always enabled but can have different 
# types. Make sure the params matches the type
NETWORK_CONFIGURATION = {
  SENSOR_REGION_NAME:
    {
      "name": SENSOR_REGION_NAME,
      "type": SENSOR_TYPE,
      "params": RECORD_SENSOR_PARAMS,
      "encoders": {
        SCALAR_ENCODER_NAME: SCALAR_ENCODER_PARAMS,
      }
    },
  SP_REGION_NAME:
    {
      "name": SP_REGION_NAME,
      "type": SP_REGION_TYPE,
      "params": SP_PARAMS,
      "enabled": True,
    },
  TM_REGION_NAME:
    {
      "name": TM_REGION_NAME,
      "type": TM_REGION_TYPE,
      "params": TM_PARAMS,
      "enabled": True,
    },
  UP_REGION_NAME:
    {
      "name": UP_REGION_NAME,
      "type": UP_REGION_TYPE,
      "params": UP_PARAMS,
      "enabled": False,
    },
  CLASSIFIER_REGION_NAME:
    {
      "name": CLASSIFIER_REGION_NAME,
      "type": CLA_CLASSIFIER_TYPE,
      "params": CLA_CLASSIFIER_PARAMS
    }
}
