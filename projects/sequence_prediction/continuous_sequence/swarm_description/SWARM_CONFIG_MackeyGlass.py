# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2015, Numenta, Inc.  Unless you have an agreement
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



SWARM_CONFIG = {
  "includedFields": [
    {
      "fieldName": "y",
      "fieldType": "float",
      "maxValue": 10.0,
      "minValue": 0.0
    }
  ],
  "streamDef": {
    "info": "MackeyGlass",
    "version": 1,
    "streams": [
      {
        "info": "MackeyGlass.csv",
        "source": "file://data/MackeyGlass.csv",
        "columns": [
          "*"
        ]
      }
    ]
  },
  "inferenceType": "TemporalMultiStep",
  "inferenceArgs": {
    "predictionSteps": [1],
    "predictedField": "y"
  },
  "metricWindow": 2000,
  "swarmSize": "medium"
}