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
      "fieldName": "data",
      "fieldType": "float",
      "maxValue": 100.0,
      "minValue": 0.0
    },
    {
      "fieldName": "date",
      "fieldType": "datetime"
    }
  ],
  "streamDef": {
    "info": "NN5-103",
    "version": 1,
    "streams": [
      {
        "source": "file://data/NN5/NN5-103.csv",
        "columns": [
          "*"
        ]
      }
    ]
  },
  "inferenceType": "TemporalMultiStep",
  "inferenceArgs": {
    "predictionSteps": range(1, 2),
    "predictedField": "data"
  },
  "metricWindow": 1000,
  "swarmSize": "medium"
}
