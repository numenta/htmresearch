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
import csv
import json
import os
import sys
import numpy

from pkg_resources import resource_filename

from nupic.algorithms.anomaly import computeRawAnomalyScore
from nupic.data.file_record_stream import FileRecordStream
from nupic.engine import Network
from nupic.encoders import MultiEncoder, ScalarEncoder, DateEncoder

_VERBOSITY = 0  # how chatty the demo should be
_SEED = 1956  # the random seed used throughout
_INPUT_FILE_PATH = resource_filename(
  "nupic.datafiles", "extra/hotgym/rec-center-hourly.csv"
)
_OUTPUT_PATH = "network-demo-output.csv"
_NUM_RECORDS = 2000

# Config field for SPRegion
SP_PARAMS = {
    "spVerbosity": _VERBOSITY,
    "spatialImp": "cpp",
    "globalInhibition": 1,
    "columnCount": 1024,
    #"columnCount": 2048,
    # This must be set before creating the SPRegion
    "inputWidth": 0,
    "numActiveColumnsPerInhArea": 40,
    "seed": 1956,
    "potentialPct": 0.8,
    "synPermConnected": 0.1,
    "synPermActiveInc": 0.0001,
    "synPermInactiveDec": 0.0005,
    "maxBoost": 1.0,
}

# Config field for TPRegion
TP_PARAMS = {
    "verbosity": _VERBOSITY,
    "columnCount": 1024,
    "cellsPerColumn": 8,
    "inputWidth": 1024,
    #"columnCount": 2048,
    #"cellsPerColumn": 32,
    #"inputWidth": 2048,
    "seed": 1960,
    "temporalImp": "cpp",
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
    "computePredictedActiveCellIndices": 1,
}


# Config field for UPRegion
UP_PARAMS = {
    "spVerbosity": _VERBOSITY,
    "globalInhibition": 1,
    "columnCount": 128,
    #"columnCount": 2048,
    # This must be set before creating the UPRegion
    "inputWidth": 0,
    "numActiveColumnsPerInhArea": 20,
    #"numActiveColumnsPerInhArea": 40,
    "seed": 1956,
    "stimulusThreshold": 0,
    "synPermInactiveDec": 0.01,
    "synPermActiveInc": 0.1,
    "synPermConnected": 0.1,
    "potentialPct": 0.5,
    "globalInhibition": 1,
    "minPctOverlapDutyCycle": 0.001,
    "minPctActiveDutyCycle": 0.001,
    "dutyCyclePeriod": 1000,
    "maxBoost": 10.0,
    "spVerbosity": 0,
    "wrapAround": 1,
    "activeOverlapWeight": 1.0,
    "predictedActiveOverlapWeight": 0.0,
    "fixedPoolingActivationBurst": 0,
    "maxUnionActivity": 0.20,
    "poolerType": "union",
}



def createEncoder():
  """Create the encoder instance for our test and return it."""
  consumption_encoder = ScalarEncoder(21, 0.0, 100.0, n=50, name="consumption",
      clipInput=True)
  time_encoder = DateEncoder(timeOfDay=(21, 9.5), name="timestamp_timeOfDay")

  encoder = MultiEncoder()
  encoder.addEncoder("consumption", consumption_encoder)
  encoder.addEncoder("timestamp", time_encoder)

  return encoder



def createNetwork(dataSource):
  """Create the Network instance.

  The network has a sensor region reading data from `dataSource` and passing
  the encoded representation to an SPRegion. The SPRegion output is passed to
  a TPRegion.

  :param dataSource: a RecordStream instance to get data from
  :returns: a Network instance ready to run
  """
  network = Network()

  # Our input is sensor data from the gym file. The RecordSensor region
  # allows us to specify a file record stream as the input source via the
  # dataSource attribute.
  network.addRegion("sensor", "py.RecordSensor",
                    json.dumps({"verbosity": _VERBOSITY}))
  sensor = network.regions["sensor"].getSelf()
  # The RecordSensor needs to know how to encode the input values
  sensor.encoder = createEncoder()
  # Specify the dataSource as a file record stream instance
  sensor.dataSource = dataSource

  # Create the spatial pooler region
  SP_PARAMS["inputWidth"] = sensor.encoder.getWidth()
  network.addRegion("spatialPoolerRegion", "py.SPRegion", json.dumps(SP_PARAMS))

  # Link the SP region to the sensor input
  network.link("sensor", "spatialPoolerRegion", "UniformLink", "")
  network.link("sensor", "spatialPoolerRegion", "UniformLink", "",
               srcOutput="resetOut", destInput="resetIn")
  network.link("spatialPoolerRegion", "sensor", "UniformLink", "",
               srcOutput="spatialTopDownOut", destInput="spatialTopDownIn")
  network.link("spatialPoolerRegion", "sensor", "UniformLink", "",
               srcOutput="temporalTopDownOut", destInput="temporalTopDownIn")

  # Add the TPRegion on top of the SPRegion
  # TODO: Needs TMRegion
  network.addRegion("temporalMemoryRegion", "py.TPRegion",
                    json.dumps(TP_PARAMS))

  network.link("spatialPoolerRegion", "temporalMemoryRegion", "UniformLink", "")
  network.link("temporalMemoryRegion", "spatialPoolerRegion", "UniformLink", "",
               srcOutput="topDownOut", destInput="topDownIn")

  # Register UPRegion since we aren't in nupic
  curDirectory = os.path.dirname(os.path.abspath(__file__))
  # directory containing the union pooler directory is 2 directories above this file
  unionPoolerDirectory = os.path.split((os.path.split(curDirectory))[0])[0]
  sys.path.append(unionPoolerDirectory)
  Network.registerRegionPackage("union_pooling")

  # Add the UPRegion on top of the TPRegion
  temporal = network.regions["temporalMemoryRegion"].getSelf()
  UP_PARAMS["inputWidth"] = temporal.getOutputElementCount("bottomUpOut")
  network.addRegion("unionPoolerRegion", "py.PoolingRegion", json.dumps(UP_PARAMS))

  network.link("temporalMemoryRegion", "unionPoolerRegion", "UniformLink", "",
               srcOutput="activeCells", destInput="activeCells")
  network.link("temporalMemoryRegion", "unionPoolerRegion", "UniformLink", "",
               srcOutput="predictedActiveCells", destInput="predictedActiveCells")

  network.initialize()

  spatial = network.regions["spatialPoolerRegion"].getSelf()
  # Make sure learning is enabled (this is the default)
  spatial.setParameter("learningMode", 1, True)
  # We want temporal anomalies so disable anomalyMode in the SP. This mode is
  # used for computing anomalies in a non-temporal model.
  spatial.setParameter("anomalyMode", 1, False)

  # Enable topDownMode to get the predicted columns output
  temporal.setParameter("topDownMode", 1, True)
  # Make sure learning is enabled (this is the default)
  temporal.setParameter("learningMode", 1, True)
  # Enable inference mode so we get predictions
  temporal.setParameter("inferenceMode", 1, True)
  temporal.setParameter("computePredictedActiveCellIndices", 1, True)

  union = network.regions["unionPoolerRegion"].getSelf()
  # Make sure learning is enabled (this is the default)
  union.setParameter("learningMode", 1, True)

  return network



def runNetwork(network, writer):
  """Run the network and write output to writer.

  :param network: a Network instance to run
  :param writer: a csv.writer instance to write output to
  """
  sensorRegion = network.regions["sensor"]
  spatialPoolerRegion = network.regions["spatialPoolerRegion"]
  temporalMemoryRegion = network.regions["temporalMemoryRegion"]
  unionPoolerRegion = network.regions["unionPoolerRegion"]

  prevPredictedColumns = []

  for i in xrange(_NUM_RECORDS):
    # Run the network for a single iteration
    network.run(1)

    # Write out the active cells along with the record number and consumption
    # value.
    activeCells = unionPoolerRegion.getOutputData("mostActiveCells")
    consumption = sensorRegion.getOutputData("sourceOut")[0]
    writer.writerow((i, consumption, numpy.sum(activeCells)))



if __name__ == "__main__":
  dataSource = FileRecordStream(streamID=_INPUT_FILE_PATH)

  network = createNetwork(dataSource)
  outputPath = os.path.join(os.path.dirname(__file__), _OUTPUT_PATH)
  with open(outputPath, "w") as outputFile:
    writer = csv.writer(outputFile)
    print "Writing output to %s" % outputPath
    runNetwork(network, writer)
