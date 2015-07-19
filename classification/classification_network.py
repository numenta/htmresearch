#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import numpy

try:
  import simplejson as json
except ImportError:
  import json

from nupic.data.file_record_stream import FileRecordStream
from nupic.encoders import CategoryEncoder, MultiEncoder, ScalarEncoder
from nupic.engine import Network


_VERBOSITY = 0

SP_PARAMS = {
    "spVerbosity": _VERBOSITY,
    "spatialImp": "cpp",
    "globalInhibition": 1,
    "columnCount": 2048,
    "inputWidth": 0,
    "numActiveColumnsPerInhArea": 40,
    "seed": 1956,
    "potentialPct": 0.8,
    "synPermConnected": 0.1,
    "synPermActiveInc": 0.0001,
    "synPermInactiveDec": 0.0005,
    "maxBoost": 1.0,
}

TM_PARAMS = {
    "verbosity": _VERBOSITY,
    "columnCount": 2048,
    "cellsPerColumn": 32,
    "inputWidth": 2048,
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

CLA_CLASSIFIER_PARAMS = {"steps": "1",
                         "implementation": "py"}



class ClassificationNetwork(object):
  """
  Factory to create a classification network:
    encoder -> SP -> TM -> (UP) -> classifier
  """

  def __init__(self,
               spParams=SP_PARAMS,
               tmParams=TM_PARAMS,
               classifierParams=CLA_CLASSIFIER_PARAMS,
               verbosity=_VERBOSITY):
    """
    @param spParams         (dict)        Spatial pooler parameters.
    @param tmParams         (dict)        Temporal memory parameters.
    @param classifierParams (dict)        CLA classifier parameters.
    @param verbosity        (dict)        Amount of info printed to the console.
    """
    self.spParams=spParams
    self.tmParams=tmParams
    self.classifierParams=classifierParams
    self.verbosity=verbosity
  
    # Init region variables
    self.encoder = MultiEncoder()
    self.network = Network()
    self.sensorRegion = None
    self.spatialPoolerRegion = None
    self.temporalMemoryRegion = None
    self.classifierRegion = None


  def createEncoder(self, name, newEncoder):
    """Add an encoder to the MultiEncoder."""
    self.encoder.addEncoder(name, newEncoder)


  def _initSensorRegion(self, dataSource):
    """
    Initializes the sensor region with an encoder and data source.
    
    @param dataSource   (RecordStream)  Sensor region reads data from here.
    @return             (int)           Region width to input to next region.
    """
    # Input data comes from a CSV file (scalar values, labels). The RecordSensor region
    # allows us to specify a file record stream as the input source via the
    # dataSource attribute.
    self.network.addRegion(
        "sensor", "py.RecordSensor", json.dumps({"verbosity": self.verbosity}))
    self.sensorRegion = self.network.regions["sensor"].getSelf()

    # Specify how RecordSensor encodes input values
    self.sensorRegion.encoder = self.encoder

    # Specify the dataSource as a file record stream instance
    self.sensorRegion.dataSource = dataSource

    return self.sensorRegion.encoder.getWidth()


  def _initSpatialPoolerRegion(self, prevRegionWidth):
    """
    Create the spatial pooler region.
    
    @return             (int)            Region width to input to next region.
    """
    self.spParams["inputWidth"] = prevRegionWidth
    self.network.addRegion("SP", "py.SPRegion", json.dumps(self.spParams))
    self.spatialPoolerRegion = self.network.regions["SP"]

    # Link the SP region to the sensor input
    self.network.link("sensor", "SP", "UniformLink", "")
    
    # Forward the sensor region sequence reset to the SP
    self.network.link("sensor", "SP", "UniformLink", "",
        srcOutput="resetOut", destInput="resetIn")
    
    # Make sure learning is ON
    self.spatialPoolerRegion.setParameter("learningMode", True)
    
    # Inference mode outputs the current inference (e.g. active columns). 
    # It's ok to always leave inference mode on - it's only there for some corner cases.
    self.spatialPoolerRegion.setParameter("inferenceMode", True)

    return self.spParams["columnCount"]


  def _initTemporalMemoryRegion(self, prevRegionWidth):
    """
    Create the temporal memory region.
    
    @return             (int)            Region width to input to next region.
    """
    # Make sure region widths fit
    if self.tmParams["columnCount"] != prevRegionWidth:
      raise ValueError("Region widths do not fit.")
    self.tmParams["inputWidth"] = self.tmParams["columnCount"]
    
    # Create the TM region
    self.network.addRegion("TM", "py.TPRegion", json.dumps(self.tmParams))
    self.temporalMemoryRegion = self.network.regions["TM"]

    # Feed forward link from SP to TM
    self.network.link("SP", "TM", "UniformLink", "",
        srcOutput="bottomUpOut", destInput="bottomUpIn")
    
    # Feedback links (unnecessary ?)
    self.network.link("TM", "SP", "UniformLink", "",
        srcOutput="topDownOut", destInput="topDownIn")
    self.network.link("TM", "sensor", "UniformLink", "",
        srcOutput="topDownOut", destInput="temporalTopDownIn")

    # Forward the sensor region sequence reset to the TM
    self.network.link("sensor", "TM", "UniformLink", "",
        srcOutput="resetOut", destInput="resetIn")

    # Make sure learning is enabled (this is the default)
    self.temporalMemoryRegion.setParameter("learningMode", False)
    
    # Inference mode outputs the current inference (i.e. active cells).
    # Okay to always leave inference mode on - only there for some corner cases.
    self.temporalMemoryRegion.setParameter("inferenceMode", True)

    return self.tmParams["inputWidth"]


  def _initClassifierRegion(self):
    """
    Create classifier region.
    """
    # Create the classifier region.
    self.network.addRegion(
      "classifier", "py.CLAClassifierRegion", json.dumps(self.classifierParams))
    self.classifierRegion = self.network.regions["classifier"]

    # Feed the TM states to the classifier.
    self.network.link("TM", "classifier", "UniformLink", "",
        srcOutput = "bottomUpOut", destInput = "bottomUpIn")
    
    
    # Link the sensor to the classifier to send in category labels.
    # TODO: this link is actually useless right now because the CLAclassifier region compute() function doesn't work
    # and that we are feeding TM states & categories manually to the classifier via the customCompute() function. 
    self.network.link("sensor", "classifier", "UniformLink", "",
        srcOutput = "categoryOut", destInput = "categoryIn")

    # Disable learning for now (will be enabled in a later training phase).
    self.classifierRegion.setParameter("learningMode", False)

    # Inference mode outputs the current inference. We can always leave it on.
    self.classifierRegion.setParameter("inferenceMode", True)


  def _setRegions(self):
    """Init regions variables of the network instance."""
    self.sensorRegion = self.network.regions["sensor"]
    self.spatialPoolerRegion = self.network.regions["SP"]
    self.temporalMemoryRegion = self.network.regions["TM"]
    self.classifierRegion = self.network.regions["classifier"]


  def createNetwork(self, dataSource):
    """
    Create and intialize the network instance, ready to run with regions for the
    sensor, SP, TM, and classifier.

    @param dataSource   (RecordStream)  Sensor region reads data from here.
    """
    sensorRegionWidth = self._initSensorRegion(dataSource)
    
    SPRegionWidth = self._initSpatialPoolerRegion(sensorRegionWidth)

    self._initTemporalMemoryRegion(SPRegionWidth)
  
    self._initClassifierRegion()
  
    # Need to init the network before it can run.
    self.network.initialize()
    self._setRegions()
