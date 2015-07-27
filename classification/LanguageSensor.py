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

from nupic.regions.PyRegion import PyRegion



class LanguageSensor(PyRegion):
  """
  LanguageSensor (LS) is an extensible sensor for text data.

  The LS obtains info from a file, csv or txt (not yet implemented).

  An LS is essentially a shell containing two objects:

    1. A DataSource object gets one record at a time. This record is returned
    as a dict object. For example, a DataSource might return:
      defaultdict(sample="Hello world!", labels=["Python"])

    2. An encoder from nupic.fluent/encoders

  The DataSource and LanguageEncoder are supplied after the node is created,
  not in the node itself.

  """

  def __init__(self,
               verbosity=0,
               numCategories=1):
    """
    Create a node without an encoder or datasource.
    """
    self.encoder = None
    self.dataSource = None
    self._outputValues = {}

    self.numCategories = numCategories
    self.verbosity = verbosity
    self._iterNum = 0


  @classmethod
  def getSpec(cls):
    """Return base spec for this region. See base class method for more info."""
    spec = {
      "description":"Sensor that reads text data records and encodes them for "
                    "an HTM network.",
      "singleNodeOnly":True,
      "outputs":{
        "dataOut":{
          "description":"Encoded text",
          "dataType":"Real32",
          "count":0,
          "regionLevel":True,
          "isDefaultOutput":True,
          },
        "categoryOut":{
          "description":"Index of the current word's category.",
          "dataType":"Int32",
          "count":0,
          "regionLevel":True,
          "isDefaultOutput":False,
          },
        "resetOut":{
          "description":"Boolean reset output.",
          "dataType":"Real32",
          "count":1,
          "regionLevel":True,
          "isDefaultOutput":False,
          },
        "sequenceIdOut":{
          "description":"Sequence ID",
          "dataType":'UInt64',
          "count":1,
          "regionLevel":True,
          "isDefaultOutput":False,
        },
      ## commented out b/c dataType not cool w/ numpy
        # "sourceOut":{
        #   "description":"Unencoded data from the source, input to the encoder",
        #   "dataType":str,
        #   "count":0,
        #   "regionLevel":True,
        #   "isDefaultOutput":False,
        # },
      ## need these...??
        # spatialTopDownOut=dict(
        #   description="""The top-down output signal, generated from
        #                 feedback from SP""",
        #   dataType='Real32',
        #   count=0,
        #   regionLevel=True,
        #   isDefaultOutput=False),
        # temporalTopDownOut=dict(
        #   description="""The top-down output signal, generated from
        #                 feedback from TP through SP""",
        #   dataType='Real32',
        #   count=0,
        #   regionLevel=True,
        #   isDefaultOutput=False),
      },
      "inputs":{
        "spatialTopDownIn":{
          "description":"The top-down input signal, generated via feedback "
                        "from SP.",
          "dataType":"Real32",
          "count":0,
          "required":False,
          "regionLevel":True,
          "isDefaultInput":False,
          "requireSplitterMap":False,
        },
        "temporalTopDownIn":{
          "description":"The top-down input signal, generated via feedback "
                        "from TP through SP.",
          "dataType":"Real32",
          "count":0,
          "required":False,
          "regionLevel":True,
          "isDefaultInput":False,
          "requireSplitterMap":False,
        },
        "classificationTopDownIn":{
          "description":"The top-down input signal, generated via feedback "
                        "from classifier through TP through SP.",
          "dataType":"int",
          "count":0,
          "required":False,
          "regionLevel":True,
          "isDefaultInput":False,
          "requireSplitterMap":False,
        },
      },
      "parameters":{
        "verbosity":{
          "description":"Verbosity level",
          "dataType":"UInt32",
          "accessMode":"ReadWrite",
          "count":1,
          "constraints":"",
        },
      },
      "commands":{},
    }

    return spec


  def initialize(self, inputs, outputs):
    """Initialize the node after the network is fully linked."""
    if self.encoder is None:
      raise Exception("Unable to initialize LanguageSensor -- encoder has not been set")
    if self.dataSource is None:
      raise Exception("Unable to initialize LanguageSensor -- dataSource has not been set")


  def compute(self, inputs, outputs):
    """
    Get a record from the dataSource and encode it. The fields for inputs and
    outputs are as defined in the LS object's spec.

    TODO: populate self._outputValues
    TODO: validate we're handling resets correctly
    """
    data = self.dataSource.getNextRecordDict()
    # The private keys in data are standard of RecordStreamIface objects. Any
    # add'l keys are column headers from the data source.
    import pdb; pdb.set_trace()

    # Copy important data input fields over to outputs dict.
    #   NOTE: set "sourceOut" explicitly b/c PyRegion.getSpec() won't take an
    #   output field w/ type str.
    outputs["resetOut"][0] = data["_reset"]
    outputs["sequenceIdOut"][0] = data["_sequenceId"]
    outputs["categoryOut"][0] = data["_category"]
    outputs["sourceOut"] = data["token"]

    # Encode the token, where the encoding is a dict as expected in
    # nupic.fluent ClassificationModel.
    # The data key must match the datafile column header
    # NOTE: this logic differs from RecordSensor, where output is a (sparse)
    # numpy array populated in place.
    outputs["dataOut"] = self.encoder.encodeIntoArray(data["token"], output=None)

    # self._outputValues <- dict() of sample that goes to Model (??)

    self._iterNum += 1


  def getOutputValues(self, outputName):
    """Return the dictionary of output values. Note that these are normal Python
    lists, rather than numpy arrays. This is to support lists with mixed scalars
    and strings, as in the case of records with categorical variables
    """
    return self._outputValues[outputName]


  def getOutputElementCount(self, name):
    """Returns the width of dataOut."""
    if name == "resetOut" or name == "sequenceIdOut":
      print ("WARNING: getOutputElementCount should not have been called with "
            "{}.".format(name))
      return 1

    elif name == "dataOut":
      if self.encoder == None:
        raise Exception("Network requested output element count for {} on a "
                        "LanguageSensor node, but the encoder has not been set."
                        .format(name))
      return self.encoder.getWidth()

    elif (name == "sourceOut" or
          name == 'spatialTopDownOut' or
          name == 'temporalTopDownOut'):
      if self.encoder == None:
        raise Exception("Network requested output element count for {} on a "
                        "LanguageSensor node, but the encoder has not been set."
                        .format(name))
      return len(self.encoder.getDescription())

    else:
      raise Exception("Unknown output {}.".format(name))
