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
import numpy

from collections import deque

from nupic.bindings.regions.PyRegion import PyRegion


class LanguageSensor(PyRegion):
  """
  LanguageSensor (LS) is an extensible sensor for text data.

  The LS obtains info from a file, csv or txt (not yet implemented).

  An LS is essentially a shell containing two objects:

    1. A DataSource object gets one record at a time. This record is returned
    as a dict object by getNextRecordDict(). For example, a DataSource might
    return:
      {sample="Hello world!", labels=["Python"]}

    2. An encoder from nupic.fluent/encoders

  The DataSource and LanguageEncoder are supplied after the node is created,
  not in the node itself.

  """

  def __init__(self,
               verbosity=0,
               numCategories=1,
               cacheRoot=None):
    """Create a node without an encoder or datasource."""
    self.cacheRoot = cacheRoot
    self.numCategories = numCategories
    self.verbosity = verbosity

    # These fields are set outside when building the region.
    self.encoder = None
    self.dataSource = None

    self._outputValues = {}
    self._iterNum = 0

    self.queue = deque()


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
          "dataType":"Real32",
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
          "dataType":'Real32',
          "count":1,
          "regionLevel":True,
          "isDefaultOutput":False,
        },
      },
      "inputs":{},
      "parameters":{
        "cacheRoot":{
          "description":"Directory for caching Cio API requests.",
          "dataType":"Byte",
          "accessMode":"ReadWrite",
          "count":0,
          "constraints":"",
        },
        "verbosity":{
          "description":"Verbosity level",
          "dataType":"UInt32",
          "accessMode":"ReadWrite",
          "count":1,
          "constraints":"",
        },
        "numCategories":{
          "description":("Total number of categories to expect from the "
                         "FileRecordStream"),
          "dataType":"UInt32",
          "accessMode":"ReadWrite",
          "count":1,
          "constraints":""},
      },
      "commands":{},
    }

    return spec


  def initialize(self, inputs, outputs):
    """Initialize the node after the network is fully linked."""
    if self.encoder is None:
      raise Exception("Unable to initialize LanguageSensor -- "
                      "encoder has not been set")


  def rewind(self):
    """Reset the sensor to the beginning of the data file."""
    self._iterNum = 0
    if self.dataSource is not None:
      self.dataSource.rewind()


  def populateCategoriesOut(self, categories, output):
    """
    Populate the output array with the category indices.
    Note: non-categories are represented with -1.
    """
    if categories[0] is None:
      # The record has no entry in category field.
      output[:] = -1
    else:
      # Populate category output array by looping over the smaller of the
      # output array (size specified by numCategories) and the record's number
      # of categories.
      [numpy.put(output, [i], cat)
          for i, (_, cat) in enumerate(zip(output, categories))]
      output[len(categories):] = -1


  def compute(self, inputs, outputs):
    """
    Get a record from the dataSource and encode it. The fields for inputs and
    outputs are as defined in the spec above.

    Expects the text data to be in under header "token" from the dataSource.

    TODO: validate we're handling resets correctly
    """
    if len(self.queue) > 0:
      # data has been added to the queue, so use it
      data = self.queue.pop()

    elif self.dataSource is None:
      raise Exception("LanguageSensor: No data to encode: queue is empty "
                        "and the dataSource is None.")
    else:
      data = self.dataSource.getNextRecordDict()
      # Keys in data that are not column headers from the data source are standard
      # of RecordStreamIface objects.

    # Copy important data input fields over to outputs dict. We set "sourceOut"
    # explicitly b/c PyRegion.getSpec() won't take an output field w/ type str.
    outputs["resetOut"][0] = data["_reset"]
    outputs["sequenceIdOut"][0] = data["_sequenceId"]
    outputs["sourceOut"] = data["_token"]
    self.populateCategoriesOut(data["_category"], outputs['categoryOut'])
    outputs["encodingOut"] = self.encoder.encodeIntoArray(
      data["_token"], outputs["dataOut"])

    if self.verbosity > 0:
      print "LanguageSensor outputs:"
      print "SeqID: ", outputs["sequenceIdOut"]
      print "Categories out: ", outputs['categoryOut']
      print "dataOut: ",outputs["dataOut"].nonzero()[0]

    self._outputValues = copy.deepcopy(outputs)

    self._iterNum += 1


  def addDataToQueue(self, token, categoryList, sequenceId, reset=0):
    """
    Add the given data item to the sensor's internal queue. Calls to compute
    will cause items in the queue to be dequeued in FIFO order.

    @param token        (str)  The text token
    @param categoryList (list) A list of one or more integer labels associated
                               with this token. If the list is [None], no
                               categories will be associated with this item.
    @param sequenceId   (int)  An integer ID associated with this token and its
                               sequence (document).
    @param reset        (int)  Should be 0 or 1. resetOut will be set to this
                               value when this item is computed.


    """
    self.queue.appendleft ({
        "_token": token,
        "_category": categoryList,
        "_sequenceId": sequenceId,
        "_reset": reset
      })


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

    elif name == "categoryOut":
      return self.numCategories

    else:
      raise Exception("Unknown output {}.".format(name))
