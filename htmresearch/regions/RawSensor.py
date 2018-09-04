# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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

from collections import deque
from nupic.bindings.regions.PyRegion import PyRegion


class RawSensor(PyRegion):
  """
  RawSensor is a simple sensor for sending sparse data into networks.

  It accepts data using the command "addDataToQueue" or through the function
  addDataToQueue() which can be called directly from Python. Data is queued up
  in a FIFO and each call to compute pops the top element.

  Each data record consists of the non-zero indices of the sparse vector,
  a 0/1 reset flag, and an integer sequence ID.
  """

  def __init__(self,
               outputWidth=2048,
               verbosity=0):
    """Create an instance with the appropriate output size."""
    self.verbosity = verbosity
    self.outputWidth = outputWidth
    self.queue = deque()


  @classmethod
  def getSpec(cls):
    """Return base spec for this region. See base class method for more info."""
    spec = {
      "description":"Sensor for sending sparse data to an HTM network.",
      "singleNodeOnly":True,
      "outputs":{
        "dataOut":{
          "description":"Encoded text",
          "dataType":"UInt32",
          "count":0,
          "regionLevel":True,
          "isDefaultOutput":True,
          "sparse": True,
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
        "verbosity":{
          "description":"Verbosity level",
          "dataType":"UInt32",
          "accessMode":"ReadWrite",
          "count":1,
          "constraints":"",
        },
        "outputWidth":{
          "description":"Size of output vector",
          "dataType":"UInt32",
          "accessMode":"ReadWrite",
          "count":1,
          "defaultValue": 2048,
          "constraints":"",
        },
      },
      "commands":{
        "addDataToQueue": {
          "description": "Add data",
        }
      },
    }

    return spec


  def compute(self, inputs, outputs):
    """
    Get the next record from the queue and encode it. The fields for inputs and
    outputs are as defined in the spec above.
    """
    if len(self.queue) > 0:
      # Take the top element of the data queue
      data = self.queue.pop()

    else:
      raise Exception("RawSensor: No data to encode: queue is empty ")

    # Copy data into output vectors
    outputs["resetOut"][0] = data["reset"]
    outputs["sequenceIdOut"][0] = data["sequenceId"]
    self.setSparseOutput(outputs, "dataOut", data["nonZeros"])

    if self.verbosity > 1:
      print "RawSensor outputs:"
      print "sequenceIdOut: ", outputs["sequenceIdOut"]
      print "resetOut: ", outputs["resetOut"]
      print "dataOut: ", outputs["dataOut"]


  def addDataToQueue(self, nonZeros, reset, sequenceId):
    """
    Add the given data item to the sensor's internal queue. Calls to compute
    will cause items in the queue to be dequeued in FIFO order.

    @param nonZeros   A list of the non-zero elements corresponding
                      to the sparse output. This list can be specified in two
                      ways, as a python list of integers or as a string which
                      can evaluate to a python list of integers.
    @param reset      An int or string that is 0 or 1. resetOut will be set to
                      this value when this item is computed.
    @param sequenceId An int or string with an integer ID associated with this
                      token and its sequence (document).
    """
    if type(nonZeros) == type(""):
      nonZeroList = eval(nonZeros)
    elif type(nonZeros) == type([]):
      nonZeroList = nonZeros
    else:
      raise Exception("RawSensor.addDataToQueue: unknown type for nonZeros")

    self.queue.appendleft({
      "sequenceId": int(sequenceId),
      "reset": int(reset),
      "nonZeros": sorted(nonZeroList),
    })


  def addResetToQueue(self, sequenceId):
    """
    Add a reset signal to the sensor's internal queue. Calls to compute
    will cause items in the queue to be dequeued in FIFO order.

    @param sequenceId An int or string with an integer ID associated with this
                      token and its sequence (document).
    """
    self.queue.appendleft({
      "sequenceId": int(sequenceId),
      "reset": 1,
      "nonZeros": [],
    })


  def getOutputElementCount(self, name):
    """Returns the width of dataOut."""

    if name == "resetOut" or name == "sequenceIdOut":
      # Should never actually be called since output size is specified in spec
      return 1

    elif name == "dataOut":
      return self.outputWidth

    else:
      raise Exception("Unknown output {}.".format(name))


  def initialize(self):
    """ Initialize the Region - nothing to do here. """
    pass

