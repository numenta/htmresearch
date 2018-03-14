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

import numpy

from collections import deque
from nupic.bindings.regions.PyRegion import PyRegion
from nupic.encoders.coordinate import CoordinateEncoder


class CoordinateSensorRegion(PyRegion):
  """
  CoordinateSensorRegion is a simple sensor for sending coordinate data into
  networks using NuPIC's CoordinateEncoder.

  It accepts data using the command "addDataToQueue" or through the function
  addDataToQueue() which can be called directly from Python. Data is queued up
  in a FIFO and each call to compute pops the top element.

  Each data record consists of the coordinate in an N-dimensional integer
  coordinate space, a 0/1 reset flag, and an integer sequence ID.
  """

  def __init__(self,
               activeBits=21,
               outputWidth=1000,
               radius=2,
               verbosity=0):
    self.verbosity = verbosity
    self.activeBits = activeBits
    self.outputWidth = outputWidth
    self.radius = radius
    self.queue = deque()
    self.encoder = CoordinateEncoder(n=self.outputWidth, w=self.activeBits,
                                     verbosity=self.verbosity)

  @classmethod
  def getSpec(cls):
    """Return base spec for this region. See base class method for more info."""
    spec = {
      "description": CoordinateSensorRegion.__doc__,
      "singleNodeOnly": True,
      "inputs": {},  # input data is added to queue via "addDataToQueue" command
      "outputs": {
        "dataOut": {
          "description": "Encoded coordinate SDR.",
          "dataType": "Real32",
          "count": 0,
          "regionLevel": True,
          "isDefaultOutput": True,
        },
        "resetOut": {
          "description": "0/1 reset flag output.",
          "dataType": "UInt32",
          "count": 1,
          "regionLevel": True,
          "isDefaultOutput": False,
        },
        "sequenceIdOut": {
          "description": "Sequence ID",
          "dataType": "UInt32",
          "count": 1,
          "regionLevel": True,
          "isDefaultOutput": False,
        },
      },
      "parameters": {
        "activeBits": {
          "description": "The number of bits that are set to encode a single "
                         "coordinate value",
          "dataType": "uint",
          "accessMode": "ReadWrite",
          "count": 1,
          "defaultValue": 21
        },
        "outputWidth": {
          "description": "Size of output vector",
          "dataType": "UInt32",
          "accessMode": "ReadWrite",
          "count": 1,
          "defaultValue": 1000
        },
        "radius": {
          "description": "Radius around 'coordinate'",
          "dataType": "UInt32",
          "accessMode": "ReadWrite",
          "count": 1,
          "defaultValue": 2
        },
        "verbosity": {
          "description": "Verbosity level",
          "dataType": "UInt32",
          "accessMode": "ReadWrite",
          "count": 1
        },
      },
      "commands": {
        "addDataToQueue": {
          "description": CoordinateSensorRegion.addDataToQueue.__doc__,
        },
        "addResetToQueue": {
          "description": CoordinateSensorRegion.addResetToQueue.__doc__,
        }
      },
    }

    return spec

  def compute(self, inputs, outputs):
    """
    Get the next record from the queue and encode it.
    @param inputs This parameter is ignored. The data comes from the queue
    @param outputs See definition in the spec above.
    """
    if len(self.queue) > 0:
      data = self.queue.pop()

    else:
      raise Exception("CoordinateSensor: No data to encode: queue is empty")

    outputs["resetOut"][0] = data["reset"]
    outputs["sequenceIdOut"][0] = data["sequenceId"]
    sdr = self.encoder.encode((numpy.array(data["coordinate"]), self.radius))
    outputs["dataOut"][:] = sdr

    if self.verbosity > 1:
      print "CoordinateSensor outputs:"
      print "Coordinate = ", data["coordinate"]
      print "sequenceIdOut: ", outputs["sequenceIdOut"]
      print "resetOut: ", outputs["resetOut"]
      print "dataOut: ", outputs["dataOut"].nonzero()[0]

  def addDataToQueue(self, coordinate, reset, sequenceId):
    """
    Add the given data item to the sensor's internal queue. Calls to compute
    will cause items in the queue to be dequeued in FIFO order.

    @param coordinate A list containing the N-dimensional integer coordinate
                      space to be encoded. This list can be specified in two
                      ways, as a python list of integers or as a string which
                      can evaluate to a python list of integers.
    @param reset      An int or string that is 0 or 1. resetOut will be set to
                      this value when this item is computed.
    @param sequenceId An int or string with an integer ID associated with this
                      token and its sequence (document).
    """
    if type(coordinate) == type(""):
      coordinateList = eval(coordinate)
    elif type(coordinate) == type([]):
      coordinateList = coordinate
    else:
      raise Exception("CoordinateSensor.addDataToQueue: unknown type for "
                      "coordinate")

    self.queue.appendleft({
      "sequenceId": int(sequenceId),
      "reset": int(reset),
      "coordinate": coordinateList,
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
      "coordinate": [],
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
