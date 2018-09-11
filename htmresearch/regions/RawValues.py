# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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



class RawValues(PyRegion):
  """
  RawDate is a simple region used to send raw scalar values into networks.

  It accepts data using the command "addDataToQueue" or through the function
  addDataToQueue() which can be called directly from Python. Data is queued up
  in a FIFO and each call to compute pops the top element.

  Each data record consists of list of floats and a 0/1 reset flag.
  """

  def __init__(self, outputWidth=1):
    self.outputWidth = outputWidth
    self.queue = deque()

  @classmethod
  def getSpec(cls):
    spec = dict(
      description=RawValues.__doc__,
      singleNodeOnly=True,
      outputs=dict(
        dataOut=dict(
          description="List of floats",
          dataType="Real32",
          count=0,
          regionLevel=True,
          isDefaultOutput=True
        ),
        resetOut=dict(
          description="Reset flag",
          dataType="Bool",
          count=1,
          regionLevel=True,
          isDefaultOutput=False
        ),
      ),
      inputs=dict(),
      parameters=dict(
        outputWidth=dict(
          description="Size of output data",
          dataType="UInt32",
          accessMode="ReadWrite",
          count=1,
          defaultValue=1,
        )
      ),
      commands=dict(
        addDataToQueue=dict(description="Add data to region. Each data record "
                                        "consists of list of and a reset flag")
      )
    )
    return spec

  def compute(self, inputs, outputs):
    """
    Get the next record from the queue and outputs it.
    """
    if len(self.queue) > 0:
      # Take the top element of the data queue
      data = self.queue.pop()
      # Copy data into output vectors
      outputs["resetOut"][0] = data["reset"]
      outputs["dataOut"][:] = data["dataOut"]

  def addDataToQueue(self, displacement, reset=False):
    """
    Add the given displacement to the region's internal queue. Calls to compute
    will cause items in the queue to be dequeued in FIFO order.

    :param displacement: Two floats representing translation vector [dx, dy] to
                         be passed to the linked regions via 'dataOut'
    :type displacement: list
    :param reset: Reset flag to be passed to the linked regions via 'resetOut'
    :type reset: bool
    """
    self.queue.appendleft({
      "dataOut": list(displacement),
      "reset": bool(reset)
    })

  def getOutputElementCount(self, name):
    if name == "resetOut":
      return 1

    elif name == "dataOut":
      return self.outputWidth

    else:
      raise Exception("Unknown output {}.".format(name))

  def initialize(self):
    pass
