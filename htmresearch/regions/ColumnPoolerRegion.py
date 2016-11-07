# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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

import copy
import numpy
import inspect

from nupic.bindings.regions.PyRegion import PyRegion
from htmresearch.algorithms.column_pooler import ColumnPooler


def getConstructorArguments():
  """
  Return constructor argument associated with ColumnPooler.
  @return defaults (list)   a list of args and default values for each argument
  """
  argspec = inspect.getargspec(ColumnPooler.__init__)
  return argspec.args[1:], argspec.defaults


class ColumnPoolerRegion(PyRegion):
  """
  The ColumnPoolerRegion implements an L2 layer within a single cortical column / cortical
  module.

  The layer supports feed forward (proximal) and lateral inputs.
  """

  @classmethod
  def getSpec(cls):
    """
    Return the Spec for ColumnPoolerRegion.

    The parameters collection is constructed based on the parameters specified
    by the various components (tmSpec and otherSpec)
    """
    spec = dict(
      description=ColumnPoolerRegion.__doc__,
      singleNodeOnly=True,
      inputs=dict(
        feedforwardInput=dict(
          description="The primary feed-forward input to the layer, this is a"
                      " binary array containing 0's and 1's",
          dataType="Real32",
          count=0,
          required=True,
          regionLevel=True,
          isDefaultInput=True,
          requireSplitterMap=False),

        lateralInput=dict(
          description="Lateral binary input into this column, presumably from"
                      " other neighboring columns.",
          dataType="Real32",
          count=0,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

        resetIn=dict(
          description="A boolean flag that indicates whether"
                      " or not the input vector received in this compute cycle"
                      " represents the first presentation in a"
                      " new temporal sequence.",
          dataType='Real32',
          count=1,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

      ),
      outputs=dict(
        feedForwardOutput=dict(
          description="The default output of ColumnPoolerRegion. By default this"
                      " outputs the active cells. You can change this "
                      " dynamically using the defaultOutputType parameter.",
          dataType="Real32",
          count=0,
          regionLevel=True,
          isDefaultOutput=True),

        activeCells=dict(
          description="A binary output containing a 1 for every"
                      " cell that is currently active.",
          dataType="Real32",
          count=0,
          regionLevel=True,
          isDefaultOutput=False),

      ),
      parameters=dict(
        learningMode=dict(
          description="Whether the node is learning (default True).",
          accessMode="ReadWrite",
          dataType="Bool",
          count=1,
          defaultValue="true"),
        cellCount=dict(
          description="Number of cells in this layer",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        inputWidth=dict(
          description='Number of inputs to the layer.',
          accessMode='Read',
          dataType='UInt32',
          count=1,
          constraints=''),
        numOtherCorticalColumns=dict(
          description="The number of lateral inputs that this L2 will receive. "
                      "This region assumes that every lateral input is of size "
                      "'cellCount'.",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        numActiveColumnsPerInhArea=dict(
          description="The number of active cells invoked per object",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        lateralConnectionsImpl=dict(
          description="'PairwiseSegments' or 'TwoSegmentsPerCell'",
          accessMode="Read",
          dataType="Byte",
          count=0,
          constraints=""),

        #
        # Proximal
        #
        synPermProximalInc=dict(
          description="Amount by which permanences of proximal synapses are "
                      "incremented during learning.",
          accessMode="Read",
          dataType="Real32",
          count=1),
        synPermProximalDec=dict(
          description="Amount by which permanences of proximal synapses are "
                      "decremented during learning.",
          accessMode="Read",
          dataType="Real32",
          count=1),
        initialProximalPermanence=dict(
          description="Initial permanence of a new proximal synapse.",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),
        sampleSizeProximal=dict(
          description="The desired number of active synapses for an active cell",
          accessMode="Read",
          dataType="Int32",
          count=1),
        minThresholdProximal=dict(
          description="If the number of synapses active on a proximal segment "
                      "is at least this threshold, it is considered as a "
                      "candidate active cell",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        connectedPermanenceProximal=dict(
          description="If the permanence value for a synapse is greater "
                      "than this value, it is said to be connected.",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),

        #
        # Distal
        #
        synPermDistalInc=dict(
          description="Amount by which permanences of synapses are "
                      "incremented during learning.",
          accessMode="Read",
          dataType="Real32",
          count=1),
        synPermDistalDec=dict(
          description="Amount by which permanences of synapses are "
                      "decremented during learning.",
          accessMode="Read",
          dataType="Real32",
          count=1),
        initialDistalPermanence=dict(
          description="Initial permanence of a new synapse.",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),
        sampleSizeDistal=dict(
          description="The desired number of active synapses for an active "
                      "segment.",
          accessMode="Read",
          dataType="Int32",
          count=1),
        minThresholdDistal=dict(
          description="If the number of synapses active on a distal segment is "
                      "at least this threshold, the segment is considered "
                      "active",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        connectedPermanenceDistal=dict(
          description="If the permanence value for a synapse is greater "
                      "than this value, it is said to be connected.",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),


        seed=dict(
          description="Seed for the random number generator.",
          accessMode="Read",
          dataType="UInt32",
          count=1),
        defaultOutputType=dict(
          description="Controls what type of cell output is placed into"
                      " the default output 'feedForwardOutput'",
          accessMode="ReadWrite",
          dataType="Byte",
          count=0,
          constraints="enum: active,predicted,predictedActiveCells",
          defaultValue="active"),
      ),
      commands=dict(
        reset=dict(description="Explicitly reset TM states now."),
      )
    )

    return spec


  def __init__(self,
               cellCount=4096,
               inputWidth=16384,
               numOtherCorticalColumns=0,
               numActiveColumnsPerInhArea=40,

               lateralConnectionsImpl="TwoSegmentsPerCell",

               # Proximal
               synPermProximalInc=0.1,
               synPermProximalDec=0.001,
               initialProximalPermanence=0.6,
               sampleSizeProximal=20,
               minThresholdProximal=1,
               connectedPermanenceProximal=0.50,

               # Distal
               synPermDistalInc=0.10,
               synPermDistalDec=0.10,
               initialDistalPermanence=0.21,
               sampleSizeDistal=20,
               minThresholdDistal=13,
               connectedPermanenceDistal=0.50,

               seed=42,
               defaultOutputType = "active",
               **kwargs):

    # Used to derive Column Pooler params
    self.numOtherCorticalColumns = numOtherCorticalColumns

    # Column Pooler params
    self.inputWidth = inputWidth
    self.cellCount = cellCount
    self.numActiveColumnsPerInhArea = numActiveColumnsPerInhArea
    self.lateralConnectionsImpl = lateralConnectionsImpl
    self.synPermProximalInc = synPermProximalInc
    self.synPermProximalDec = synPermProximalDec
    self.initialProximalPermanence = initialProximalPermanence
    self.sampleSizeProximal = sampleSizeProximal
    self.minThresholdProximal = minThresholdProximal
    self.connectedPermanenceProximal = connectedPermanenceProximal
    self.synPermDistalInc = synPermDistalInc
    self.synPermDistalDec = synPermDistalDec
    self.initialDistalPermanence = initialDistalPermanence
    self.sampleSizeDistal = sampleSizeDistal
    self.minThresholdDistal = minThresholdDistal
    self.connectedPermanenceDistal = connectedPermanenceDistal
    self.seed = seed

    # Region params
    self.learningMode = True
    self.defaultOutputType = defaultOutputType

    self._pooler = None

    PyRegion.__init__(self, **kwargs)


  def initialize(self, inputs, outputs):
    """
    Initialize the internal objects.
    """
    if self._pooler is None:
      params = {
        "inputWidth": self.inputWidth,
        "lateralInputWidths": [self.cellCount] * self.numOtherCorticalColumns,
        "cellCount": self.cellCount,
        "numActiveColumnsPerInhArea": self.numActiveColumnsPerInhArea,
        "lateralConnectionsImpl": self.lateralConnectionsImpl,
        "synPermProximalInc": self.synPermProximalInc,
        "synPermProximalDec": self.synPermProximalDec,
        "initialProximalPermanence": self.initialProximalPermanence,
        "minThresholdProximal": self.minThresholdProximal,
        "sampleSizeProximal": self.sampleSizeProximal,
        "connectedPermanenceProximal": self.connectedPermanenceProximal,
        "synPermDistalInc": self.synPermDistalInc,
        "synPermDistalDec": self.synPermDistalDec,
        "initialDistalPermanence": self.initialDistalPermanence,
        "minThresholdDistal": self.minThresholdDistal,
        "sampleSizeDistal": self.sampleSizeDistal,
        "connectedPermanenceDistal": self.connectedPermanenceDistal,
        "seed": self.seed,
      }
      self._pooler = ColumnPooler(**params)


  def compute(self, inputs, outputs):
    """
    Run one iteration of compute.

    Note that if the reset signal is True (1) we assume this iteration
    represents the *end* of a sequence. The output will contain the
    representation to this point and any history will then be reset. The output
    at the next compute will start fresh, presumably with bursting columns.
    """
    # Handle reset first (should be sent with an empty signal)
    if "resetIn" in inputs:
      assert len(inputs["resetIn"]) == 1
      if inputs["resetIn"][0] != 0:
        # send empty output
        self.reset()
        outputs["feedForwardOutput"][:] = 0
        outputs["activeCells"][:] = 0
        return

    feedforwardInput = inputs["feedforwardInput"].nonzero()[0]

    if "lateralInput" in inputs:
      if self.lateralConnectionsImpl == "PairwiseSegments":
        lateralInputs = tuple(singleInput.nonzero()[0]
                              for singleInput
                              in numpy.split(inputs["lateralInput"],
                                             self.numOtherCorticalColumns))
      elif self.lateralConnectionsImpl == "TwoSegmentsPerCell":
        lateralInputs = inputs["lateralInput"].nonzero()[0];
      else:
        raise ValueError("Unrecognized lateralConnectionsImpl",
                         self.lateralConnectionsImpl)
    else:
      lateralInputs = ()

    # Send the inputs into the Column Pooler.
    self._pooler.compute(feedforwardInput, lateralInputs,
                         learn=self.learningMode)

    # Extract the active / predicted cells and put them into binary arrays.
    outputs["activeCells"][:] = 0
    outputs["activeCells"][self._pooler.getActiveCells()] = 1

    # Send appropriate output to feedForwardOutput.
    if self.defaultOutputType == "active":
      outputs["feedForwardOutput"][:] = outputs["activeCells"]
    else:
      raise Exception("Unknown outputType: " + self.defaultOutputType)


  def reset(self):
    """ Reset the state of the layer"""
    if self._pooler is not None:
      self._pooler.reset()


  def getParameter(self, parameterName, index=-1):
    """
    Get the value of a NodeSpec parameter. Most parameters are handled
    automatically by PyRegion's parameter get mechanism. The ones that need
    special treatment are explicitly handled here.
    """
    return PyRegion.getParameter(self, parameterName, index)


  def setParameter(self, parameterName, index, parameterValue):
    """
    Set the value of a Spec parameter.
    """
    if hasattr(self, parameterName):
      setattr(self, parameterName, parameterValue)
    else:
      raise Exception("Unknown parameter: " + parameterName)


  def getOutputElementCount(self, name):
    """
    Return the number of elements for the given output.
    """
    if name in ["feedForwardOutput", "activeCells"]:
      return self.cellCount
    else:
      raise Exception("Invalid output name specified: " + name)
