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

        predictedCells=dict(
          description="A binary output containing a 1 for every"
                      " cell that was predicted for this timestep.",
          dataType="Real32",
          count=0,
          regionLevel=True,
          isDefaultOutput=False),

        predictedActiveCells=dict(
          description="A binary output containing a 1 for every"
                      " cell that transitioned from predicted to active.",
          dataType="Real32",
          count=0,
          regionLevel=True,
          isDefaultOutput=False),

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
        inferenceMode=dict(
          description="Whether the node is inferring (default True).",
          accessMode='ReadWrite',
          dataType='Bool',
          count=1,
          defaultValue="true"),
        columnCount=dict(
          description="Number of columns in this layer",
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
        lateralInputWidth=dict(
          description='Number of lateral inputs to the layer.',
          accessMode='Read',
          dataType='UInt32',
          count=1,
          constraints=''),
        activationThresholdDistal=dict(
          description="If the number of active connected synapses on a "
                      "distal segment is at least this threshold, the segment "
                      "is said to be active.",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        initialPermanence=dict(
          description="Initial permanence of a new synapse.",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),
        connectedPermanence=dict(
          description="If the permanence value for a synapse is greater "
                      "than this value, it is said to be connected.",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints=""),
        minThresholdProximal=dict(
          description="If the number of synapses active on a proximal segment "
                      "is at least this threshold, it is considered as a "
                      "candidate active cell",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        minThresholdDistal=dict(
          description="If the number of synapses active on a distal segment is "
                      "at least this threshold, it is selected as the best "
                      "matching cell in a bursting column.",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        maxNewProximalSynapseCount=dict(
          description="The maximum number of synapses added to a proximal segment "
                      "at each iteration during learning.",
          accessMode="Read",
          dataType="UInt32",
          count=1),
        maxNewDistalSynapseCount=dict(
          description="The maximum number of synapses added to a distal segment "
                      "at each iteration during learning.",
          accessMode="Read",
          dataType="UInt32",
          count=1),
        maxSynapsesPerDistalSegment=dict(
          description="The maximum number of synapses on a distal segment ",
          accessMode="Read",
          dataType="UInt32",
          count=1),
        maxSynapsesPerProximalSegment=dict(
          description="The maximum number of synapses on a proximal segment ",
          accessMode="Read",
          dataType="UInt32",
          count=1),
        permanenceIncrement=dict(
          description="Amount by which permanences of synapses are "
                      "incremented during learning.",
          accessMode="Read",
          dataType="Real32",
          count=1),
        permanenceDecrement=dict(
          description="Amount by which permanences of synapses are "
                      "decremented during learning.",
          accessMode="Read",
          dataType="Real32",
          count=1),
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
        predictedSegmentDecrement=dict(
          description="Amount by which active permanences of synapses of "
                      "previously predicted but inactive segments are "
                      "decremented.",
          accessMode="Read",
          dataType="Real32",
          count=1),
        numActiveColumnsPerInhArea=dict(
          description="The number of active cells invoked per object",
          accessMode="Read",
          dataType="UInt32",
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
               columnCount=2048,
               inputWidth=16384,
               lateralInputWidth=0,
               activationThresholdDistal=13,
               initialPermanence=0.21,
               connectedPermanence=0.50,
               minThresholdProximal=1,
               minThresholdDistal=10,
               maxNewProximalSynapseCount=20,
               maxNewDistalSynapseCount=20,
               permanenceIncrement=0.10,
               permanenceDecrement=0.10,
               predictedSegmentDecrement=0.0,
               synPermProximalInc=0.1,
               synPermProximalDec=0.001,
               initialProximalPermanence = 0.6,
               seed=42,
               numActiveColumnsPerInhArea=40,
               defaultOutputType = "active",
               **kwargs):

    # Modified Column Pooler params
    self.columnCount = columnCount

    # Column Pooler params
    self.inputWidth = inputWidth
    self.lateralInputWidth = lateralInputWidth
    self.activationThresholdDistal = activationThresholdDistal
    self.initialPermanence = initialPermanence
    self.connectedPermanence = connectedPermanence
    self.minThresholdProximal = minThresholdProximal
    self.minThresholdDistal = minThresholdDistal
    self.maxNewProximalSynapseCount = maxNewProximalSynapseCount
    self.maxNewDistalSynapseCount = maxNewDistalSynapseCount
    self.permanenceIncrement = permanenceIncrement
    self.permanenceDecrement = permanenceDecrement
    self.predictedSegmentDecrement = predictedSegmentDecrement
    self.synPermProximalInc = synPermProximalInc
    self.synPermProximalDec = synPermProximalDec
    self.initialProximalPermanence = initialProximalPermanence
    self.seed = seed
    self.numActiveColumnsPerInhArea = numActiveColumnsPerInhArea
    self.maxSynapsesPerSegment = inputWidth

    # Region params
    self.learningMode = True
    self.inferenceMode = True
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
        "lateralInputWidth": self.lateralInputWidth,
        "columnDimensions": (self.columnCount,),
        "activationThresholdDistal": self.activationThresholdDistal,
        "initialPermanence": self.initialPermanence,
        "connectedPermanence": self.connectedPermanence,
        "minThresholdProximal": self.minThresholdProximal,
        "minThresholdDistal": self.minThresholdDistal,
        "maxNewProximalSynapseCount": self.maxNewProximalSynapseCount,
        "maxNewDistalSynapseCount": self.maxNewDistalSynapseCount,
        "permanenceIncrement": self.permanenceIncrement,
        "permanenceDecrement": self.permanenceDecrement,
        "predictedSegmentDecrement": self.predictedSegmentDecrement,
        "synPermProximalInc": self.synPermProximalInc,
        "synPermProximalDec": self.synPermProximalDec,
        "initialProximalPermanence": self.initialProximalPermanence,
        "seed": self.seed,
        "numActiveColumnsPerInhArea": self.numActiveColumnsPerInhArea,
        "maxSynapsesPerProximalSegment": self.inputWidth,
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
        outputs["predictedCells"][:] = 0
        outputs["predictedActiveCells"][:] = 0
        return

    feedforwardInput = set(inputs["feedforwardInput"].nonzero()[0])

    if "lateralInput" in inputs:
      lateralInput = set(inputs["lateralInput"].nonzero()[0])
    else:
      lateralInput = set()

    # Send the inputs into the Column Pooler.
    self._pooler.depolarizeCells(lateralInput,
                                 learn=self.learningMode)
    self._pooler.activateCells(
      feedforwardInput=feedforwardInput,
      reinforceCandidatesExternal=lateralInput,
      growthCandidatesExternal=lateralInput,
      learn=self.learningMode)

    # Extract the active / predicted cells and put them into binary arrays.
    outputs["activeCells"][:] = 0
    outputs["activeCells"][self._pooler.getActiveCells()] = 1
    outputs["predictedCells"][:] = 0
    outputs["predictedCells"][self._pooler.getPredictiveCells()] = 1
    outputs["predictedActiveCells"][:] = (outputs["activeCells"] *
                                          outputs["predictedCells"])

    # Send appropriate output to feedForwardOutput.
    if self.defaultOutputType == "active":
      outputs["feedForwardOutput"][:] = outputs["activeCells"]
    elif self.defaultOutputType == "predicted":
      outputs["feedForwardOutput"][:] = outputs["predictedCells"]
    elif self.defaultOutputType == "predictedActiveCells":
      outputs["feedForwardOutput"][:] = outputs["predictedActiveCells"]
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
    if name in ["feedForwardOutput", "predictedActiveCells", "predictedCells",
                "activeCells"]:
      return self.columnCount
    else:
      raise Exception("Invalid output name specified: " + name)
