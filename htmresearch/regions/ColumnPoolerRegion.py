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

        predictiveCells=dict(
          description="A binary output containing a 1 for every"
                      " cell currently in predicted state.",
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
          description="1 if the node is learning (default 1).",
          accessMode="ReadWrite",
          dataType="UInt32",
          count=1,
          defaultValue=1,
          constraints="bool"),
        inferenceMode=dict(
          description='1 if the node is inferring (default 1).',
          accessMode='ReadWrite',
          dataType='UInt32',
          count=1,
          defaultValue=1,
          constraints='bool'),
        columnCount=dict(
          description="Number of columns in this layer",
          accessMode='ReadWrite',
          dataType="UInt32",
          count=1,
          constraints=""),
        inputWidth=dict(
          description='Number of inputs to the layer.',
          accessMode='Read',
          dataType='UInt32',
          count=1,
          constraints=''),
        activationThreshold=dict(
          description="If the number of active connected synapses on a "
                      "segment is at least this threshold, the segment "
                      "is said to be active.",
          accessMode='ReadWrite',
          dataType="UInt32",
          count=1,
          constraints=""),
        initialPermanence=dict(
          description="Initial permanence of a new synapse.",
          accessMode='ReadWrite',
          dataType="Real32",
          count=1,
          constraints=""),
        connectedPermanence=dict(
          description="If the permanence value for a synapse is greater "
                      "than this value, it is said to be connected.",
          accessMode='ReadWrite',
          dataType="Real32",
          count=1,
          constraints=""),
        minThreshold=dict(
          description="If the number of synapses active on a segment is at "
                      "least this threshold, it is selected as the best "
                      "matching cell in a bursting column.",
          accessMode='ReadWrite',
          dataType="UInt32",
          count=1,
          constraints=""),
        maxNewSynapseCount=dict(
          description="The maximum number of synapses added to a segment "
                      "during learning.",
          accessMode='ReadWrite',
          dataType="UInt32",
          count=1),
        permanenceIncrement=dict(
          description="Amount by which permanences of synapses are "
                      "incremented during learning.",
          accessMode='ReadWrite',
          dataType="Real32",
          count=1),
        permanenceDecrement=dict(
          description="Amount by which permanences of synapses are "
                      "decremented during learning.",
          accessMode='ReadWrite',
          dataType="Real32",
          count=1),
        synPermProximalInc=dict(
          description="Amount by which permanences of proximal synapses are "
                      "incremented during learning.",
          accessMode='ReadWrite',
          dataType="Real32",
          count=1),
        synPermProximalDec=dict(
          description="Amount by which permanences of proximal synapses are "
                      "decremented during learning.",
          accessMode='ReadWrite',
          dataType="Real32",
          count=1),
        initialProximalPermanence=dict(
          description="Initial permanence of a new proximal synapse.",
          accessMode='ReadWrite',
          dataType="Real32",
          count=1,
          constraints=""),
        predictedSegmentDecrement=dict(
          description="Amount by which active permanences of synapses of "
                      "previously predicted but inactive segments are "
                      "decremented.",
          accessMode='ReadWrite',
          dataType="Real32",
          count=1),
        numActiveColumnsPerInhArea=dict(
          description="The number of active cells invoked per object",
          accessMode='ReadWrite',
          dataType="UInt32",
          count=1,
          constraints=""),
        seed=dict(
          description="Seed for the random number generator.",
          accessMode='ReadWrite',
          dataType="UInt32",
          count=1),
        defaultOutputType=dict(
          description="Controls what type of cell output is placed into"
                      " the default output 'feedForwardOutput'",
          accessMode="ReadWrite",
          dataType="Byte",
          count=0,
          constraints="enum: active,predictive,predictedActiveCells",
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
               activationThreshold=13,
               initialPermanence=0.21,
               connectedPermanence=0.50,
               minThreshold=10,
               maxNewSynapseCount=20,
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
    # Defaults for all other parameters
    self.columnCount = columnCount
    self.inputWidth = inputWidth
    self.activationThreshold = activationThreshold
    self.initialPermanence = initialPermanence
    self.connectedPermanence = connectedPermanence
    self.minThreshold = minThreshold
    self.maxNewSynapseCount = maxNewSynapseCount
    self.permanenceIncrement = permanenceIncrement
    self.permanenceDecrement = permanenceDecrement
    self.predictedSegmentDecrement = predictedSegmentDecrement
    self.synPermProximalInc = synPermProximalInc
    self.synPermProximalDec = synPermProximalDec
    self.initialProximalPermanence = initialProximalPermanence
    self.seed = seed
    self.learningMode = True
    self.inferenceMode = True
    self.defaultOutputType = defaultOutputType
    self.numActiveColumnsPerInhArea = numActiveColumnsPerInhArea

    self._pooler = None

    PyRegion.__init__(self, **kwargs)


  def initialize(self, inputs, outputs):
    """
    Initialize the internal objects.
    """
    if self._pooler is None:
      args = copy.deepcopy(self.__dict__)

      # Ensure we only pass in those args that are expected.
      expectedArgs = getConstructorArguments()[0]
      for arg in args.keys():
        if not arg in expectedArgs:
          args.pop(arg)

      self._pooler = ColumnPooler(
        columnDimensions=[self.columnCount, 1],
        maxSynapsesPerSegment = self.inputWidth,
        **args
      )

      # numpy arrays we will use for some of the outputs
      self.activeState = numpy.zeros(self._pooler.numberOfCells())
      self.previouslyPredictedCells = numpy.zeros(self._pooler.numberOfCells())


  def compute(self, inputs, outputs):
    """
    Run one iteration of compute.

    Note that if the reset signal is True (1) we assume this iteration
    represents the *end* of a sequence. The output will contain the
    representation to this point and any history will then be reset. The output
    at the next compute will start fresh, presumably with bursting columns.
    """
    feedforwardInput = inputs['feedforwardInput'].nonzero()[0]
    lateralInput = inputs.get('lateralInput', None)
    if lateralInput is not None:
      lateralInput = set(lateralInput.nonzero()[0])

    # Handle reset first (should be sent with an empty signal)
    if "resetIn" in inputs:
      assert len(inputs["resetIn"]) == 1
      if inputs["resetIn"][0] != 0:
        # send empty output
        self.reset()
        self.activeState[:] = 0
        outputs["feedForwardOutput"][:] = 0
        outputs["activeCells"][:] = 0
        outputs["predictiveCells"][:] = 0
        outputs["predictedActiveCells"][:] = 0
        return

    self._pooler.compute(
      feedforwardInput=feedforwardInput,
      activeExternalCells=lateralInput,
      learn=self.learningMode
    )

    # Compute predictedActiveCells explicitly
    self.activeState[:] = 0
    self.activeState[self._pooler.getActiveCells()] = 1
    predictedActiveCells = self.activeState * self.previouslyPredictedCells

    self.previouslyPredictedCells[:] = 0
    self.previouslyPredictedCells[self._pooler.getPredictiveCells()] = 1

    # Copy numpy values into the various outputs
    outputs["activeCells"][:] = self.activeState
    outputs["predictiveCells"][:] = self.previouslyPredictedCells
    outputs["predictedActiveCells"][:] = predictedActiveCells

    # Send appropriate output to feedForwardOutput
    if self.defaultOutputType == "active":
      outputs["feedForwardOutput"][:] = self.activeState
    elif self.defaultOutputType == "predictive":
      outputs["feedForwardOutput"][:] = self.previouslyPredictedCells
    elif self.defaultOutputType == "predictedActiveCells":
      outputs["feedForwardOutput"][:] = predictedActiveCells
    else:
      raise Exception("Unknown outputType: " + self.defaultOutputType)

    # Handle reset after current input has been processed
    # if "resetIn" in inputs:
    #   assert len(inputs["resetIn"]) == 1
    #   if inputs["resetIn"][0] != 0:
    #     self.reset()


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
    Set the value of a Spec parameter. Most parameters are handled
    automatically by PyRegion's parameter set mechanism. The ones that need
    special treatment are explicitly handled here.
    """
    if parameterName in ["learningMode", "inferenceMode"]:
      setattr(self, parameterName, bool(parameterValue))
    elif hasattr(self, parameterName):
      setattr(self, parameterName, parameterValue)
    else:
      raise Exception("Unknown parameter: " + parameterName)
    self.inputWidth = self.columnCount


  def getOutputElementCount(self, name):
    """
    Return the number of elements for the given output.
    """
    if name in ["feedForwardOutput", "predictedActiveCells", "predictiveCells",
                "activeCells"]:
      return self.columnCount
    else:
      raise Exception("Invalid output name specified")

