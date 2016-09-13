# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015-2016, Numenta, Inc.  Unless you have an agreement
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

from nupic.bindings.regions.PyRegion import PyRegion

from htmresearch.algorithms.temporal_memory_factory import  createModel
from htmresearch.algorithms.extended_temporal_memory import ExtendedTemporalMemory
from nupic.research.monitor_mixin.temporal_memory_monitor_mixin import (
  TemporalMemoryMonitorMixin)


class MonitoredExtendedTemporalMemory(TemporalMemoryMonitorMixin,
                                      ExtendedTemporalMemory): pass



class ExtendedTMRegion(PyRegion):
  """
  The ExtendedTMRegion implements temporal memory for the HTM
  network API.

  The ExtendedTMRegion's computation implementations come from the
  nupic.research class ExtendedTemporalMemory.

  The region supports external inputs and top down apical inputs.
  """

  @classmethod
  def getSpec(cls):
    """
    Return the Spec for ExtendedTMRegion.
    """
    spec = dict(
      description=ExtendedTMRegion.__doc__,
      singleNodeOnly=True,
      inputs=dict(
        feedForwardInput=dict(
          description="The primary feed-forward input to the layer, this is a"
                      " binary array containing 0's and 1's",
          dataType="Real32",
          count=0,
          required=True,
          regionLevel=True,
          isDefaultInput=True,
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

        externalInput=dict(
          description="An array of 0's and 1's representing external input"
                      " such as motor commands that are available to distal"
                      " segments",
          dataType="Real32",
          count=0,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

        apicalInput=dict(
          description="An array of 0's and 1's representing top down input."
                      " The input will be provided to apical dendrites.",
          dataType="Real32",
          count=0,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

      ),
      outputs=dict(

        feedForwardOutput=dict(
          description="The default output of ExtendedTMRegion. By "
                      " default this outputs the active cells. You can change"
                      " this dynamically using defaultOutputType parameter.",
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
          description="Number of columns in this temporal memory",
          accessMode='ReadWrite',
          dataType="UInt32",
          count=1,
          constraints=""),
        columnDimensions=dict(
          description="Number of colums in this temporal memory (vector"
                      " version).",
          dataType="Real32",
          accessMode='ReadWrite',
          count=0,
          regionLevel=True,
          isDefaultOutput=True),
        inputWidth=dict(
          description='Number of inputs to the TM.',
          accessMode='Read',
          dataType='UInt32',
          count=1,
          constraints=''),
        cellsPerColumn=dict(
          description="Number of cells per column",
          accessMode='ReadWrite',
          dataType="UInt32",
          count=1,
          constraints=""),
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
        maxSegmentsPerCell=dict(
          description="The maximum number of segments per cell",
          accessMode='ReadWrite',
          dataType="UInt32",
          count=1),
        maxSynapsesPerSegment=dict(
          description="The maximum number of synapses per segment",
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
        predictedSegmentDecrement=dict(
          description="Amount by which active permanences of synapses of "
                      "previously predicted but inactive segments are "
                      "decremented.",
          accessMode='ReadWrite',
          dataType="Real32",
          count=1),
        seed=dict(
          description="Seed for the random number generator.",
          accessMode='ReadWrite',
          dataType="UInt32",
          count=1),
        formInternalConnections=dict(
          description="Flag to determine whether to form connections "
                      "with internal cells within this temporal memory",
          accessMode="ReadWrite",
          dataType="UInt32",
          count=1,
          defaultValue=1,
          constraints="bool"),
        formInternalBasalConnections=dict(
          description="Flag to determine whether to form connections "
                      "with internal cells within this temporal memory",
          accessMode="ReadWrite",
          dataType="UInt32",
          count=1,
          defaultValue=0,
          constraints="bool"),
        learnOnOneCell=dict(
          description="If True, the winner cell for each column will be"
                      " fixed between resets.",
          accessMode="ReadWrite",
          dataType="UInt32",
          count=1,
          constraints="bool"),
        defaultOutputType=dict(
          description="Controls what type of cell output is placed into"
                      " the default output 'feedForwardOutput'",
          accessMode="ReadWrite",
          dataType="Byte",
          count=0,
          constraints="enum: active,predictive,predictedActiveCells",
          defaultValue="active"),
        implementation=dict(
          description="ETM implementation. If cpp is used, the order of steps"
                      " will be artificially reversed, to use in L4.",
          accessMode="ReadWrite",
          dataType="Byte",
          count=0,
          constraints="enum: cpp, py",
          defaultValue="py"),
        monitor=dict(
          description="If True, the temporal memory will be a monitored one",
          accessMode="ReadWrite",
          dataType="UInt32",
          count=1,
          constraints="bool"),
      ),
      commands=dict(
        reset=dict(description="Explicitly reset TM states now."),
      )
    )

    return spec


  def __init__(self,
               columnCount=2048,
               cellsPerColumn=32,
               activationThreshold=13,
               initialPermanence=0.21,
               connectedPermanence=0.50,
               minThreshold=10,
               maxNewSynapseCount=20,
               permanenceIncrement=0.10,
               permanenceDecrement=0.10,
               predictedSegmentDecrement=0.0,
               seed=42,
               learnOnOneCell=1,
               formInternalConnections=1,
               formInternalBasalConnections=1,
               defaultOutputType = "active",
               monitor=False,
               implementation="cpp",
               **kwargs):
    # Defaults for all other parameters

    self.columnCount = columnCount
    self.cellsPerColumn = cellsPerColumn
    self.activationThreshold = activationThreshold
    self.initialPermanence = initialPermanence
    self.connectedPermanence = connectedPermanence
    self.minThreshold = minThreshold
    self.maxNewSynapseCount = maxNewSynapseCount
    self.permanenceIncrement = permanenceIncrement
    self.permanenceDecrement = permanenceDecrement
    self.predictedSegmentDecrement = predictedSegmentDecrement
    self.seed = seed
    self.learnOnOneCell = bool(learnOnOneCell)
    self.learningMode = True
    self.inferenceMode = True
    self.formInternalConnections = bool(formInternalConnections)
    self.formInternalBasalConnections = bool(formInternalBasalConnections)
    self.defaultOutputType = defaultOutputType
    self.monitor = monitor
    self.implementation = implementation

    PyRegion.__init__(self, **kwargs)

    # TM instance
    self._tm = None


  def initialize(self, inputs, outputs):
    """
    Initialize the self._tm if not already initialized. We need to figure out
    the constructor parameters for each class, and send it to that constructor.
    """
    if self._tm is None:
      # Create dict of arguments we will pass to the temporal memory class
      args = copy.deepcopy(self.__dict__)
      args["columnDimensions"] = (self.columnCount,)

      # Create the TM instance
      if self.monitor:
        self._tm = createModel("extendedMixin", **args)
      elif self.implementation == "cpp":
        del args["columnCount"]
        del args["formInternalConnections"]
        del args["monitor"]
        del args["implementation"]
        del args["learningMode"]
        del args["inferenceMode"]
        del args["defaultOutputType"]
        del args["_tm"]
        self._tm = createModel("reversedExtendedCPP", **args)
      else:
        self._tm = createModel("extended", **args)

      # numpy arrays we will use for some of the outputs
      self.activeState = numpy.zeros(self._tm.numberOfCells())
      self.previouslyPredictedCells = numpy.zeros(self._tm.numberOfCells())


  def compute(self, inputs, outputs):
    """
    Run one iteration of TM's compute.

    Note that if the reset signal is True (1) we assume this iteration
    represents the *end* of a sequence. The output will contain the TM
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
        outputs["predictiveCells"][:] = 0
        outputs["predictedActiveCells"][:] = 0
        return

    activeColumns = set(numpy.where(inputs["feedForwardInput"] == 1)[0])

    if "apicalInput" in inputs:
      activeApicalCells = set(numpy.where(inputs["apicalInput"] == 1)[0])
    else:
      activeApicalCells = None

    if "externalInput" in inputs:
      activeExternalCells = set(numpy.where(inputs["externalInput"] == 1)[0])
    else:
      activeExternalCells = None

    self._tm.compute(activeColumns,
                     activeExternalCells=activeExternalCells,
                     activeApicalCells=activeApicalCells,
                     formInternalConnections=self.formInternalConnections,
                     learn=self.learningMode)

    # Compute predictedActiveCells explicitly
    self.activeState[:] = 0
    self.activeState[self._tm.getActiveCells()] = 1
    predictedActiveCells = self._tm.getPredictedActiveCells()

    self.previouslyPredictedCells[:] = 0
    self.previouslyPredictedCells[self._tm.getPredictiveCells()] = 1

    # Copy numpy values into the various outputs
    outputs["activeCells"][:] = self.activeState
    outputs["predictiveCells"][:] = self.previouslyPredictedCells

    outputs["predictedActiveCells"][:] = 0
    outputs["predictedActiveCells"][self._tm.getPredictedActiveCells()] = 1

    # Send appropriate output to feedForwardOutput
    if self.defaultOutputType == "active":
      outputs["feedForwardOutput"][:] = self.activeState
    elif self.defaultOutputType == "predictive":
      outputs["feedForwardOutput"][:] = self.previouslyPredictedCells
    elif self.defaultOutputType == "predictedActiveCells":
      outputs["feedForwardOutput"][:] = 0
      outputs["feedForwardOutput"][self._tm.getPredictedActiveCells()] = 1
    else:
      raise Exception("Unknown outputType: " + self.defaultOutputType)


  def reset(self):
    """ Reset the state of the TM """
    self.activeState[:] = 0
    self.previouslyPredictedCells[:] = 0

    if self._tm is not None:
      self._tm.reset()
      self.previouslyPredictedCells[:] = 0


  def debugPlot(self, name):
    self._tm.mmGetCellActivityPlot(activityType="activeCells",
                                   showReset=True,
                                   resetShading=0.75,
                                   title="ac-{name}".format(name=name))
    self._tm.mmGetCellActivityPlot(activityType="predictiveCells",
                                   showReset=True,
                                   resetShading=0.75,
                                   title="p1-{name}".format(name=name))
    self._tm.mmGetCellActivityPlot(activityType="predictedCells",
                                   showReset=True,
                                   resetShading=0.75,
                                   title="p2-{name}".format(name=name))
    self._tm.mmGetCellActivityPlot(activityType="predictedActiveCells",
                                   showReset=True,
                                   resetShading=0.75,
                                   title="pa-{name}".format(name=name))


  def getParameter(self, parameterName, index=-1):
    """
      Get the value of a NodeSpec parameter. Most parameters are handled
      automatically by PyRegion's parameter get mechanism. The ones that need
      special treatment are explicitly handled here.
    """
    # TODO SPECIAL CASES
    return PyRegion.getParameter(self, parameterName, index)


  def setParameter(self, parameterName, index, parameterValue):
    """
    Set the value of a Spec parameter. Most parameters are handled
    automatically by PyRegion's parameter set mechanism. The ones that need
    special treatment are explicitly handled here.
    """
    if parameterName in ["learningMode", "inferenceMode",
                         "formInternalConnections"]:
      setattr(self, parameterName, bool(parameterValue))
    elif hasattr(self, parameterName):
      setattr(self, parameterName, parameterValue)
    else:
      raise Exception("Unknown parameter: " + parameterName)


  def getOutputElementCount(self, name):
    """
    Return the number of elements for the given output.
    """
    if name in ["feedForwardOutput", "predictedActiveCells", "predictiveCells",
                "activeCells"]:
      return self.columnCount * self.cellsPerColumn
    else:
      raise Exception("Invalid output name specified")


  def prettyPrintTraces(self):
    if "mixin" in self.temporalImp.lower():
      print self._tm.mmPrettyPrintTraces([
        self._tm.mmGetTraceNumSegments(),
        self._tm.mmGetTraceNumSynapses(),
      ])
