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

import copy
import numpy

from nupic.support import getArgumentDescriptions
from nupic.bindings.regions.PyRegion import PyRegion

from htmresearch.algorithms.temporal_memory_factory import (
  createModel, getConstructorArguments)


class TMRegion(PyRegion):
  """
  The TMRegion implements temporal memory for the HTM network API.

  The TMRegion's computation implementations come from the various
  Temporal Memory classes found in nupic and nupic.research
  (TemporalMemory, ExtendedTemporalMemory).
  """

  @classmethod
  def getSpec(cls):
    """
    Return the Spec for TMRegion.

    The parameters collection is constructed based on the parameters specified
    by the various components (tmSpec and otherSpec)
    """
    spec = dict(
        description=TMRegion.__doc__,
        singleNodeOnly=True,
        inputs=dict(
            bottomUpIn=dict(
                description="The primary input to the region, this is an array"
                            " containing 0's and 1's",
                dataType="Real32",
                count=0,
                required=True,
                regionLevel=False,
                isDefaultInput=True,
                requireSplitterMap=False),
            resetIn=dict(
                description="""A boolean flag that indicates whether
                        or not the input vector received in this compute cycle
                        represents the first training presentation in a
                        new temporal sequence. The TM state will be reset before
                        this input is processed""",
                dataType='Real32',
                count=1,
                required=False,
                regionLevel=True,
                isDefaultInput=False,
                requireSplitterMap=False),
            sequenceIdIn=dict(
                description="Sequence Id, for debugging",
                dataType='Real32',
                count=1,
                required=False,
                regionLevel=True,
                isDefaultInput=False,
                requireSplitterMap=False),
            externalInput=dict(
                description="An array of 0's and 1's representing external input"
                            " such as motor commands. Use of this input"
                            " requires use of a compatible TM implementation.",
                dataType="Real32",
                count=0,
                required=True,
                regionLevel=False,
                isDefaultInput=False,
                requireSplitterMap=False),
            topDownIn=dict(
                description="An array of 0's and 1's representing top down input."
                            "The input will be provided to apical dendrites."
                            " Use of this input requires use of a compatible TM"
                            " implementation.",
                dataType="Real32",
                count=0,
                required=False,
                regionLevel=False,
                isDefaultInput=False,
                requireSplitterMap=False),
        ),
        outputs=dict(
            bottomUpOut=dict(
              description="The default output of TMRegion. By default this"
                   " outputs the active cells. You can change this dynamically"
                   " using the defaultOutputType parameter.",
              dataType="Real32",
              count=0,
              regionLevel=True,
              isDefaultOutput=True),
            predictedCells=dict(
                description="A dense binary output containing a 1 for every"
                            " cell that was predicted for this time step.",
                dataType="Real32",
                count=0,
                regionLevel=True,
                isDefaultOutput=False),
            predictiveCells=dict(
                description="A dense binary output containing a 1 for every"
                            " cell predicted for the next time step.",
                dataType="Real32",
                count=0,
                regionLevel=True,
                isDefaultOutput=False),
            predictedActiveCells=dict(
                description="A dense binary output containing a 1 for every"
                            " cell that transitioned from predicted to active.",
                dataType="Real32",
                count=0,
                regionLevel=True,
                isDefaultOutput=False),
            activeCells=dict(
                description="A dense binary output containing a 1 for every"
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
                accessMode="Read",
                dataType="UInt32",
                count=1,
                constraints=""),
            basalInputWidth=dict(
              description='Number of basal inputs to the TM.',
              accessMode='Read',
              dataType='UInt32',
              count=1,
              constraints=''),
            apicalInputWidth=dict(
              description='Number of apical inputs to the TM.',
              accessMode='Read',
              dataType='UInt32',
              count=1,
              constraints=''),
            cellsPerColumn=dict(
                description="Number of cells per column",
                accessMode="Read",
                dataType="UInt32",
                count=1,
                constraints=""),
            activationThreshold=dict(
                description="If the number of active connected synapses on a "
                            "segment is at least this threshold, the segment "
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
            minThreshold=dict(
                description="If the number of synapses active on a segment is at "
                            "least this threshold, it is selected as the best "
                            "matching cell in a bursting column.",
                accessMode="Read",
                dataType="UInt32",
                count=1,
                constraints=""),
            maxNewSynapseCount=dict(
                description="The maximum number of synapses added to a segment "
                            "during learning.",
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
            predictedSegmentDecrement=dict(
                description="Amount by which active permanences of synapses of "
                            "previously predicted but inactive segments are "
                            "decremented.",
                accessMode="Read",
                dataType="Real32",
                count=1),
            seed=dict(
                description="Seed for the random number generator.",
                accessMode="Read",
                dataType="UInt32",
                count=1),
            temporalImp=dict(
                description="Type of tm to use",
                accessMode="Read",
                dataType="Byte",
                count=0,
                constraints=("enum: tm_py, tm_cpp, monitored_tm_py, monitored_tm_cpp, "
                             "etm_py, etm_cpp, monitored_etm_py, monitored_etm_cpp")),
            formInternalBasalConnections=dict(
                description="Flag to determine whether to form connections "
                            "with internal cells within this temporal memory",
                accessMode="Read",
                dataType="UInt32",
                count=1,
                defaultValue=1,
                constraints="bool"),
            learnOnOneCell=dict(
                description="If True, the winner cell for each column will be"
                            " fixed between resets.",
                accessMode="Read",
                dataType="UInt32",
                count=1,
                constraints="bool"),
            defaultOutputType=dict(
                description="Controls what type of cell output is placed into"
                            " the default output 'bottomUpOut'",
                accessMode="ReadWrite",
                dataType="Byte",
                count=0,
                constraints="enum: active,predictive,predictedActiveCells",
                defaultValue="active"),
        ),
        commands=dict(
            reset=dict(description="Explicitly reset TM states now."),
            prettyPrintTraces=dict(description="Print monitoring info."),
            debugPlot=dict(description="Show the mixin plot..."),
        )
    )

    return spec


  def __init__(self,

               # Modified TM params
               columnCount=2048,
               basalInputWidth=0,
               apicalInputWidth=0,

               # ETM params
               cellsPerColumn=32,
               activationThreshold=13,
               initialPermanence=0.21,
               connectedPermanence=0.50,
               minThreshold=10,
               maxNewSynapseCount=20,
               permanenceIncrement=0.10,
               permanenceDecrement=0.10,
               predictedSegmentDecrement=0.0,
               formInternalBasalConnections=True,
               learnOnOneCell=False,
               maxSegmentsPerCell=255,
               maxSynapsesPerSegment=255,
               seed=42,
               checkInputs=True,

               # Region params
               defaultOutputType = "active",
               implementation="etm_cpp",
               learningMode=True,
               inferenceMode=True,
               **kwargs):

    # Modified TM params
    self.columnCount = columnCount
    self.basalInputWidth = basalInputWidth
    self.apicalInputWidth = apicalInputWidth

    # TM params
    self.cellsPerColumn = cellsPerColumn
    self.activationThreshold = activationThreshold
    self.initialPermanence = initialPermanence
    self.connectedPermanence = connectedPermanence
    self.minThreshold = minThreshold
    self.maxNewSynapseCount = maxNewSynapseCount
    self.permanenceIncrement = permanenceIncrement
    self.permanenceDecrement = permanenceDecrement
    self.predictedSegmentDecrement = predictedSegmentDecrement
    self.formInternalBasalConnections = formInternalBasalConnections
    self.learnOnOneCell = learnOnOneCell
    self.maxSegmentsPerCell = maxSegmentsPerCell
    self.maxSynapsesPerSegment = maxSynapsesPerSegment
    self.seed = seed
    self.checkInputs = checkInputs

    # Region params
    self.defaultOutputType = defaultOutputType
    self.implementation = implementation
    self.learningMode = learningMode
    self.inferenceMode = inferenceMode

    PyRegion.__init__(self, **kwargs)

    # TM instance
    self._tm = None


  def initialize(self):
    """
    Initialize the self._tm if not already initialized. We need to figure out
    the constructor parameters for each class, and send it to that constructor.
    """
    if self._tm is None:
      args = {
        "columnDimensions": (self.columnCount,),
        "cellsPerColumn": self.cellsPerColumn,
        "activationThreshold": self.activationThreshold,
        "initialPermanence": self.initialPermanence,
        "connectedPermanence": self.connectedPermanence,
        "minThreshold": self.minThreshold,
        "maxNewSynapseCount": self.maxNewSynapseCount,
        "permanenceIncrement": self.permanenceIncrement,
        "permanenceDecrement": self.permanenceDecrement,
        "predictedSegmentDecrement": self.predictedSegmentDecrement,
        "formInternalBasalConnections": self.formInternalBasalConnections,
        "learnOnOneCell": self.learnOnOneCell,
        "maxSegmentsPerCell": self.maxSegmentsPerCell,
        "maxSynapsesPerSegment": self.maxSynapsesPerSegment,
        "seed": self.seed,
        "checkInputs": self.checkInputs,
      }

      # Ensure we only pass in those args that are expected by this
      # implementation. This is important for SWIG'ified classes, such as
      # TemporalMemoryCPP, which don't take kwargs.
      expectedArgs = getConstructorArguments(self.temporalImp)[0]
      for arg in args.keys():
        if not arg in expectedArgs:
          args.pop(arg)

      # Create the TM instance.
      self._tm = createModel(self.implementation, **args)

      # Carry some information to the next time step.
      self.prevPredictiveCells = ()
      self.prevActiveExternalCells = ()
      self.prevActiveApicalCells = ()



  def compute(self, inputs, outputs):
    """
    Run one iteration of TM's compute.

    Note that if the reset signal is True (1) we assume this iteration
    represents the *end* of a sequence. The output will contain the TM
    representation to this point and any history will then be reset. The output
    at the next compute will start fresh, presumably with bursting columns.
    """

    activeColumns = inputs["bottomUpIn"].nonzero()[0]

    if "externalInput" in inputs:
      activeExternalCells = inputs["externalInput"].nonzero()[0]
    else:
      activeExternalCells = ()

    if "topDownIn" in inputs:
      activeApicalCells = inputs["topDownIn"].nonzero()[0]
    else:
      activeApicalCells = ()

    # Figure out if our class is one of the "extended types"
    args = getArgumentDescriptions(self._tm.compute)
    if len(args) > 3:
      # Extended temporal memory
      self._tm.compute(activeColumns,
                       activeCellsExternalBasal=activeExternalCells,
                       activeCellsExternalApical=activeApicalCells,
                       reinforceCandidatesExternalBasal=self.prevActiveExternalCells,
                       reinforceCandidatesExternalApical=self.prevActiveApicalCells,
                       growthCandidatesExternalBasal=self.prevActiveExternalCells,
                       growthCandidatesExternalApical=self.prevActiveApicalCells,
                       learn=self.learningMode)
      self.prevActiveExternalCells = activeExternalCells
      self.prevActiveApicalCells = activeApicalCells
    else:
      # Plain old temporal memory
      self._tm.compute(activeColumns, learn=self.learningMode)

    # Extract the active / predictive cells and put them into binary arrays.
    outputs["activeCells"][:] = 0
    outputs["activeCells"][self._tm.getActiveCells()] = 1
    outputs["predictedCells"][:] = 0
    outputs["predictedCells"][self.prevPredictiveCells] = 1
    outputs["predictedActiveCells"][:] = (outputs["activeCells"] *
                                          outputs["predictedActiveCells"])

    predictiveCells = self._tm.getPredictiveCells()
    outputs["predictiveCells"][:] = 0
    outputs["predictiveCells"][predictiveCells] = 0
    self.prevPredictiveCells = predictiveCells

    # Select appropriate output for bottomUpOut
    if self.defaultOutputType == "active":
      outputs["bottomUpOut"][:] = outputs["activeCells"]
    elif self.defaultOutputType == "predictive":
      outputs["bottomUpOut"][:] = outputs["predictiveCells"]
    elif self.defaultOutputType == "predictedActiveCells":
      outputs["bottomUpOut"][:] = outputs["predictedActiveCells"]
    else:
      raise Exception("Unknown outputType: " + self.defaultOutputType)

    # Handle reset after current input has been processed
    if "resetIn" in inputs:
      assert len(inputs["resetIn"]) == 1
      if inputs["resetIn"][0] != 0:
        self.reset()


  def reset(self):
    """ Reset the state of the TM """
    if self._tm is not None:
      self._tm.reset()
      self.prevPredictiveCells = ()


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


  def getOutputElementCount(self, name):
    """
    Return the number of elements for the given output.
    """
    if name in ["bottomUpOut", "predictedActiveCells", "predictiveCells",
                "activeCells"]:
      return self._columnCount * self._cellsPerColumn
    else:
      raise Exception("Invalid output name specified")


  def prettyPrintTraces(self):
    if "mixin" in self.temporalImp.lower():
      print self._tm.mmPrettyPrintTraces([
        self._tm.mmGetTraceNumSegments(),
        self._tm.mmGetTraceNumSynapses(),
      ])
