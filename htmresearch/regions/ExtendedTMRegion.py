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

from htmresearch.algorithms.temporal_memory_factory import createModel



class ExtendedTMRegion(PyRegion):
  """
  The ExtendedTMRegion implements temporal memory for the HTM network API.

  The ExtendedTMRegion's computation implementations come from the
  nupic.research class ExtendedTemporalMemory.

  The region supports external basal and apical inputs.

  The main difference between the ExtendedTMRegion and the TMRegion is that the
  ExtendedTMRegion uses the basal / apical input to predict cells for the
  current time step, while the TMRegion uses them to predict cells for the next
  time step. The ExtendedTMRegion can't output predictions for the next input.
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

        externalBasalInput=dict(
          description="An array of 0's and 1's representing external input"
                      " such as motor commands that are available to basal"
                      " segments",
          dataType="Real32",
          count=0,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

        externalApicalInput=dict(
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
          description="True if the node is learning (default true).",
          accessMode="ReadWrite",
          dataType="Bool",
          count=1,
          defaultValue="true"),
        inferenceMode=dict(
          description='True if the node is inferring (default true).',
          accessMode='ReadWrite',
          dataType="Bool",
          count=1,
          defaultValue="true"),
        columnCount=dict(
          description="Number of columns in this temporal memory",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints=""),
        columnDimensions=dict(
          description="Number of colums in this temporal memory (vector"
                      " version).",
          dataType="Real32",
          accessMode="Read",
          count=0,
          regionLevel=True,
          isDefaultOutput=True),
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
        maxSegmentsPerCell=dict(
          description="The maximum number of segments per cell",
          accessMode="Read",
          dataType="UInt32",
          count=1),
        maxSynapsesPerSegment=dict(
          description="The maximum number of synapses per segment",
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
        formInternalBasalConnections=dict(
          description="Flag to determine whether to form basal connections "
                      "with internal cells within this temporal memory",
          accessMode="Read",
          dataType="Bool",
          count=1,
          defaultValue="true"),
        learnOnOneCell=dict(
          description="If True, the winner cell for each column will be"
                      " fixed between resets.",
          accessMode="Read",
          dataType="Bool",
          count=1,
          defaultValue="false"),
        defaultOutputType=dict(
          description="Controls what type of cell output is placed into"
                      " the default output 'feedForwardOutput'",
          accessMode="ReadWrite",
          dataType="Byte",
          count=0,
          constraints="enum: active,predicted,predictedActiveCells",
          defaultValue="active"),
        implementation=dict(
          description="ETM implementation",
          accessMode="Read",
          dataType="Byte",
          count=0,
          constraints=("enum: etm_py, etm_cpp, monitored_etm_py, "
                       "monitored_etm_cpp"),
          defaultValue="py"),
      ),
      commands=dict(
        reset=dict(description="Explicitly reset TM states now."),
      )
    )

    return spec


  def __init__(self,

               # Modified ETM params
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


  def initialize(self, dims, splitterMaps):
    """
    Initialize the self._tm if not already initialized. We need to figure out
    the constructor parameters for each class, and send it to that constructor.
    """
    if self._tm is None:
      params = {
        "columnDimensions": (self.columnCount,),
        "basalInputDimensions": (self.basalInputWidth,),
        "apicalInputDimensions": (self.apicalInputWidth,),
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
      self._tm = createModel(self.implementation, **params)


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
        outputs["predictedCells"][:] = 0
        outputs["predictedActiveCells"][:] = 0
        return

    activeColumns = inputs["feedForwardInput"].nonzero()[0]

    if "externalBasalInput" in inputs:
      activeCellsExternalBasal = inputs["externalBasalInput"].nonzero()[0]
    else:
      activeCellsExternalBasal = ()

    if "externalApicalInput" in inputs:
      activeCellsExternalApical = inputs["externalApicalInput"].nonzero()[0]
    else:
      activeCellsExternalApical = ()

    # Run the TM for one time step.
    self._tm.depolarizeCells(
      activeCellsExternalBasal,
      activeCellsExternalApical,
      learn=self.learningMode)
    self._tm.activateCells(
      activeColumns,
      reinforceCandidatesExternalBasal=activeCellsExternalBasal,
      reinforceCandidatesExternalApical=activeCellsExternalApical,
      growthCandidatesExternalBasal=activeCellsExternalBasal,
      growthCandidatesExternalApical=activeCellsExternalApical,
      learn=self.learningMode)

    # Extract the active / predicted cells and put them into binary arrays.
    outputs["activeCells"][:] = 0
    outputs["activeCells"][self._tm.getActiveCells()] = 1
    outputs["predictedCells"][:] = 0
    outputs["predictedCells"][self._tm.getPredictiveCells()] = 1
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
    """ Reset the state of the TM """
    if self._tm is not None:
      self._tm.reset()


  def debugPlot(self, name):
    self._tm.mmGetCellActivityPlot(activityType="activeCells",
                                   showReset=True,
                                   resetShading=0.75,
                                   title="ac-{name}".format(name=name))
    self._tm.mmGetCellActivityPlot(activityType="predictedCells",
                                   showReset=True,
                                   resetShading=0.75,
                                   title="p1-{name}".format(name=name))
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
      return self.cellsPerColumn * self.columnCount
    else:
      raise Exception("Invalid output name specified")


  def prettyPrintTraces(self):
    if "mixin" in self.temporalImp.lower():
      print self._tm.mmPrettyPrintTraces([
        self._tm.mmGetTraceNumSegments(),
        self._tm.mmGetTraceNumSynapses(),
      ])
