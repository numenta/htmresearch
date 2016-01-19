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
  createModel)
from htmresearch.algorithms.general_temporal_memory import GeneralTemporalMemory
from htmresearch.algorithms.fast_general_temporal_memory import FastGeneralTemporalMemory
from nupic.research.monitor_mixin.temporal_memory_monitor_mixin import (
  TemporalMemoryMonitorMixin)



class MonitoredFastGeneralTemporalMemory(TemporalMemoryMonitorMixin,
                                         FastGeneralTemporalMemory): pass
class MonitoredGeneralTemporalMemory(TemporalMemoryMonitorMixin,
                                      GeneralTemporalMemory): pass



class TMRegion(PyRegion):
  """
  The TMRegion implements temporal memory for the HTM network API.

  The TMRegion's computation implementations come from the various
  Temporal Memory classes found in nupic and nupic.research
  (TemporalMemory, FastTemporalMemory, GeneralTemporalMemory and
  FastGeneralTemporalMemory).

  The region supports external inputs and top down inputs. If these inputs
  are specified, temporalImp must be one of the GeneralTemporalMemory
  implementations.
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
              description="""The primary output of the temporal memory.""",
              dataType="Real32",
              count=0,
              regionLevel=True,
              isDefaultOutput=True),

            predictiveCells=dict(
                description="An output containing a 1 for every cell currently"
                            " in predicted state.",
                dataType="Real32",
                count=0,
                regionLevel=True,
                isDefaultOutput=False),
            predictedActiveCells=dict(
                description="An output containing a 1 for every cell that"
                            " transitioned from predicted to active",
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
            temporalImp=dict(
                description="Type of tm to use",
                accessMode="ReadWrite",
                dataType="Byte",
                count=0,
                constraints="enum: tm,general,fast,fastGeneral,tmMixin"),
            formInternalConnections=dict(
                description="Flag to determine whether to form connections "
                            "with internal cells within this temporal memory",
                accessMode="ReadWrite",
                dataType="UInt32",
                count=1,
                defaultValue=1,
                constraints="bool"),
            learnOnOneCell=dict(
                description="If True, the winner cell for each column will be"
                            " fixed between resets.",
                accessMode="ReadWrite",
                dataType="UInt32",
                count=1,
                constraints="bool")
        ),
        commands=dict(
            reset=dict(description='Explicitly reset TM states now.'),
            debugPlot=dict(description='Show the mixin plot...'),
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
               temporalImp="fast",
               formInternalConnections = 1,
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
    self.temporalImp = temporalImp
    self.formInternalConnections = bool(formInternalConnections)
    self.previouslyPredictedCells = set()

    PyRegion.__init__(self, **kwargs)

    # TM instance
    self._tm = None


  def initialize(self, inputs, outputs):
    """
    Initialize the self._tm. We need to figure out the constructor
    parameters for each class, and send it to that constructor.
    """
    # Create dict of arguments we will pass to the temporal memory class
    args = copy.deepcopy(self.__dict__)
    args["columnDimensions"] = (self.columnCount,)

    # Allocate the tm
    self._tm = createModel(self.temporalImp, **args)


  def compute(self, inputs, outputs):
    """
    Run one iteration of TM's compute.

    The guts of the compute are contained in the self._tmClass compute() call
    """

    # Handle reset input
    if 'resetIn' in inputs:
      assert len(inputs['resetIn']) == 1
      if inputs['resetIn'][0] != 0:
        self.reset()

    activeColumns = set(numpy.where(inputs["bottomUpIn"] == 1)[0])

    if "externalInput" in inputs:
      activeExternalCells = set(numpy.where(inputs["externalInput"] == 1)[0])
    else:
      activeExternalCells = None

    if "topDownIn" in inputs:
      activeApicalCells = set(numpy.where(inputs["topDownIn"] == 1)[0])
    else:
      activeApicalCells = None

    # Figure out if our class is one of the "general types"
    args = getArgumentDescriptions(self._tm.compute)
    if len(args) > 3:
      # General temporal memory
      self._tm.compute(activeColumns,
                       activeExternalCells=activeExternalCells,
                       activeApicalCells=activeApicalCells,
                       formInternalConnections=self.formInternalConnections,
                       learn=self.learningMode)
      predictedActiveCells = self._tm.predictedActiveCells
    else:
      # Plain old temporal memory
      self._tm.compute(activeColumns, learn=self.learningMode)
      # Normal temporal memory doesn't compute predictedActiveCells
      predictedActiveCells = self._tm.activeCells & self.previouslyPredictedCells
      self.previouslyPredictedCells = self._tm.predictiveCells


    # Set the various outputs
    outputs['bottomUpOut'][:] = 0
    outputs['bottomUpOut'][list(self._tm.activeCells)] = 1

    outputs['predictiveCells'][:] = 0
    outputs['predictiveCells'][list(self._tm.predictiveCells)] = 1

    outputs['predictedActiveCells'][:] = 0
    outputs['predictedActiveCells'][list(predictedActiveCells)] = 1


  def reset(self):
    """ Reset the state of the TM """
    if self._tm is not None:
      self._tm.reset()


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
    if name in ["bottomUpOut", "predictedActiveCells", "predictiveCells"]:
      return self.columnCount * self.cellsPerColumn
    else:
      raise Exception("Invalid output name specified")
