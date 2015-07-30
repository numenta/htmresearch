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
from operator import mul
import numpy

from nupic.bindings.math import GetNTAReal
from nupic.support import getArgumentDescriptions
from nupic.regions.PyRegion import PyRegion

from nupic.research.monitor_mixin.monitor_mixin_base import MonitorMixinBase
from sensorimotor.general_temporal_memory import GeneralTemporalMemory
from sensorimotor.fast_general_temporal_memory import FastGeneralTemporalMemory
from nupic.research.monitor_mixin.temporal_memory_monitor_mixin import (
  TemporalMemoryMonitorMixin)
class MonitoredFastGeneralTemporalMemory(TemporalMemoryMonitorMixin,
                                         FastGeneralTemporalMemory): pass
class MonitoredGeneralTemporalMemory(TemporalMemoryMonitorMixin,
                                      GeneralTemporalMemory): pass



def getDefaultTMImp():
  """
  Return the default temporal memory implementation for this region.
  """
  return "fast"



def getTMClass(tmImp):
  """ Return the class corresponding to the given spatialImp string
  """

  if tmImp == "general":
    return GeneralTemporalMemory
  elif tmImp == "fast":
    return FastGeneralTemporalMemory
  elif tmImp == "generalMonitored":
    return MonitoredGeneralTemporalMemory
  elif tmImp == "fastMonitored":
    raise NotImplementedError
  else:
    raise RuntimeError("Invalid temporal memory implementation '{imp}'. "
                       "Legal values are: 'general' and 'fast'"
                       .format(imp=tmImp))



def _buildArgs(tmClass, self=None, kwargs={}):
  """
  Get the default arguments from the function and assign as instance vars.

  Return a list of 3-tuples with (name, description, defaultValue) for each
    argument to the function.

  Assigns all arguments to the function as instance variables of TMRegion.
  If the argument was not provided, uses the default value.

  Pops any values from kwargs that go to the function.

  """
  # Get the name, description, and default value for each argument
  argTuples = getArgumentDescriptions(tmClass.__init__)
  argTuples = argTuples[1:]  # Remove "self"

  # Get the names of the parameters to our own constructor and remove them
  init = TMRegion.__init__
  ourArgNames = [t[0] for t in getArgumentDescriptions(init)]
  # Also remove a few other names that aren't in our constructor but are
  #  computed automatically
  #ourArgNames += [
  #  "inputDimensions", # TODO: CHECK IF WE NEED TO DO THIS
  #]
  for argTuple in argTuples[:]:
    if argTuple[0] in ourArgNames:
      argTuples.remove(argTuple)

  # Build the dictionary of arguments
  if self:
    for argTuple in argTuples:
      argName = argTuple[0]
      if argName in kwargs:
        # Argument was provided
        argValue = kwargs.pop(argName)
      else:
        # Argument was not provided; use the default value if there is one, and
        #  raise an exception otherwise
        if len(argTuple) == 2:
          # No default value
          raise TypeError("Must provide value for '%s'" % argName)
        argValue = argTuple[2]
      # Set as an instance variable if "self" was passed in
      setattr(self, argName, argValue)

  return argTuples


def _getAdditionalSpecs(tmImp):
  """Build the additional specs in three groups (for the inspector)

  Use the type of the default argument to set the Spec type, defaulting
  to "Byte" for None and complex types

  Determines the tm parameters based on the selected implementation.
  """
  typeNames = {int: "UInt32", float: "Real32", str: "Byte", bool: "bool", tuple: "tuple"}

  def getArgType(arg):
    t = typeNames.get(type(arg), "Byte")
    count = 0 if t == "Byte" else 1
    if t == "tuple":
      t = typeNames.get(type(arg[0]), "Byte")
      count = len(arg)
    if t == "bool":
      t = "UInt32"
    return (t, count)

  def getConstraints(arg):
    t = typeNames.get(type(arg), "Byte")
    if t == "Byte":
      return "multiple"
    elif t == "bool":
      return "bool"
    else:
      return ""

  # Get arguments from tm constructors, figure out types of
  # variables and populate tmSpec.
  TMClass = getTMClass(tmImp)
  tmArgTuples = _buildArgs(TMClass)
  tmSpec = {}
  for argTuple in tmArgTuples:
    d = dict(
      description=argTuple[1],
      accessMode="ReadWrite",
      dataType=getArgType(argTuple[2])[0],
      count=getArgType(argTuple[2])[1],
      constraints=getConstraints(argTuple[2]))
    tmSpec[argTuple[0]] = d

  #TODO Take a look at what's already in the tmSpec

  # Add special parameters that weren't handled automatically
  # TM parameters only
  tmSpec.update(dict(
      columnDimensions=dict(
          description="Dimensions of the column space",
          accessMode='ReadWrite',
          dataType="UInt32",
          count=0,
          constraints=""),
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
          count=0),
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
          count=0),
      seed=dict(
          description="Seed for the random number generator.",
          accessMode='ReadWrite',
          dataType="UInt32",
          count=0),
      tmType=dict(
          description="Type of tm to use: general",
          accessMode="ReadWrite",
          dataType="Byte",
          count=0,
          constraints="enum: general"),
      learnOnOneCell=dict(
          description="If True, the winner cell for each column will be fixed "
                      "between resets.",
          accessMode="ReadWrite",
          dataType="UInt32",
          count=1,
          constraints="bool")))

  # The last group is for parameters that aren't specific to tm
  otherSpec = dict(
    learningMode=dict(
      description="1 if the node is learning (default 1).",
      accessMode="ReadWrite",
      dataType="UInt32",
      count=1,
      constraints="bool"),
  )

  return tmSpec, otherSpec



class TMRegion(PyRegion):

  """
  """

  def __init__(self,
               columnDimensions=(2048,),
               cellsPerColumn=32,
               activationThreshold=13,
               initialPermanence=0.21,
               connectedPermanence=0.50,
               minThreshold=10,
               maxNewSynapseCount=20,
               permanenceIncrement=0.10,
               permanenceDecrement=0.10,
               predictedSegmentDecrement = 0.0,
               seed=42,
               learnOnOneCell=False,
               tmImp=getDefaultTMImp(),
               **kwargs):
    # Pull out the pooler arguments automatically
    # These calls whittle down kwargs and create instance variables of TMRegion
    self._tmClass = getTMClass(tmImp)
    tmArgTuples = _buildArgs(self._tmClass, self, kwargs)

    # Make a list of automatic pooler arg names for later use
    self._tmArgNames = [t[0] for t in tmArgTuples]

    # Defaults for all other parameters
    self.columnDimensions = copy.deepcopy(columnDimensions)
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
    self.learnOnOneCell = learnOnOneCell
    self.learningMode = True

    PyRegion.__init__(self, **kwargs)

    # TM instance
    self._tm = None


  def initialize(self, inputs, outputs):
    """
    Initialize the self._tmClass
    """

    # Retrieve the necessary extra arguments that were handled automatically
    autoArgs = {name: getattr(self, name) for name in self._tmArgNames}

    # TODO Check what's already here

    autoArgs["learnOnOneCell"] = self.learnOnOneCell
    autoArgs["columnDimensions"] = self.columnDimensions
    autoArgs["cellsPerColumn"] = self.cellsPerColumn
    autoArgs["activationThreshold"] = self.activationThreshold
    autoArgs["initialPermanence"] = self.initialPermanence
    autoArgs["connectedPermanence"] = self.connectedPermanence
    autoArgs["minThreshold"] = self.minThreshold
    autoArgs["maxNewSynapseCount"] = self.maxNewSynapseCount
    autoArgs["permanenceIncrement"] = self.permanenceIncrement
    autoArgs["permanenceDecrement"] = self.permanenceDecrement
    autoArgs["predictedSegmentDecrement"] = self.predictedSegmentDecrement
    autoArgs["seed"] = self.seed

    # Allocate the pooler
    self._tm = self._tmClass(**autoArgs)


  def compute(self, inputs, outputs):
    """
    Run one iteration of TM's compute.

    The guts of the compute are contained in the self._tmClass compute() call
    """

    if self._tm is None:
      raise RuntimeError("Temporal memory has not been initialized")

    activeColumns = set(numpy.where(inputs["activeColumns"] == 1)[0])

    if "activeExternalCells" in inputs:
      activeExternalCells = set(numpy.where(inputs["activeColumns"] == 1)[0])
    else:
      activeExternalCells = None

    if "activeApicalCells" in inputs:
      activeApicalCells = set(numpy.where(inputs["activeColumns"] == 1)[0])
    else:
      activeApicalCells = None

    if "formInternalConnections" in inputs:
      formInternalConnections = inputs["formInternalConnections"]
    else:
      formInternalConnections = True
    self._tm.compute(activeColumns,
                     activeExternalCells=activeExternalCells,
                     activeApicalCells=activeApicalCells,
                     formInternalConnections=formInternalConnections,
                     learn=self.learningMode)

    predictedActiveCellsOutput = numpy.zeros(
      self.getOutputElementCount("predictedActiveCells"), dtype=GetNTAReal())

    activeCells = [cell.idx for cell in (self._tm.predictedActiveCells)]
    predictedActiveCellsOutput[activeCells] = 1.0

    outputs["predictedActiveCells"][:] = predictedActiveCellsOutput[:]

    # TODO: Add other outputs
    #self._tm.activeExternalCells
    #self._tm.activeApicalCells
    #self._tm.unpredictedActiveColumns
    #self._tm.predictedActiveCells
    #self._tm.activeCells
    #self._tm.winnerCells
    #self._tm.activeSegments
    #self._tm.activeApicalSegments
    #self._tm.predictiveCells
    #self._tm.chosenCellForColumn
    #self._tm.matchingSegments
    #self._tm.matchingApicalSegments
    #self._tm.matchingCells


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


  @classmethod
  def getBaseSpec(cls):
    """Return the base Spec for TMRegion.

    Doesn't include the tm parameters
    """
    spec = dict(
      description=TMRegion.__doc__,
      singleNodeOnly=True,
      inputs=dict(
          activeColumns=dict(
              description="Indices of active columns in `t`",
              dataType="Real32",
              count=0,
              required=True,
              regionLevel=False,
              isDefaultInput=True,
              requireSplitterMap=False),
          activeExternalCells=dict(
              description="Indices of active external inputs in `t`",
              dataType="Real32",
              count=0,
              required=True,
              regionLevel=False,
              isDefaultInput=False,
              requireSplitterMap=False),
          activeApicalCells=dict(
              description="Active apical cells",
              dataType="Real32",
              count=0,
              required=False,
              regionLevel=False,
              isDefaultInput=False,
              requireSplitterMap=False),
          formInternalConnections=dict(
              description="Flag to determine whether to form connections "
                          "with internal cells within this temporal memory",
              dataType="bool",
              count=0,
              required=False,
              regionLevel=True,
              isDefaultInput=False,
              requireSplitterMap=False),
      ),
      outputs=dict(
          activeCells=dict(
              description="Active cells",
              dataType="Real32",
              count=0,
              regionLevel=True,
              isDefaultOutput=True),
          predictiveCells=dict(
              description="Predictive cells",
              dataType="Real32",
              count=0,
              regionLevel=True,
              isDefaultOutput=True),
          predictedActiveCells=dict(
              description="Predicted active cells",
              dataType="Real32",
              count=0,
              regionLevel=True,
              isDefaultOutput=True),
      ),
      parameters=dict(),
      commands=dict(
          reset=dict(description='Reset the temporal memory'),
          debugPlot=dict(description='Show the mixin plot...'),
      )
    )

    return spec


  @classmethod
  def getSpec(cls):
    """
    Return the Spec for TMRegion.

    The parameters collection is constructed based on the parameters specified
    by the various components (tmSpec and otherSpec)
    """
    spec = cls.getBaseSpec()
    t, o = _getAdditionalSpecs(tmImp=getDefaultTMImp())
    spec["parameters"].update(t)
    spec["parameters"].update(o)

    return spec


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
    # TODO SPECIAL CASES
    if parameterName in self._tmArgNames:
      setattr(self._tm, parameterName, parameterValue)
    elif hasattr(self, parameterName):
      setattr(self, parameterName, parameterValue)
    else:
      raise Exception("Unknown parameter: " + parameterName)


  def getOutputElementCount(self, name):

    # TODO: Add other outputs
    if name == 'activeCells':
      return reduce(mul, self.columnDimensions, 1) * self.cellsPerColumn
    elif name == 'predictiveCells':
      return reduce(mul, self.columnDimensions, 1) * self.cellsPerColumn
    elif name == 'predictedActiveCells':
      return reduce(mul, self.columnDimensions, 1) * self.cellsPerColumn
    else:
      raise Exception("Invalid output name specified")


  def getParameterArrayCount(self, name, index):
    p = self.getParameter(name)
    if not hasattr(p, "__len__"):
      raise Exception("Attempt to access parameter '%s' as an array but it is not an array" % name)
    return len(p)


  def getParameterArray(self, name, index, a):
    p = self.getParameter(name)
    if not hasattr(p, "__len__"):
      raise Exception("Attempt to access parameter '%s' as an array but it is not an array" % name)

    if len(p) >  0:
      a[:] = p[:]
