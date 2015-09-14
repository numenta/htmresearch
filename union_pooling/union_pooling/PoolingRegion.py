# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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

import yaml

import numpy

from nupic.bindings.math import GetNTAReal
#from nupic.research.union_pooler import UnionPooler
from union_pooling.union_pooler import UnionPooler
from nupic.support import getArgumentDescriptions
from nupic.regions.PyRegion import PyRegion



def _getPoolerClass(name):
    if name=="union":
      return UnionPooler
    else:
      raise RuntimeError("Invalid pooling implementation %s. Valid ones are: union" % (name))


def _getDefaultPoolerClass():
    return UnionPooler


def _buildArgs(poolerClass, self=None, kwargs={}):
  """
  Get the default arguments from the function and assign as instance vars.

  Return a list of 3-tuples with (name, description, defaultValue) for each
    argument to the function.

  Assigns all arguments to the function as instance variables of PoolingRegion.
  If the argument was not provided, uses the default value.

  Pops any values from kwargs that go to the function.

  """
  # Get the name, description, and default value for each argument
  argTuples = getArgumentDescriptions(poolerClass.__init__)
  argTuples = argTuples[1:]  # Remove "self"

  # Get the names of the parameters to our own constructor and remove them
  init = PoolingRegion.__init__
  ourArgNames = [t[0] for t in getArgumentDescriptions(init)]
  # Also remove a few other names that aren't in our constructor but are
  #  computed automatically
  ourArgNames += [
    "inputDimensions",
  ]
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


def _getAdditionalSpecs(poolerClass=_getDefaultPoolerClass()):
  """Build the additional specs in three groups (for the inspector)

  Use the type of the default argument to set the Spec type, defaulting
  to "Byte" for None and complex types

  Determines the pooler parameters based on the selected implementation.
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

  # Get arguments from pooler constructors, figure out types of
  # variables and populate poolerSpec.
  pArgTuples = _buildArgs(poolerClass)
  poolerSpec = {}
  for argTuple in pArgTuples:
    d = dict(
      description=argTuple[1],
      accessMode="ReadWrite",
      dataType=getArgType(argTuple[2])[0],
      count=getArgType(argTuple[2])[1],
      constraints=getConstraints(argTuple[2]))
    poolerSpec[argTuple[0]] = d

  # Add special parameters that weren't handled automatically
  # Pooler parameters only
  poolerSpec.update(dict(
    columnCount=dict(
      description="Total number of columns (coincidences).",
      accessMode="ReadWrite",
      dataType="UInt32",
      count=1),

    inputWidth=dict(
      description="Size of inputs to the UP.",
      accessMode="ReadWrite",
      dataType="UInt32",
      count=1),

     poolerType=dict(
      description="Type of pooler to use: union",
      accessMode="ReadWrite",
      dataType="Byte",
      count=0,
      constraints="enum: union"),
     ))


  # The last group is for parameters that aren't specific to pooler
  otherSpec = dict(
    learningMode=dict(
      description="1 if the node is learning (default 1).",
      accessMode="ReadWrite",
      dataType="bool",
      count=1,
      constraints="bool"),

    spatialPoolerParams=dict(
      description="The union pooler is built on top of the spatial pooler. "
                  "To pass initialization parameters to the spatial pooler, "
                  "send them as a YAML serialized dictionary. They are then "
                  "passed during initialization to the spatial pooler. There "
                  "is currently no way to interact with these parameters after "
                  "initialization (i.e. the getParameter and setParameter "
                  "methods have no affect on the SP paramaters, and calls to "
                  "these methods through the C++ implementation of the network "
                  "API will likely result in failures.",
      accessMode="ReadWrite",
      dataType="Byte",
      count=0)
  )

  return poolerSpec, otherSpec



class PoolingRegion(PyRegion):

  """
  PoolingRegion is designed to implement the pooler compute for a given
  HTM level.

  Uses a pooler class to do most of the work. This node has just one
  pooler instance for the enitire level and does *not* support the concept
  of "baby nodes" within it.

  Automatic parameter handling:

  Parameter names, default values, and descriptions are retrieved automatically
  from pooler. Thus, there are only a few hardcoded arguments in __init__,
  and the rest are passed to the appropriate underlying pooler class. The NodeSpec is
  mostly built automatically from these parameters, too.

  If you add a parameter to pooler, it will be exposed through PoolingRegion
  automatically as if it were in PoolingRegion.__init__, with the right default
  value. Add an entry in the __init__ docstring for it too, and that will be
  brought into the NodeSpec. PoolingRegion will maintain the parameter as its own
  instance variable and also pass it to pooler. If the parameter is
  changed, PoolingRegion will propagate the change.

  If you want to do something different with the parameter, add it as an
  argument into PoolingRegion.__init__, which will override all the default handling.
  """

  def __init__(self, columnCount, inputWidth, poolerType, **kwargs):
    if columnCount <= 0 or inputWidth <=0:
      raise TypeError("Parameters columnCount and inputWidth must be > 0")
    # Pull out the pooler arguments automatically
    # These calls whittle down kwargs and create instance variables of PoolingRegion
    self._poolerClass = _getPoolerClass(poolerType)
    pArgTuples = _buildArgs(self._poolerClass, self, kwargs)

    # Make a list of automatic pooler arg names for later use
    self._poolerArgNames = [t[0] for t in pArgTuples]

    PyRegion.__init__(self, **kwargs)

    # Defaults for all other parameters
    self.learningMode = True
    self._inputWidth = inputWidth
    self._columnCount = columnCount

    # pooler instance
    self._pooler = None


  def initialize(self, inputs, outputs):
    """
    Initialize the self._poolerClass
    """

    # Retrieve the necessary extra arguments that were handled automatically
    autoArgs = {name: getattr(self, name) for name in self._poolerArgNames}
    autoArgs["spatialPoolerParams"] = yaml.load(autoArgs["spatialPoolerParams"])
    autoArgs["spatialPoolerParams"]["inputDimensions"] = [self._inputWidth]
    autoArgs["spatialPoolerParams"]["columnDimensions"] = [self._columnCount]

    # Allocate the pooler
    self._pooler = self._poolerClass(**autoArgs)


  def compute(self, inputs, outputs):
    """
    Run one iteration of PoolingRegion's compute.

    The guts of the compute are contained in the self._poolerClass compute() call
    """
    activeCells = inputs["activeCells"]
    predictedActiveCells = inputs["predictedActiveCells"]

    mostActiveCellsIndices = self._pooler.compute(activeCells, predictedActiveCells, self.learningMode)

    # Convert to SDR
    outputs["mostActiveCells"][:] = numpy.zeros(self._columnCount, dtype=GetNTAReal())
    outputs["mostActiveCells"][mostActiveCellsIndices] = 1


  def reset(self):
    """ Reset the state of the Union Pooler """
    if self._pooler is not None:
      self._pooler.reset()


  @classmethod
  def getBaseSpec(cls):
    """Return the base Spec for PoolingRegion.

    Doesn't include the pooler parameters
    """
    spec = dict(
      description=PoolingRegion.__doc__,
      singleNodeOnly=True,
      inputs=dict(
          activeCells=dict(
          description="Active cells",
          dataType="Real32",
          count=0,
          required=True,
          regionLevel=False,
          isDefaultInput=False,
          requireSplitterMap=False),

          predictedActiveCells=dict(
          description="Predicted Actived Cells",
          dataType="Real32",
          count=0,
          required=True,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

      ),
      outputs=dict(
        mostActiveCells=dict(
          description="Most active cells in the pooler SDR having non-zero activation",
          dataType="Real32",
          count=0,
          regionLevel=True,
          isDefaultOutput=True),
      ),
      parameters=dict(),
      commands=dict(
        reset=dict(description="Reset the union pooler."),
      )
    )

    return spec


  @classmethod
  def getSpec(cls):
    """
    Return the Spec for PoolingRegion.

    The parameters collection is constructed based on the parameters specified
    by the various components (poolerSpec and otherSpec)
    """
    spec = cls.getBaseSpec()
    p, o = _getAdditionalSpecs()
    spec["parameters"].update(p)
    spec["parameters"].update(o)

    return spec


  def setParameter(self, parameterName, index, parameterValue):
    """
    Set the value of a Spec parameter. Most parameters are handled
    automatically by PyRegion's parameter set mechanism. The ones that need
    special treatment are explicitly handled here.
    """
    if parameterName in self._poolerArgNames:
      setattr(self._pooler, parameterName, parameterValue)
    elif hasattr(self, parameterName):
      setattr(self, parameterName, parameterValue)
    else:
      raise Exception("Unknown parameter: " + parameterName)


  def getOutputElementCount(self, name):
    return self._columnCount


  def getParameterArrayCount(self, name, index):
    p = self.getParameter(name)
    if (not hasattr(p, "__len__")):
      raise Exception("Attempt to access parameter '%s' as an array but it is not an array" % name)
    return len(p)


  def getParameterArray(self, name, index, a):

    p = self.getParameter(name)
    if (not hasattr(p, "__len__")):
      raise Exception("Attempt to access parameter '%s' as an array but it is not an array" % name)

    if len(p) >  0:
      a[:] = p[:]
