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

import numpy
from nupic.bindings.math import GetNTAReal
#from nupic.research.union_pooler import UnionPooler
from union_pooling.union_pooler import UnionPooler
from nupic.support import getArgumentDescriptions
from nupic.regions.PyRegion import PyRegion



def _buildArgs(self=None, kwargs={}):
  """
  Get the default arguments from the function and assign as instance vars.

  Return a list of 3-tuples with (name, description, defaultValue) for each
    argument to the function.

  Assigns all arguments to the function as instance variables of UPRegion.
  If the argument was not provided, uses the default value.

  Pops any values from kwargs that go to the function.

  """
  # Get the name, description, and default value for each argument
  argTuples = getArgumentDescriptions(UnionPooler.__init__)
  argTuples = argTuples[1:]  # Remove "self"

  # Get the names of the parameters to our own constructor and remove them
  init = UPRegion.__init__
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


def _getAdditionalSpecs():
  """Build the additional specs in three groups (for the inspector)

  Use the type of the default argument to set the Spec type, defaulting
  to "Byte" for None and complex types

  Determines the union parameters based on the selected implementation.
  It defaults to UnionPooler.
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

  # Get arguments from union pooler constructors, figure out types of
  # variables and populate unionSpec.
  uArgTuples = _buildArgs()
  unionSpec = {}
  for argTuple in uArgTuples:
    d = dict(
      description=argTuple[1],
      accessMode="ReadWrite",
      dataType=getArgType(argTuple[2])[0],
      count=getArgType(argTuple[2])[1],
      constraints=getConstraints(argTuple[2]))
    unionSpec[argTuple[0]] = d

  # Add special parameters that weren't handled automatically
  # Union pooler parameters only
  unionSpec.update(dict(
    columnCount=dict(
      description="Total number of columns (coincidences).",
      accessMode="ReadWrite",
      dataType="UInt32",
      count=1,
      constraints=""),

    inputWidth=dict(
      description="Size of inputs to the UP.",
      accessMode="ReadWrite",
      dataType="UInt32",
      count=1,
      constraints=""),
    ))


  # The last group is for parameters that aren't specific to union pooler
  otherSpec = dict(
    learningMode=dict(
      description="1 if the node is learning (default 1).",
      accessMode="ReadWrite",
      dataType="bool",
      count=1,
      constraints="bool"),
  )

  return unionSpec, otherSpec



class UPRegion(PyRegion):

  """
  UPRegion is designed to implement the union pooler compute for a given
  HTM level.

  Uses the UnionPooler class to do most of the work. This node has just one
  UnionPooler instance for the enitire level and does *not* support the concept
  of "baby nodes" within it.

  Automatic parameter handling:

  Parameter names, default values, and descriptions are retrieved automatically
  from UnionPooler. Thus, there are only a few hardcoded arguments in __init__,
  and the rest are passed to the appropriate underlying class. The NodeSpec is
  mostly built automatically from these parameters, too.

  If you add a parameter to UnionPooler, it will be exposed through UPRegion
  automatically as if it were in UPRegion.__init__, with the right default
  value. Add an entry in the __init__ docstring for it too, and that will be
  brought into the NodeSpec. UPRegion will maintain the parameter as its own
  instance variable and also pass it to UnionPooler. If the parameter is
  changed, UPRegion will propagate the change.

  If you want to do something different with the parameter, add it as an
  argument into UPRegion.__init__, which will override all the default handling.
  """

  def __init__(self, columnCount, inputWidth, **kwargs):

    if columnCount <= 0 or inputWidth <=0:
      raise TypeError("Parameters columnCount and inputWidth must be > 0")
    # Pull out the union arguments automatically
    # These calls whittle down kwargs and create instance variables of UPRegion
    uArgTuples = _buildArgs(self, kwargs)

    # Make a list of automatic union arg names for later use
    self._unionArgNames = [t[0] for t in uArgTuples]

    PyRegion.__init__(self, **kwargs)

    # Defaults for all other parameters
    self.learningMode = True
    self._inputWidth = inputWidth
    self._columnCount    = columnCount


    #
    # Variables set up in initInNetwork()
    #

    # Union instance
    self._union = None

    # union pooler's bottom-up output value: hang on to this  output for
    # top-down inference and for debugging
    self._unionPoolerOutput = None


  def initialize(self, inputs, outputs):
    """
    Initialize the UnionPooler
    """

    # Zero out the union output in case it is requested
    self._unionPoolerOutput = numpy.zeros(self._columnCount,
                                          dtype=GetNTAReal())

    # Retrieve the necessary extra arguments that were handled automatically
    autoArgs = {name: getattr(self, name) for name in self._unionArgNames}
    autoArgs["inputDimensions"] = [self._inputWidth]
    autoArgs["columnDimensions"] = [self._columnCount]
    autoArgs["potentialRadius"] = self._inputWidth

    # Allocate the union pooler
    self._union = UnionPooler(**autoArgs)
  

  def compute(self, inputs, outputs):
    """
    Run one iteration of UPRegion's compute.

    The guts of the compute are contained in the UnionPooler compute() call
    """
    activeCells = numpy.zeros(self._inputWidth, dtype=GetNTAReal())
    activeCellsIndices = [int(i) for i in inputs["activeCellsIndices"]]
    activeCells[activeCellsIndices] = 1

    predictedActiveCells= numpy.zeros(self._inputWidth, dtype=GetNTAReal())
    predictedActiveCellsIndices = [int(i) for i in inputs["predictedActiveCellsIndices"]]
    predictedActiveCells[predictedActiveCellsIndices]= 1

    # Convert
    self.unionPoolerOutput = self._union.compute(activeCells,
                              predictedActiveCells, self.learningMode)

    outputs["mostActiveCells"] = self.unionPoolerOutput


  @classmethod
  def getBaseSpec(cls):
    """Return the base Spec for UPRegion.

    Doesn't include the union parameters
    """
    spec = dict(
      description=UPRegion.__doc__,
      singleNodeOnly=True,
      inputs=dict(
          activeCellsIndices=dict(
          description="Active cells indices",
          dataType="Real32",
          count=0,
          required=True,
          regionLevel=False,
          isDefaultInput=False,
          requireSplitterMap=False),

          predictedActiveCellsIndices=dict(
          description="Predicted Actived Cell Indices",
          dataType="Real32",
          count=0,
          required=True,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

      ),
      outputs=dict(
        mostActiveCells=dict(
          description="Most active cells in the Union SDR having non-zero activation",
          dataType="Real32",
          count=0,
          regionLevel=True,
          isDefaultOutput=True),
      ),

      parameters=dict(),
    )

    return spec


  @classmethod
  def getSpec(cls):
    """
    Return the Spec for UPRegion.

    The parameters collection is constructed based on the parameters specified
    by the various components (unionSpec and otherSpec)
    """
    spec = cls.getBaseSpec()
    u, o = _getAdditionalSpecs()
    spec["parameters"].update(u)
    spec["parameters"].update(o)

    return spec


  def setParameter(self, parameterName, index, parameterValue):
    """
    Set the value of a Spec parameter. Most parameters are handled
    automatically by PyRegion's parameter set mechanism. The ones that need
    special treatment are explicitly handled here.
    """
    if parameterName in self._unionArgNames:
      setattr(self._union, parameterName, parameterValue)
    elif hasattr(self, parameterName):
      setattr(self, parameterName, parameterValue)
    else:
      raise Exception("Unknown parameter: " + parameterName)


  def __getstate__(self):
    """
    Return serializable state.  
    """
    return self.__dict__.copy()


  def __setstate__(self, state):
    """
    Set the state of ourself from a serialized state.
    """

    self.__dict__.update(state)


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
