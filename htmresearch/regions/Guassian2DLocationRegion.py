# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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

from nupic.bindings.regions.PyRegion import PyRegion

from htmresearch.algorithms.location_modules import (
  ThresholdedGaussian2DLocationModule)



class Guassian2DLocationRegion(PyRegion):
  """
  The Guassian2DLocationRegion computes the location of the sensor in the space
  of the object given sensory and motor inputs using gaussian grid cell modules
  to update the location.

  Sensory input drives the activity of the region while Motor input causes the
  region to perform path integration, updating its activity.

  The grid cell module algorithm used by this region is based on the
  :class:`ThresholdedGaussian2DLocationModule` where each module has one or more
  gaussian activity bumps that move as the population receives motor input. When
  two bumps are near each other, the intermediate cells have higher firing rates
  than they would with a single bump. The cells with firing rates above a
  certain threshold are considered "active". When the network receives a motor
  command, it shifts its bumps.

  The cells are distributed uniformly through the rhombus, packed in the optimal
  hexagonal arrangement. During learning, the cell nearest to the current phase
  is associated with the sensed feature.

  See :class:`ThresholdedGaussian2DLocationModule` for more details
  """

  @classmethod
  def getSpec(cls):
    """
    Return the spec for the Guassian2DLocationRegion
    """
    spec = dict(
      description=Guassian2DLocationRegion.__doc__,
      singleNodeOnly=True,
      inputs=dict(
        anchorInput=dict(
          description="An array of 0's and 1's representing the sensory input "
                      "during inference. This will often come from a "
                      "feature-location pair layer (L4 active cells).",
          dataType="Real32",
          count=0,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False,
        ),
        anchorGrowthCandidates=dict(
          description="An array of 0's and 1's representing the sensory input "
                      "during learning. This will often come from a "
                      "feature-location pair layer (L4 winner cells).",
          dataType="Real32",
          count=0,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False,
        ),
        displacement=dict(
          description="Pair of floats representing the displacement as 2D "
                      "translation vector [dx, dy].",
          dataType="Real32",
          count=2,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False,
        ),
        resetIn=dict(
          description="Clear all cell activity.",
          dataType="Real32",
          count=1,
          required=False,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False,
        )
      ),
      outputs=dict(
        activeCells=dict(
          description="A binary output containing a 1 for every"
                      " cell that is currently active.",
          dataType="Real32",
          count=0,
          regionLevel=True,
          isDefaultOutput=False
        ),
        learnableCells=dict(
          description="A binary output containing a 1 for every"
                      " cell that is currently learnable",
          dataType="Real32",
          count=0,
          regionLevel=True,
          isDefaultOutput=False
        ),
        sensoryAssociatedCells=dict(
          description="A binary output containing a 1 for every"
                      " cell that is currently associated with a sensory input",
          dataType="Real32",
          count=0,
          regionLevel=True,
          isDefaultOutput=False
        )
      ),
      parameters=dict(
        moduleCount=dict(
          description="Number of grid cell modules",
          dataType="UInt32",
          accessMode="Read",
          count=1
        ),
        cellsPerAxis=dict(
          description="Determines the number of cells. Determines how space is "
                      "divided between the cells",
          dataType="UInt32",
          accessMode="Read",
          count=1
        ),
        scale=dict(
          description="Determines the amount of world space covered by all of "
                      "the cells combined. In grid cell terminology, this is "
                      "equivalent to the 'scale' of a module. One scale value "
                      "for each grid cell module. Array size must match "
                      "'moduleCount' parameter",
          dataType="Real32",
          accessMode="Read",
          count=0,
        ),
        orientation=dict(
          description="The rotation of this map, measured in radians. One "
                      "orientation value for each grid cell module. Array size "
                      "must match 'moduleCount' parameter",
          dataType="Real32",
          accessMode="Read",
          count=0,
        ),
        anchorInputSize=dict(
          description="The number of input bits in the anchor input",
          dataType="UInt32",
          accessMode="Read",
          count=1,
        ),
        activeFiringRate=dict(
          description="Between 0.0 and 1.0. A cell is considered active if its "
                      "firing rate is at least this value",
          dataType="Real32",
          accessMode="Read",
          count=1,
        ),
        bumpSigma=dict(
          description="Specifies the diameter of a gaussian bump, in units of "
                      "'rhombus edges'. A single edge of the rhombus has length "
                      "1, and this bumpSigma would typically be less than 1. We "
                      "often use 0.18172 as an estimate for the sigma of a rat "
                      "entorhinal bump",
          dataType="Real32",
          accessMode="Read",
          count=1,
          defaultValue=0.18172,
        ),
        activationThreshold=dict(
          description="If the number of active connected synapses on a "
                      "segment is at least this threshold, the segment "
                      "is said to be active",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          constraints="",
          defaultValue=10
        ),
        initialPermanence=dict(
          description="Initial permanence of a new synapse",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints="",
          defaultValue=0.21
        ),
        connectedPermanence=dict(
          description="If the permanence value for a synapse is greater "
                      "than this value, it is said to be connected",
          accessMode="Read",
          dataType="Real32",
          count=1,
          constraints="",
          defaultValue=0.50
        ),
        learningThreshold=dict(
          description="Minimum overlap required for a segment to learned",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          defaultValue=10
        ),
        sampleSize=dict(
          description="The desired number of active synapses for an "
                      "active cell",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          defaultValue=20
        ),
        permanenceIncrement=dict(
          description="Amount by which permanences of synapses are "
                      "incremented during learning",
          accessMode="Read",
          dataType="Real32",
          count=1,
          defaultValue=0.1
        ),
        permanenceDecrement=dict(
          description="Amount by which permanences of synapses are "
                      "decremented during learning",
          accessMode="Read",
          dataType="Real32",
          count=1,
          defaultValue=0.0
        ),
        maxSynapsesPerSegment=dict(
          description="The maximum number of synapses per segment",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          defaultValue=-1
        ),
        bumpOverlapMethod=dict(
          description="Specifies the firing rate of a cell when it's part of "
                      "two bumps. ('probabilistic' or 'sum')",
          dataType="Byte",
          accessMode="Read",
          constraints=("enum: probabilistic, sum"),
          defaultValue="probabilistic",
          count=0,
        ),
        learningMode=dict(
          description="A boolean flag that indicates whether or not we should "
                      "learn by associating the location with the sensory "
                      "input",
          dataType="Bool",
          accessMode="ReadWrite",
          count=1,
          defaultValue=False
        ),
        seed=dict(
          description="Seed for the random number generator",
          accessMode="Read",
          dataType="UInt32",
          count=1,
          defaultValue=42
        )
      ),
      commands=dict(
        reset=dict(description="Clear all cell activity"),
        activateRandomLocation=dict(description="Set the location to a random "
                                                "point"),
      )
    )
    return spec

  def __init__(self,
               moduleCount,
               cellsPerAxis,
               scale,
               orientation,
               anchorInputSize,
               activeFiringRate,
               bumpSigma,
               activationThreshold=10,
               initialPermanence=0.21,
               connectedPermanence=0.50,
               learningThreshold=10,
               sampleSize=20,
               permanenceIncrement=0.1,
               permanenceDecrement=0.0,
               maxSynapsesPerSegment=-1,
               bumpOverlapMethod="probabilistic",
               learningMode=False,
               seed=42,
               **kwargs):
    if moduleCount <= 0 or cellsPerAxis <= 0:
      raise TypeError("Parameters moduleCount and cellsPerAxis must be > 0")
    if moduleCount != len(scale) or moduleCount != len(orientation):
      raise TypeError("scale and orientation arrays len must match moduleCount")

    self.moduleCount = moduleCount
    self.cellsPerAxis = cellsPerAxis
    self.cellCount = cellsPerAxis * cellsPerAxis
    self.scale = list(scale)
    self.orientation = list(orientation)
    self.anchorInputSize = anchorInputSize
    self.activeFiringRate = activeFiringRate
    self.bumpSigma = bumpSigma
    self.activationThreshold = activationThreshold
    self.initialPermanence = initialPermanence
    self.connectedPermanence = connectedPermanence
    self.learningThreshold = learningThreshold
    self.sampleSize = sampleSize
    self.permanenceIncrement = permanenceIncrement
    self.permanenceDecrement = permanenceDecrement
    self.maxSynapsesPerSegment = maxSynapsesPerSegment
    self.bumpOverlapMethod = bumpOverlapMethod
    self.learningMode = learningMode
    self.seed = seed

    self._modules = None

    PyRegion.__init__(self, **kwargs)

  def initialize(self):
    """ Initialize grid cell modules """

    if self._modules is None:
      self._modules = []
      for i in xrange(self.moduleCount):
        self._modules.append(ThresholdedGaussian2DLocationModule(
          cellsPerAxis=self.cellsPerAxis,
          scale=self.scale[i],
          orientation=self.orientation[i],
          anchorInputSize=self.anchorInputSize,
          activeFiringRate=self.activeFiringRate,
          bumpSigma=self.bumpSigma,
          activationThreshold=self.activationThreshold,
          initialPermanence=self.initialPermanence,
          connectedPermanence=self.connectedPermanence,
          learningThreshold=self.learningThreshold,
          sampleSize=self.sampleSize,
          permanenceIncrement=self.permanenceIncrement,
          permanenceDecrement=self.permanenceDecrement,
          maxSynapsesPerSegment=self.maxSynapsesPerSegment,
          bumpOverlapMethod=self.bumpOverlapMethod,
          seed=self.seed))

  def compute(self, inputs, outputs):
    """
    Compute the location based on the 'displacement' and 'anchorInput' by first
    applying the  movement, if 'displacement' is present in the 'input' array
    and then applying the sensation if 'anchorInput' is present in the input
    array. The 'anchorGrowthCandidates' input array is used during learning

    See :meth:`ThresholdedGaussian2DLocationModule.movementCompute` and
        :meth:`ThresholdedGaussian2DLocationModule.sensoryCompute`
    """

    if "resetIn" in inputs:
      if len(inputs['resetIn']) != 1:
        raise Exception("resetIn has invalid length")
      self.reset()
      outputs["activeCells"][:] = 0
      outputs["learnableCells"][:] = 0
      outputs["sensoryAssociatedCells"][:] = 0
      return

    displacement = inputs.get("displacement")
    anchorInput = inputs.get("anchorInput")
    if anchorInput is not None:
      anchorInput = anchorInput.nonzero()[0]

    anchorGrowthCandidates = inputs.get("anchorGrowthCandidates")
    if anchorGrowthCandidates is not None:
      anchorGrowthCandidates = anchorGrowthCandidates.nonzero()[0]

    # Concatenate the output of all modules
    outputs["activeCells"][:] = 0
    outputs["learnableCells"][:] = 0
    outputs["sensoryAssociatedCells"][:] = 0
    for i in xrange(self.moduleCount):
      module = self._modules[i]

      # Compute movement
      if displacement is not None:
        module.movementCompute(displacement)

      # Compute sensation
      if anchorInput is not None or anchorGrowthCandidates is not None:
        module.sensoryCompute(anchorInput, anchorGrowthCandidates, self.learningMode)

      # Concatenate outputs
      start = i * self.cellCount
      end = start + self.cellCount
      outputs["activeCells"][start:end][module.getActiveCells()] = 1
      outputs["learnableCells"][start:end][module.getLearnableCells()] = 1
      outputs["sensoryAssociatedCells"][start:end][module.getSensoryAssociatedCells()] = 1

  def reset(self):
    """ Clear the active cells """
    for module in self._modules:
      module.reset()

  def activateRandomLocation(self):
    """
    Set the location to a random point.
    """
    for module in self._modules:
      module.activateRandomLocation()

  def setParameter(self, parameterName, index, parameterValue):
    """
    Set the value of a Spec parameter.
    """
    spec = self.getSpec()
    if parameterName not in spec['parameters']:
      raise Exception("Unknown parameter: " + parameterName)

    setattr(self, parameterName, parameterValue)

  def getOutputElementCount(self, name):
    """
    Returns the size of the output array
    """
    if name in ["activeCells", "learnableCells", "sensoryAssociatedCells"]:
      return self.cellCount * self.moduleCount
    else:
      raise Exception("Invalid output name specified: " + name)

  def getModules(self):
    """
    Returns underlying list of modules used by this region
    """
    return self._modules
