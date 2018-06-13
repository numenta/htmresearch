# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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

"""Emulates a grid cell module"""

import math
import random
import copy

import numpy as np

from htmresearch.support import numpy_helpers as np2
from htmresearch.algorithms.multiconnections import Multiconnections
from nupic.bindings.math import SparseMatrixConnections, Random



class Superficial2DLocationModule(object):
  """
  A model of a location module. It's similar to a grid cell module, but it uses
  squares rather than triangles.

  The cells are arranged into a m*n rectangle which is tiled onto 2D space.
  Each cell represents a small rectangle in each tile.

  +------+------+------++------+------+------+
  | Cell | Cell | Cell || Cell | Cell | Cell |
  |  #1  |  #2  |  #3  ||  #1  |  #2  |  #3  |
  |      |      |      ||      |      |      |
  +--------------------++--------------------+
  | Cell | Cell | Cell || Cell | Cell | Cell |
  |  #4  |  #5  |  #6  ||  #4  |  #5  |  #6  |
  |      |      |      ||      |      |      |
  +--------------------++--------------------+
  | Cell | Cell | Cell || Cell | Cell | Cell |
  |  #7  |  #8  |  #9  ||  #7  |  #8  |  #9  |
  |      |      |      ||      |      |      |
  +------+------+------++------+------+------+

  We assume that path integration works *somehow*. This model receives a "delta
  location" vector, and it shifts the active cells accordingly. The model stores
  intermediate coordinates of active cells. Whenever sensory cues activate a
  cell, the model adds this cell to the list of coordinates being shifted.
  Whenever sensory cues cause a cell to become inactive, that cell is removed
  from the list of coordinates.

  (This model doesn't attempt to propose how "path integration" works. It
  attempts to show how locations are anchored to sensory cues.)

  When orientation is set to 0 degrees, the displacement is a [di, dj],
  moving di cells "down" and dj cells "right".

  When orientation is set to 90 degrees, the displacement is essentially a
  [dx, dy], applied in typical x,y coordinates with the origin on the bottom
  left.

  Usage:
  - When the sensor moves, call movementCompute.
  - When the sensor senses something, call sensoryCompute.

  The "anchor input" is typically a feature-location pair SDR.

  To specify how points are tracked, pass anchoringMethod = "corners",
  "narrowing" or "discrete".  "discrete" will cause the network to operate in a
  fully discrete space, where uncertainty is impossible as long as movements are
  integers.  "narrowing" is designed to narrow down uncertainty of initial
  locations of sensory stimuli.  "corners" is designed for noise-tolerance, and
  will activate all cells that are possible outcomes of path integration.
  """

  def __init__(self,
               cellDimensions,
               moduleMapDimensions,
               orientation,
               anchorInputSize,
               cellCoordinateOffsets=(0.5,),
               activationThreshold=10,
               initialPermanence=0.21,
               connectedPermanence=0.50,
               learningThreshold=10,
               sampleSize=20,
               permanenceIncrement=0.1,
               permanenceDecrement=0.0,
               maxSynapsesPerSegment=-1,
               anchoringMethod="narrowing",
               rotationMatrix = None,
               seed=42):
    """
    @param cellDimensions (tuple(int, int))
    Determines the number of cells. Determines how space is divided between the
    cells.

    @param moduleMapDimensions (tuple(float, float))
    Determines the amount of world space covered by all of the cells combined.
    In grid cell terminology, this is equivalent to the "scale" of a module.
    A module with a scale of "5cm" would have moduleMapDimensions=(5.0, 5.0).

    @param orientation (float)
    The rotation of this map, measured in radians.

    @param anchorInputSize (int)
    The number of input bits in the anchor input.

    @param cellCoordinateOffsets (list of floats)
    These must each be between 0.0 and 1.0. Every time a cell is activated by
    anchor input, this class adds a "phase" which is shifted in subsequent
    motions. By default, this phase is placed at the center of the cell. This
    parameter allows you to control where the point is placed and whether multiple
    are placed. For example, with value [0.2, 0.8], when cell [2, 3] is activated
    it will place 4 phases, corresponding to the following points in cell
    coordinates: [2.2, 3.2], [2.2, 3.8], [2.8, 3.2], [2.8, 3.8]
    """

    self.cellDimensions = np.asarray(cellDimensions, dtype="int")
    self.moduleMapDimensions = np.asarray(moduleMapDimensions, dtype="float")
    self.phasesPerUnitDistance = 1.0 / self.moduleMapDimensions

    if rotationMatrix is None:
      self.orientation = orientation
      self.rotationMatrix = np.array(
        [[math.cos(orientation), -math.sin(orientation)],
         [math.sin(orientation), math.cos(orientation)]])
      if anchoringMethod == "discrete":
        # Need to convert matrix to have integer values
        nonzeros = self.rotationMatrix[np.where(np.abs(self.rotationMatrix)>0)]
        smallestValue = np.amin(nonzeros)
        self.rotationMatrix /= smallestValue
        self.rotationMatrix = np.ceil(self.rotationMatrix)
    else:
      self.rotationMatrix = rotationMatrix

    self.cellCoordinateOffsets = cellCoordinateOffsets

    # Phase is measured as a number in the range [0.0, 1.0)
    self.activePhases = np.empty((0,2), dtype="float")
    self.cellsForActivePhases = np.empty(0, dtype="int")
    self.phaseDisplacement = np.empty((0,2), dtype="float")

    self.activeCells = np.empty(0, dtype="int")
    self.activeSegments = np.empty(0, dtype="uint32")

    self.connections = SparseMatrixConnections(np.prod(cellDimensions),
                                               anchorInputSize)

    self.initialPermanence = initialPermanence
    self.connectedPermanence = connectedPermanence
    self.learningThreshold = learningThreshold
    self.sampleSize = sampleSize
    self.permanenceIncrement = permanenceIncrement
    self.permanenceDecrement = permanenceDecrement
    self.activationThreshold = activationThreshold
    self.maxSynapsesPerSegment = maxSynapsesPerSegment

    self.anchoringMethod = anchoringMethod

    self.rng = Random(seed)


  def reset(self):
    """
    Clear the active cells.
    """
    self.activePhases = np.empty((0,2), dtype="float")
    self.phaseDisplacement = np.empty((0,2), dtype="float")
    self.cellsForActivePhases = np.empty(0, dtype="int")
    self.activeCells = np.empty(0, dtype="int")


  def _computeActiveCells(self):
    # Round each coordinate to the nearest cell.
    activeCellCoordinates = np.floor(
      self.activePhases * self.cellDimensions).astype("int")

    # Convert coordinates to cell numbers.
    self.cellsForActivePhases = (
      np.ravel_multi_index(activeCellCoordinates.T, self.cellDimensions))
    self.activeCells = np.unique(self.cellsForActivePhases)


  def activateRandomLocation(self):
    """
    Set the location to a random point.
    """
    self.activePhases = np.array([np.random.random(2)])
    if self.anchoringMethod == "discrete":
      # Need to place the phase in the middle of a cell
      self.activePhases = np.floor(
        self.activePhases * self.cellDimensions)/self.cellDimensions
    self._computeActiveCells()


  def movementCompute(self, displacement, noiseFactor = 0):
    """
    Shift the current active cells by a vector.

    @param displacement (pair of floats)
    A translation vector [di, dj].
    """

    if noiseFactor != 0:
      displacement = copy.deepcopy(displacement)
      xnoise = np.random.normal(0, noiseFactor)
      ynoise = np.random.normal(0, noiseFactor)
      displacement[0] += xnoise
      displacement[1] += ynoise


    # Calculate delta in the module's coordinates.
    phaseDisplacement = (np.matmul(self.rotationMatrix, displacement) *
                         self.phasesPerUnitDistance)

    # Shift the active coordinates.
    np.add(self.activePhases, phaseDisplacement, out=self.activePhases)

    # In Python, (x % 1.0) can return 1.0 because of floating point goofiness.
    # Generally this doesn't cause problems, it's just confusing when you're
    # debugging.
    np.round(self.activePhases, decimals=9, out=self.activePhases)
    np.mod(self.activePhases, 1.0, out=self.activePhases)

    self._computeActiveCells()
    self.phaseDisplacement = phaseDisplacement


  def _sensoryComputeInferenceMode(self, anchorInput):
    """
    Infer the location from sensory input. Activate any cells with enough active
    synapses to this sensory input. Deactivate all other cells.

    @param anchorInput (numpy array)
    A sensory input. This will often come from a feature-location pair layer.
    """
    if len(anchorInput) == 0:
      return

    overlaps = self.connections.computeActivity(anchorInput,
                                                self.connectedPermanence)
    activeSegments = np.where(overlaps >= self.activationThreshold)[0]

    sensorySupportedCells = np.unique(
      self.connections.mapSegmentsToCells(activeSegments))

    inactivated = np.setdiff1d(self.activeCells, sensorySupportedCells)
    inactivatedIndices = np.in1d(self.cellsForActivePhases,
                                 inactivated).nonzero()[0]
    if inactivatedIndices.size > 0:
      self.activePhases = np.delete(self.activePhases, inactivatedIndices,
                                    axis=0)

    activated = np.setdiff1d(sensorySupportedCells, self.activeCells)

    # Find centers of point clouds
    if "corners" in self.anchoringMethod:
      activatedCoordsBase = np.transpose(
        np.unravel_index(sensorySupportedCells,
                         self.cellDimensions)).astype('float')
    else:
      activatedCoordsBase = np.transpose(
        np.unravel_index(activated, self.cellDimensions)).astype('float')

    # Generate points to add
    activatedCoords = np.concatenate(
      [activatedCoordsBase + [iOffset, jOffset]
       for iOffset in self.cellCoordinateOffsets
       for jOffset in self.cellCoordinateOffsets]
    )
    if "corners" in self.anchoringMethod:
      self.activePhases = activatedCoords / self.cellDimensions

    else:
      if activatedCoords.size > 0:
        self.activePhases = np.append(self.activePhases,
                                      activatedCoords / self.cellDimensions,
                                      axis=0)

    self._computeActiveCells()
    self.activeSegments = activeSegments


  def _sensoryComputeLearningMode(self, anchorInput):
    """
    Associate this location with a sensory input. Subsequently, anchorInput will
    activate the current location during anchor().

    @param anchorInput (numpy array)
    A sensory input. This will often come from a feature-location pair layer.
    """
    overlaps = self.connections.computeActivity(anchorInput,
                                                self.connectedPermanence)
    activeSegments = np.where(overlaps >= self.activationThreshold)[0]

    potentialOverlaps = self.connections.computeActivity(anchorInput)
    matchingSegments = np.where(potentialOverlaps >=
                                self.learningThreshold)[0]

    # Cells with a active segment: reinforce the segment
    cellsForActiveSegments = self.connections.mapSegmentsToCells(
      activeSegments)
    learningActiveSegments = activeSegments[
      np.in1d(cellsForActiveSegments, self.activeCells)]
    remainingCells = np.setdiff1d(self.activeCells, cellsForActiveSegments)

    # Remaining cells with a matching segment: reinforce the best
    # matching segment.
    candidateSegments = self.connections.filterSegmentsByCell(
      matchingSegments, remainingCells)
    cellsForCandidateSegments = (
      self.connections.mapSegmentsToCells(candidateSegments))
    candidateSegments = candidateSegments[
      np.in1d(cellsForCandidateSegments, remainingCells)]
    onePerCellFilter = np2.argmaxMulti(potentialOverlaps[candidateSegments],
                                       cellsForCandidateSegments)
    learningMatchingSegments = candidateSegments[onePerCellFilter]

    newSegmentCells = np.setdiff1d(remainingCells, cellsForCandidateSegments)

    for learningSegments in (learningActiveSegments,
                             learningMatchingSegments):
      self._learn(self.connections, self.rng, learningSegments,
                  anchorInput, potentialOverlaps,
                  self.initialPermanence, self.sampleSize,
                  self.permanenceIncrement, self.permanenceDecrement,
                  self.maxSynapsesPerSegment)

    # Remaining cells without a matching segment: grow one.
    numNewSynapses = len(anchorInput)

    if self.sampleSize != -1:
      numNewSynapses = min(numNewSynapses, self.sampleSize)

    if self.maxSynapsesPerSegment != -1:
      numNewSynapses = min(numNewSynapses, self.maxSynapsesPerSegment)

    newSegments = self.connections.createSegments(newSegmentCells)

    self.connections.growSynapsesToSample(
      newSegments, anchorInput, numNewSynapses,
      self.initialPermanence, self.rng)
    self.activeSegments = activeSegments


  def sensoryCompute(self, anchorInput, anchorGrowthCandidates, learn):
    if learn:
      self._sensoryComputeLearningMode(anchorGrowthCandidates)
    else:
      self._sensoryComputeInferenceMode(anchorInput)


  @staticmethod
  def _learn(connections, rng, learningSegments, activeInput,
             potentialOverlaps, initialPermanence, sampleSize,
             permanenceIncrement, permanenceDecrement, maxSynapsesPerSegment):
    """
    Adjust synapse permanences, grow new synapses, and grow new segments.

    @param learningActiveSegments (numpy array)
    @param learningMatchingSegments (numpy array)
    @param segmentsToPunish (numpy array)
    @param activeInput (numpy array)
    @param potentialOverlaps (numpy array)
    """
    # Learn on existing segments
    connections.adjustSynapses(learningSegments, activeInput,
                               permanenceIncrement, -permanenceDecrement)

    # Grow new synapses. Calculate "maxNew", the maximum number of synapses to
    # grow per segment. "maxNew" might be a number or it might be a list of
    # numbers.
    if sampleSize == -1:
      maxNew = len(activeInput)
    else:
      maxNew = sampleSize - potentialOverlaps[learningSegments]

    if maxSynapsesPerSegment != -1:
      synapseCounts = connections.mapSegmentsToSynapseCounts(
        learningSegments)
      numSynapsesToReachMax = maxSynapsesPerSegment - synapseCounts
      maxNew = np.where(maxNew <= numSynapsesToReachMax,
                        maxNew, numSynapsesToReachMax)

    connections.growSynapsesToSample(learningSegments, activeInput,
                                     maxNew, initialPermanence, rng)


  def getActiveCells(self):
    return self.activeCells


  def numberOfCells(self):
    return np.prod(self.cellDimensions)



class BodyToSpecificObjectModule2D(object):
  """
  Represents the body's location relative to a specific object. Typically
  these modules are arranged in an array which mirrors an array of
  SensorToSpecificObjectModules.
  """

  def __init__(self, cellDimensions):
    """
    Initialize this instance, and form all reciprocal connections between this
    instance and an array of SensorToSpecificObjectModules.

    @param cellDimensions (sequence of ints)
    @param sensorToSpecificObjectByColumn (sequence of SensorToSpecificObjectModules)
    """
    self.cellCount = np.prod(cellDimensions)
    self.cellDimensions = np.asarray(cellDimensions)
    self.connectedPermanence = 0.5

    # This offset logic assumes there are no "middle" rows or columns
    assert cellDimensions[0] % 2 == 0
    assert cellDimensions[1] % 2 == 0

  def reset(self):
    self.activeCells = np.empty(0, dtype="int")
    self.inhibitedCells = np.empty(0, dtype="int")


  def formReciprocalSynapses(self, sensorToSpecificObjectByColumn):
    cellCountBySource = {
      "sensorToSpecificObject": self.cellCount,
      "sensorToBody": self.cellCount,
    }
    self.connectionsByColumn = [
      Multiconnections(self.cellCount, cellCountBySource)
      for _ in xrange(len(sensorToSpecificObjectByColumn))]

    # Create a list of location-location-offset triples as 3 numpy arrays.
    # In the math that follows, make sure that the results correspond to this
    # cell order:
    #  body location: [0, 0, 0, 0, 0, 0, 0, 0, ..., 1, 1, 1, 1, 1, 1, 1, 1, ...]
    #  sensor offset: [0, 0, 0, 0, 1, 1, 1, 1, ..., 0, 0, 0, 0, 1, 1, 1, 1, ...]
    #  sensor location: (Computed. This changes most often.)
    #
    # This specific order isn't important, it's just important not to let the
    # lists get scrambled relative to each other.

    # Determine the offset vector for each sensorOffset.
    sensorOffset_i, sensorOffset_j = np.unravel_index(np.arange(self.cellCount),
                                                      self.cellDimensions)
    d_i = sensorOffset_i - (self.cellDimensions[0] // 2)
    d_j = sensorOffset_j - (self.cellDimensions[1] // 2)

    # An offset covers a 1x1 range of offsets, so create 4 triples for each
    # bodyLocation-sensorOffset.
    d_i = (d_i.reshape((-1,1)) + [0, -1, 0, -1]).flatten()
    d_j = (d_j.reshape((-1,1)) + [0, 0, -1, -1]).flatten()

    # Calculate the sensor location for each tuple.
    bodyLocation_i, bodyLocation_j = np.unravel_index(np.arange(self.cellCount),
                                                      self.cellDimensions)
    sensorLocation_i = (bodyLocation_i.reshape((-1, 1)) + d_i).flatten()
    sensorLocation_j = (bodyLocation_j.reshape((-1, 1)) + d_j).flatten()
    np.mod(sensorLocation_i, self.cellDimensions[0], out=sensorLocation_i)
    np.mod(sensorLocation_j, self.cellDimensions[1], out=sensorLocation_j)

    # Gather the results
    sensorLocationCells = np.ravel_multi_index(
      (sensorLocation_i, sensorLocation_j), self.cellDimensions)

    # Gather the pairs that correspond with these results.
    bodyLocationCells = np.repeat(
      np.arange(self.cellCount, dtype="uint32"),
      4 * self.cellCount)
    sensorOffsetCells = np.tile(
      np.repeat(np.arange(self.cellCount, dtype="uint32"), 4),
      self.cellCount)

    # Grow bidirectional segments for each of these triples. The connections
    # will be the same for every column.
    presynapticCellsForBodyToObject = {
      "sensorToBody": sensorOffsetCells,
      "sensorToSpecificObject": sensorLocationCells
    }
    presynapticCellsForSensorToObject = {
      "sensorToBody": sensorOffsetCells,
      "bodyToSpecificObject": bodyLocationCells
    }

    for (connections,
         sensorToSpecificObject) in zip(self.connectionsByColumn,
                                        sensorToSpecificObjectByColumn):
      bodySegments = connections.createSegments(
        bodyLocationCells)
      connections.setPermanences(
        bodySegments, presynapticCellsForBodyToObject, 1.0)

      sensorSegments = sensorToSpecificObject.metricConnections.createSegments(
        sensorLocationCells)
      sensorToSpecificObject.metricConnections.setPermanences(
        sensorSegments, presynapticCellsForSensorToObject, 1.0)


  def activateRandomLocation(self):
    self.activeCells = np.array([random.choice(xrange(self.cellCount))],
                                dtype="int")


  def compute(self, sensorToBodyByColumn, sensorToSpecificObjectByColumn):
    """
    Compute the
      "body's location relative to a specific object"
    from an array of
      "sensor's location relative to a specific object"
    and an array of
      "sensor's location relative to body"

    These arrays consist of one module per cortical column.

    This is a metric computation, similar to that of the
    SensorToSpecificObjectModule, but with voting. In effect, the columns vote
    on "the body's location relative to a specific object".

    Note: Each column can vote for an arbitrary number of cells, but it can't
    vote for a single cell more than once. This is necessary because we don't
    want ambiguity in a column to cause some cells to get extra votes. There are
    a few ways that this could be biologically plausible:

    - Explanation 1: Nearby dendritic segments are independent coincidence
      detectors, but perhaps their dendritic spikes don't sum. Meanwhile,
      maybe dendritic spikes from far away dendritic segments do sum.
    - Explanation 2: Dendritic spikes from different columns are separated
      temporally, not spatially. All the spikes from one column "arrive" at
      the cell at the same time, but the dendritic spikes from other columns
      arrive at other times. With each of these temporally-separated dendritic
      spikes, the unsupported cells are inhibited, or the spikes' effects are
      summed.
    - Explanation 3: Another population of cells within the cortical column
      might calculate the "body's location relative to a specific object" in
      this same "metric" way, but without tallying any votes. Then it relays
      this SDR subcortically, voting 0 or 1 times for each cell.

    @param sensorToBodyInputs (list of numpy arrays)
    The "sensor's location relative to the body" input from each cortical column

    @param sensorToSpecificObjectInputs (list of numpy arrays)
    The "sensor's location relative to specific object" input from each
    cortical column
    """
    votesByCell = np.zeros(self.cellCount, dtype="int")

    self.activeSegmentsByColumn = []

    for (connections,
         activeSensorToBodyCells,
         activeSensorToSpecificObjectCells) in zip(self.connectionsByColumn,
                                                   sensorToBodyByColumn,
                                                   sensorToSpecificObjectByColumn):
      overlaps = connections.computeActivity({
        "sensorToBody": activeSensorToBodyCells,
        "sensorToSpecificObject": activeSensorToSpecificObjectCells,
      })
      activeSegments = np.where(overlaps >= 2)[0]
      votes = connections.mapSegmentsToCells(activeSegments)
      votes = np.unique(votes)  # Only allow a column to vote for a cell once.
      votesByCell[votes] += 1

      self.activeSegmentsByColumn.append(activeSegments)

    candidates = np.where(votesByCell == np.max(votesByCell))[0]

    # If possible, select only from current active cells.
    #
    # If we were to always activate all candidates, there would be an explosive
    # back-and-forth between this layer and the sensorToSpecificObject layer.
    self.activeCells = np.intersect1d(self.activeCells, candidates)

    if self.activeCells.size == 0:
      # Otherwise, activate all cells with the maximum number of active
      # segments.
      self.activeCells = candidates

    self.inhibitedCells = np.setdiff1d(np.where(votesByCell > 0)[0],
                                       self.activeCells)


  def getActiveCells(self):
    return self.activeCells




class SensorToSpecificObjectModule(object):
  """
  Represents the sensor location relative to a specific object. Typically
  these modules are arranged in an array, and the combined population SDR is
  used to predict a feature-location pair.

  This class has two sets of connections. Both of them compute the "sensor's
  location relative to a specific object" in different ways.

  The "metric connections" compute it from the
    "body's location relative to a specific object"
  and the
    "sensor's location relative to body"
  These connections are learned once and then never need to be updated. They
  might be genetically hardcoded. They're initialized externally, e.g. in
  BodyToSpecificObjectModule2D.

  The "anchor connections" compute it from the sensory input. Whenever a
  cortical column learns a feature-location pair, this layer forms reciprocal
  connections with the feature-location pair layer.

  These segments receive input at different times. The metric connections
  receive input first, and they activate a set of cells. This set of cells is
  used externally to predict a feature-location pair. Then this feature-location
  pair is the input to the anchor connections.
  """

  def __init__(self, cellDimensions, anchorInputSize,
               activationThreshold=10,
               initialPermanence=0.21,
               connectedPermanence=0.50,
               learningThreshold=10,
               sampleSize=20,
               permanenceIncrement=0.1,
               permanenceDecrement=0.0,
               maxSynapsesPerSegment=-1,
               seed=42):
    """
    @param cellDimensions (sequence of ints)
    @param anchorInputSize (int)
    @param activationThreshold (int)
    """
    self.activationThreshold = activationThreshold
    self.initialPermanence = initialPermanence
    self.connectedPermanence = connectedPermanence
    self.learningThreshold = learningThreshold
    self.sampleSize = sampleSize
    self.permanenceIncrement = permanenceIncrement
    self.permanenceDecrement = permanenceDecrement
    self.activationThreshold = activationThreshold
    self.maxSynapsesPerSegment = maxSynapsesPerSegment

    self.rng = Random(seed)

    self.cellCount = np.prod(cellDimensions)
    cellCountBySource = {
      "bodyToSpecificObject": self.cellCount,
      "sensorToBody": self.cellCount,
    }
    self.metricConnections = Multiconnections(self.cellCount,
                                              cellCountBySource)
    self.anchorConnections = SparseMatrixConnections(self.cellCount,
                                                     anchorInputSize)


  def reset(self):
    self.activeCells = np.empty(0, dtype="int")


  def metricCompute(self, sensorToBody, bodyToSpecificObject):
    """
    Compute the
      "sensor's location relative to a specific object"
    from the
      "body's location relative to a specific object"
    and the
      "sensor's location relative to body"

    @param sensorToBody (numpy array)
    Active cells of a single module that represents the sensor's location
    relative to the body

    @param bodyToSpecificObject (numpy array)
    Active cells of a single module that represents the body's location relative
    to a specific object
    """
    overlaps = self.metricConnections.computeActivity({
      "bodyToSpecificObject": bodyToSpecificObject,
      "sensorToBody": sensorToBody,
    })

    self.activeMetricSegments = np.where(overlaps >= 2)[0]
    self.activeCells = np.unique(
      self.metricConnections.mapSegmentsToCells(
        self.activeMetricSegments))


  def anchorCompute(self, anchorInput, learn):
    """
    Compute the
      "sensor's location relative to a specific object"
    from the feature-location pair.

    @param anchorInput (numpy array)
    Active cells in the feature-location pair layer

    @param learn (bool)
    If true, maintain current cell activity and learn this input on the
    currently active cells
    """
    if learn:
      self._anchorComputeLearningMode(anchorInput)
    else:
      overlaps = self.anchorConnections.computeActivity(
        anchorInput, self.connectedPermanence)

      self.activeSegments = np.where(overlaps >= self.activationThreshold)[0]
      self.activeCells = np.unique(
        self.anchorConnections.mapSegmentsToCells(self.activeSegments))


  def _anchorComputeLearningMode(self, anchorInput):
    """
    Associate this location with a sensory input. Subsequently, anchorInput will
    activate the current location during anchor().

    @param anchorInput (numpy array)
    A sensory input. This will often come from a feature-location pair layer.
    """

    overlaps = self.anchorConnections.computeActivity(
      anchorInput, self.connectedPermanence)

    activeSegments = np.where(overlaps >= self.activationThreshold)[0]

    potentialOverlaps = self.anchorConnections.computeActivity(anchorInput)
    matchingSegments = np.where(potentialOverlaps >=
                                self.learningThreshold)[0]

    # Cells with a active segment: reinforce the segment
    cellsForActiveSegments = self.anchorConnections.mapSegmentsToCells(
      activeSegments)
    learningActiveSegments = activeSegments[
      np.in1d(cellsForActiveSegments, self.activeCells)]
    remainingCells = np.setdiff1d(self.activeCells, cellsForActiveSegments)

    # Remaining cells with a matching segment: reinforce the best
    # matching segment.
    candidateSegments = self.anchorConnections.filterSegmentsByCell(
      matchingSegments, remainingCells)
    cellsForCandidateSegments = (
      self.anchorConnections.mapSegmentsToCells(candidateSegments))
    candidateSegments = candidateSegments[
      np.in1d(cellsForCandidateSegments, remainingCells)]
    onePerCellFilter = np2.argmaxMulti(potentialOverlaps[candidateSegments],
                                       cellsForCandidateSegments)
    learningMatchingSegments = candidateSegments[onePerCellFilter]

    newSegmentCells = np.setdiff1d(remainingCells, cellsForCandidateSegments)

    for learningSegments in (learningActiveSegments,
                             learningMatchingSegments):
      self._learn(self.anchorConnections, self.rng, learningSegments,
                  anchorInput, potentialOverlaps,
                  self.initialPermanence, self.sampleSize,
                  self.permanenceIncrement, self.permanenceDecrement,
                  self.maxSynapsesPerSegment)

    # Remaining cells without a matching segment: grow one.
    numNewSynapses = len(anchorInput)

    if self.sampleSize != -1:
      numNewSynapses = min(numNewSynapses, self.sampleSize)

    if self.maxSynapsesPerSegment != -1:
      numNewSynapses = min(numNewSynapses, self.maxSynapsesPerSegment)

    newSegments = self.anchorConnections.createSegments(newSegmentCells)

    self.anchorConnections.growSynapsesToSample(
      newSegments, anchorInput, numNewSynapses,
      self.initialPermanence, self.rng)

    self.activeSegments = activeSegments


  @staticmethod
  def _learn(connections, rng, learningSegments, activeInput,
             potentialOverlaps, initialPermanence, sampleSize,
             permanenceIncrement, permanenceDecrement, maxSynapsesPerSegment):
    """
    Adjust synapse permanences, grow new synapses, and grow new segments.

    @param learningActiveSegments (numpy array)
    @param learningMatchingSegments (numpy array)
    @param segmentsToPunish (numpy array)
    @param activeInput (numpy array)
    @param potentialOverlaps (numpy array)
    """
    # Learn on existing segments
    connections.adjustSynapses(learningSegments, activeInput,
                               permanenceIncrement, -permanenceDecrement)

    # Grow new synapses. Calculate "maxNew", the maximum number of synapses to
    # grow per segment. "maxNew" might be a number or it might be a list of
    # numbers.
    if sampleSize == -1:
      maxNew = len(activeInput)
    else:
      maxNew = sampleSize - potentialOverlaps[learningSegments]

    if maxSynapsesPerSegment != -1:
      synapseCounts = connections.mapSegmentsToSynapseCounts(
        learningSegments)
      numSynapsesToReachMax = maxSynapsesPerSegment - synapseCounts
      maxNew = np.where(maxNew <= numSynapsesToReachMax,
                        maxNew, numSynapsesToReachMax)

    connections.growSynapsesToSample(learningSegments, activeInput,
                                     maxNew, initialPermanence, rng)


  def getActiveCells(self):
    return self.activeCells



class SensorToBodyModule2D(object):
  """
  This is essentially an encoder. It takes a location vector and converts it
  into an active cell in a "sensor relative to body" module.

  This class captures all logic pertaining to the module's scale and
  orientation.
  """

  def __init__(self, cellDimensions, moduleMapDimensions, orientation):
    """
    @param cellDimensions (tuple(int, int))
    Determines the number of cells. Determines how space is divided between the
    cells.

    @param moduleMapDimensions (tuple(float, float))
    Determines the amount of world space covered by all of the cells combined.
    In grid cell terminology, this is equivalent to the "scale" of a module.
    A module with a scale of "5cm" would have moduleMapDimensions=(5.0, 5.0).

    @param orientation (float)
    The rotation of this map, measured in radians.
    """
    self.cellDimensions = np.asarray(cellDimensions, dtype="int")
    self.moduleMapDimensions = np.asarray(moduleMapDimensions, dtype="float")
    self.cellFieldsPerUnitDistance = self.cellDimensions / self.moduleMapDimensions

    self.orientation = orientation
    self.rotationMatrix = np.array(
      [[math.cos(orientation), -math.sin(orientation)],
       [math.sin(orientation), math.cos(orientation)]])

    self.activeCells = np.array(0, dtype="uint32")


  def compute(self, egocentricLocation):
    """
    Compute the new active cells from the given "sensor location relative to
    body" vector.

    @param egocentricLocation (pair of floats)
    [di, dj] offset of the sensor from the body.
    """
    offsetInCellFields = (np.matmul(self.rotationMatrix, egocentricLocation) *
                          self.cellFieldsPerUnitDistance)

    np.mod(offsetInCellFields, self.cellDimensions, out=offsetInCellFields)
    self.activeCells = np.unique(
      np.ravel_multi_index(np.floor(offsetInCellFields).T.astype('int'),
                           self.cellDimensions))


  def getActiveCells(self):
    return self.activeCells
