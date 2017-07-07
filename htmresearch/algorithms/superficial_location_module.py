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

import numpy as np

from htmresearch.support import numpy_helpers as np2
from nupic.bindings.math import SparseMatrixConnections, Random



class SuperficialLocationModule2D(object):
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

  When orientation is set to 0 degrees, the deltaLocation is a [di, dj],
  moving di cells "down" and dj cells "right".

  When orientation is set to 90 degrees, the deltaLocation is essentially a
  [dx, dy], applied in typical x,y coordinates with the origin on the bottom
  left.

  Usage:

  Adjust the location in response to motor input:
    lm.shift([di, dj])

  Adjust the location in response to sensory input:
    lm.anchor(anchorInput)

  Learn an anchor input for the current location:
    lm.learn(anchorInput)

  The "anchor input" is typically a feature-location pair SDR.

  During inference, you'll typically call:
    lm.shift(...)
    # Consume lm.getActiveCells()
    # ...
    lm.anchor(...)

  During learning, you'll do the same, but you'll call lm.learn() instead of
  lm.anchor().
  """


  def __init__(self,
               cellDimensions,
               moduleMapDimensions,
               orientation,
               anchorInputSize,
               pointOffsets=(0.5,),
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

    @param pointOffsets (list of floats)
    These must each be between 0.0 and 1.0. Every time a cell is activated by
    anchor input, this class adds a "point" which is shifted in subsequent
    motions. By default, this point is placed at the center of the cell. This
    parameter allows you to control where the point is placed and whether multiple
    are placed. For example, With value [0.2, 0.8], it will place 4 points:
    [0.2, 0.2], [0.2, 0.8], [0.8, 0.2], [0.8, 0.8]
    """

    self.cellDimensions = np.asarray(cellDimensions, dtype="int")
    self.moduleMapDimensions = np.asarray(moduleMapDimensions, dtype="float")
    self.cellFieldsPerUnitDistance = self.cellDimensions / self.moduleMapDimensions

    self.orientation = orientation
    self.rotationMatrix = np.array(
      [[math.cos(orientation), -math.sin(orientation)],
       [math.sin(orientation), math.cos(orientation)]])

    self.pointOffsets = pointOffsets

    # These coordinates are in units of "cell fields".
    self.activePoints = np.empty((0,2), dtype="float")
    self.cellsForActivePoints = np.empty(0, dtype="int")

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

    self.rng = Random(seed)


  def reset(self):
    """
    Clear the active cells.
    """
    self.activePoints = np.empty((0,2), dtype="float")
    self.cellsForActivePoints = np.empty(0, dtype="int")
    self.activeCells = np.empty(0, dtype="int")


  def _computeActiveCells(self):
    # Round each coordinate to the nearest cell.
    flooredActivePoints = np.floor(self.activePoints).astype("int")

    # Convert coordinates to cell numbers.
    self.cellsForActivePoints = (
      np.ravel_multi_index(flooredActivePoints.T, self.cellDimensions))
    self.activeCells = np.unique(self.cellsForActivePoints)


  def activateRandomLocation(self):
    """
    Set the location to a random point.
    """
    self.activePoints = np.array([np.random.random(2) * self.cellDimensions])
    self._computeActiveCells()


  def shift(self, deltaLocation):
    """
    Shift the current active cells by a vector.

    @param deltaLocation (pair of floats)
    A translation vector [di, dj].
    """
    # Calculate delta in the module's coordinates.
    deltaLocationInCellFields = (np.matmul(self.rotationMatrix, deltaLocation) *
                                 self.cellFieldsPerUnitDistance)

    # Shift the active coordinates.
    np.add(self.activePoints, deltaLocationInCellFields, out=self.activePoints)
    np.mod(self.activePoints, self.cellDimensions, out=self.activePoints)

    self._computeActiveCells()


  def anchor(self, anchorInput):
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
    inactivatedIndices = np.in1d(self.cellsForActivePoints,
                                 inactivated).nonzero()[0]
    if inactivatedIndices.size > 0:
      self.activePoints = np.delete(self.activePoints, inactivatedIndices,
                                    axis=0)

    activated = np.setdiff1d(sensorySupportedCells, self.activeCells)

    activatedCoordsBase = np.transpose(
      np.unravel_index(activated, self.cellDimensions)).astype('float')

    activatedCoords = np.concatenate(
      [activatedCoordsBase + [iOffset, jOffset]
       for iOffset in self.pointOffsets
       for jOffset in self.pointOffsets]
    )
    if activatedCoords.size > 0:
      self.activePoints = np.append(self.activePoints, activatedCoords, axis=0)

    self._computeActiveCells()
    self.activeSegments = activeSegments


  def learn(self, anchorInput):
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
