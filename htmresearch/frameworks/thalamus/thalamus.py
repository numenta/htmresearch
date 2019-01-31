# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

"""

An implementation of thalamic control and routing as proposed in the Cosyne
submission:

  A dendritic mechanism for dynamic routing and control in the thalamus
           Carmen Varela & Subutai Ahmad

"""

from __future__ import print_function

import numpy as np

from nupic.bindings.math import Random, SparseMatrixConnections



class Thalamus(object):
  """
  A simple discrete time thalamus.
  """

  def __init__(self,
               trnCellShape=(32, 32),
               relayCellShape=(32, 32),
               inputShape=(32, 32),
               l6CellCount=1024,
               trnThreshold=10,
               relayThreshold=1,
               seed=42):
    """

    :param trnCellShape: a 2D shape for the TRN
    :param relayCellShape: a 2D shape for the relay cells
    :param l6CellCount:
    :param trnThreshold:
    :param relayThreshold:

    :param seed:
        Seed for the random number generator.
    """

    self.trnCellShape = trnCellShape
    self.trnWidth = trnCellShape[0]
    self.trnHeight = trnCellShape[1]
    self.relayCellShape = relayCellShape
    self.relayWidth = relayCellShape[0]
    self.relayHeight = relayCellShape[1]
    self.l6CellCount = l6CellCount
    self.trnThreshold = trnThreshold
    self.relayThreshold = relayThreshold
    self.inputShape = inputShape
    self.seed = seed
    self.rng = Random(seed)
    self.trnActivationThreshold = 5

    self.trnConnections = SparseMatrixConnections(
      trnCellShape[0]*trnCellShape[1], l6CellCount)

    self.relayConnections = SparseMatrixConnections(
      relayCellShape[0]*relayCellShape[1],
      trnCellShape[0]*trnCellShape[1])

    # Initialize/reset variables that are updated with calls to compute
    self.reset()


  def learnL6Pattern(self, l6Pattern, cellsToLearnOn):
    """
    Learn the given l6Pattern on TRN cell dendrites. The TRN cells to learn
    are given in cellsTeLearnOn. Each of these cells will learn this pattern on
    a single dendritic segment.

    :param l6Pattern: an SDR from L6. List of indices corresponding to L6 cells.

    :param cellsToLearnOn:
      Each cell index is (x,y) corresponding to the TRN cells that should learn
      this pattern. For each cell, create a new dendrite that stores this
      pattern. The SDR is stored on this dendrite


    """
    cellIndices = [self.trnCellIndex(x) for x in cellsToLearnOn]
    newSegments = self.trnConnections.createSegments(cellIndices)
    self.trnConnections.growSynapses(newSegments, l6Pattern, 1.0)

    print("Learning L6 SDR:", l6Pattern,
          "new segments: ", newSegments,
          "cells:", self.trnConnections.mapSegmentsToCells(newSegments))


  def deInactivateCells(self, l6Input):
    """
    Activate trnCells according to the l6Input. These in turn will impact 
    bursting mode in relay cells that are connected to these trnCells.
    Given the feedForwardInput, compute which cells will be silent, tonic,
    or bursting.
    
    :param l6Input: 

    :return: nothing
    """

    # Figure out which TRN cells recognize the L6 pattern.
    self.trnOverlaps = self.trnConnections.computeActivity(l6Input, 0.5)
    self.activeSegments = np.flatnonzero(
      self.trnOverlaps >= self.trnActivationThreshold)
    self.cellIndices = self.trnConnections.mapSegmentsToCells(
      self.activeSegments)

    print("trnOverlaps:", self.trnOverlaps,
          "active segments:", self.activeSegments)
    for s, idx in zip(self.activeSegments, self.cellIndices):
      print(self.trnOverlaps[s], idx, self.trnIndextoCoord(idx))

    # TODO: Figure out which relay cells have dendrites in de-inactivated state


  def computeFeedForwardActivity(self, feedForwardInput):
    """
    Activate trnCells according to the l6Input. These in turn will impact
    bursting mode in relay cells that are connected to these trnCells.
    Given the feedForwardInput, compute which cells will be silent, tonic,
    or bursting.

    :param feedForwardInput:

    :return: a numpy matrix of shape relayCellShape
    """
    pass


  def reset(self):
    """
    Set everything back to zero
    """
    self.trnOverlaps = []
    self.activeSegments = []
    self.cellIndices = []


  def trnCellIndex(self, x):
    """
    Map a 2D coordinate to 1D cell index.
    :param x: a 2D coordinate
    :return: integer index
    """
    return x[1]*self.trnWidth + x[0]


  def trnIndextoCoord(self, i):
    """
    Map 1D cell index to a 2D coordinate
    :param i: 1D cell index
    :return: (x, y), a 2D coordinate
    """
    x = i % self.trnWidth
    y = i / self.trnWidth
    return x, y


  def _initializeTRNToRelayCellConnections(self):
    """
    Initialize TRN to relay cell connectivity.
    For each relay cell, create a dendritic segment for each TRN cell it
    connects to.
    """
    pass