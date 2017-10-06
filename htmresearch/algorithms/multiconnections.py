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

"""Connections class that allows segments to connect to multiple layers"""

import numpy as np

from nupic.bindings.math import SparseMatrixConnections


class Multiconnections(object):
  """
  A Connections class that organizes its connections by presynaptic layer.
  Every segment can form synapses to multiple presynaptic layers.

  We could port this class to C++ and reduce a lot of redundant segment
  bookkeeping.
  """
  def __init__(self, cellCount, cellCountBySource):
    """
    @param cellCountBySource (dict)
    The number of cells in each source. Example:
      {"customInputName1": 16,
       "customInputName2": 42}
    """

    self.connectionsBySource = dict(
      (source, SparseMatrixConnections(cellCount, presynapticCellCount))
      for source, presynapticCellCount in cellCountBySource.iteritems())


  def computeActivity(self, activeInputsBySource, permanenceThreshold=None):
    """
    Calculate the number of active synapses per segment.

    @param activeInputsBySource (dict)
    The active cells in each source. Example:
      {"customInputName1": np.array([42, 69])}
    """
    overlaps = None

    for source, connections in self.connectionsBySource.iteritems():
      o = connections.computeActivity(activeInputsBySource[source],
                                      permanenceThreshold)
      if overlaps is None:
        overlaps = o
      else:
        overlaps += o

    return overlaps


  def createSegments(self, cells):
    """
    Create a segment on each of the specified cells.

    @param cells (numpy array)
    """
    segments = None

    for connections in self.connectionsBySource.itervalues():
      created = connections.createSegments(cells)

      if segments is None:
        segments = created
      else:
        # Sanity-check that the segment numbers are the same.
        np.testing.assert_equal(segments, created)

    return segments


  def growSynapses(self, segments, activeInputsBySource, initialPermanence):
    """
    Grow synapses to each of the specified inputs on each specified segment.

    @param segments (numpy array)
    The segments that should add synapses

    @param activeInputsBySource (dict)
    The active cells in each source. Example:
      {"customInputName1": np.array([42, 69])}

    @param initialPermanence (float)
    """
    for source, connections in self.connectionsBySource.iteritems():
      connections.growSynapses(segments, activeInputsBySource[source],
                               initialPermanence)


  def setPermanences(self, segments, presynapticCellsBySource, permanence):
    """
    Set the permanence of a specific set of synapses. Any synapses that don't
    exist will be initialized. Any existing synapses will be overwritten.

    Conceptually, this method takes a list of [segment, presynapticCell] pairs
    and initializes their permanence. For each segment, one synapse is added
    (although one might be added for each "source"). To add multiple synapses to
    a segment, include it in the list multiple times.

    The total number of affected synapses is len(segments)*number_of_sources*1.

    @param segments (numpy array)
    One segment for each synapse that should be added

    @param presynapticCellsBySource (dict of numpy arrays)
    One presynaptic cell for each segment.
    Example:
      {"customInputName1": np.array([42, 69])}

    @param permanence (float)
    The permanence to assign the synapse
    """
    permanences = np.repeat(np.float32(permanence), len(segments))

    for source, connections in self.connectionsBySource.iteritems():
      if source in presynapticCellsBySource:
        connections.matrix.setElements(segments, presynapticCellsBySource[source],
                                       permanences)


  def mapSegmentsToCells(self, segments):
    """
    @param segments (numpy array)
    """
    connections = next(self.connectionsBySource.itervalues())
    return connections.mapSegmentsToCells(segments)
