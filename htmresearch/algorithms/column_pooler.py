# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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

import numpy

import scipy.sparse as sparse

from htmresearch.algorithms.extended_temporal_memory import ExtendedTemporalMemory

realDType = numpy.float32
uintType = "uint32"



class ColumnPooler(ExtendedTemporalMemory):
  """
  This class constitutes a temporary implementation for a cross-column pooler.

  the default implementation of the temporal pooler (TP).
  The main goal of the TP is to form stable and unique representations of an
  input data stream of cell activity from a Temporal Memory. More specifically,
  the TP forms its stable representation based on input cells that were
  correctly temporally predicted by Temporal Memory.
  """

  def __init__(self,
               inputWidth,
               numActiveColumnsPerInhArea=40,
               **kwargs):
    """
    Please see spatial_pooler.py in NuPIC for descriptions of common
    constructor parameters.
    """
    # Override: we only support one cell per column for now
    kwargs['cellsPerColumn'] = 1
    super(ColumnPooler, self).__init__(**kwargs)

    self.inputWidth = inputWidth
    self.numActiveColumnsPerInhArea = numActiveColumnsPerInhArea
    self.proximalSegments = sparse.lil_matrix((self.numberOfCells(),inputWidth),
                                             dtype=realDType)
    self.previousOverlaps = None


  def compute(self,
              feedforwardInput=None,
              activeExternalCells=None,
              activeApicalCells=None,
              formInternalConnections=True,
              learn=True):
    """

    @param feedforwardInput     A numpy array of 0's and 1's that comprises
                                the input (typically the active cells in TM)

    @param learn                If True, we are learning a new object

    """
    if activeExternalCells is None:
      activeExternalCells = set()

    if learn:
      self.computeLearningMode(feedforwardInput=feedforwardInput,
                               lateralInput=activeExternalCells)


  def computeLearningMode(self, feedforwardInput, lateralInput):
    """
    Computes when learning new object

    Learning mode: we are learning a new object. If there is no prior
    activity, we randomly activate 2% of cells and create connections to
    incoming input. If there was prior activity, we maintain it.

    These cells will represent the object and learn distal connections to
    lateral cortical columns.

    @param feedforwardInput     A numpy array of 0's and 1's that comprises
                                the input (typically the active cells in TM)
    @param lateralInput         A list of list of active cells from neighboring
                                columns. len(lateralInput) == number of
                                connected neighboring cortical columns.

    """
    assert (numpy.size(feedforwardInput) == self.inputWidth)

    # Figure out which cells are active due to feedforward proximal inputs
    overlap = self.proximalSegments.dot(feedforwardInput)

    # If we have bottom up input and there are no previously active cells,
    # select a random subset of the cells
    if overlap.max() < self.minThreshold:
      if len(self.activeCells) == 0:
        # No previously active cells, need to create new SDR
        self.activeCells = set(self._random.shuffle(
              numpy.array(range(self.numberOfCells()),
                          dtype="uint32"))[0:self.numActiveColumnsPerInhArea])

    # else: we maintain previous activity
    print self.activeCells

    # Compute distal segment activity for each cell
    if len(lateralInput) > 0:
      # Figure out distal input into active cells
      pass

    # Reconcile and select the cells with sufficient bottom up activity plus
    # maximal lateral activity


    # Those cells that remain active will learn on their proximal and distal
    # dendrites.  If there are no cells active, no learning happens.


  def computeInferenceMode(self, lateralInput, learn):
    """
    Inference mode: if there is some feedforward activity, perform
    spatial pooling on it to recognize previously known objects. If there
    is no feedforward activity, maintain previous activity.


    @param feedforwardInput     A numpy array of 0's and 1's that comprises
                                the input (typically the active cells in TM)
    @param lateralInput         A list of list of active cells from neighboring
                                columns. len(lateralInput) == number of
                                connected neighboring cortical columns.

    @param learn                If true, learn on distal segments.

    """
    pass


  def numberOfInputs(self):
    """
    Returns the number of inputs into this layer
    @return (int) Number of inputs
    """
    return self.inputWidth




