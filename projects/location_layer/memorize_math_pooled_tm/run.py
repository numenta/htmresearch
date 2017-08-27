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

"""
Train a network to associate pairs of operands with a pooled result.

The network uses a TemporalMemory to pair the operands, and a pooling layer to
pool the pairs -- e.g. to learn the equivalence of 2+2 and 1+3.

In this network, one operand is the "driving" operand, and the other is the
"context" operand. The "context" operand can be a union, but the "driving"
operand can't.

This is one possible network that could be used to determine a location from a
location-offset pair, or for determining an offset from a location-location
pair.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import defaultdict
import glob
import json
import multiprocessing
import os
import random

import numpy as np

from nupic.bindings.math import SparseMatrixConnections, SparseMatrix
from htmresearch.algorithms.apical_tiebreak_temporal_memory import (
  ApicalTiebreakPairMemory)

from generate_sdrs import (generateMinicolumnSDRs, createEvenlySpreadSDRs,
                           carefullyCollideContexts)


class ForwardModel(object):
  """
  Simple forward model. Every cell has a set of synapses. The cell fires when
  its number of active synapses reaches a threshold.
  """

  def __init__(self, cellCount, inputSize, threshold):
    self.permanences = SparseMatrix(cellCount, inputSize)
    self.threshold = threshold
    self.activeCells = np.empty(0, dtype='uint32')

  def associate(self, activeCells, activeInput):
    self.activeCells = activeCells
    self.permanences.setZerosOnOuter(
      self.activeCells, activeInput, 1.0)

  def infer(self, activeInput):
    overlaps = self.permanences.rightVecSumAtNZSparse(activeInput)
    self.activeCells = np.where(overlaps >= self.threshold)[0]
    self.activeCells.sort()


class SegmentedForwardModel(object):
  """
  A forward model that uses dendrite segments. Every cell has a set of segments.
  Every segment has a set of synapses. The cell fires when the number of active
  synapses on one of its segments reaches a threshold.
  """

  def __init__(self, cellCount, inputSize, threshold):
    self.proximalConnections = SparseMatrixConnections(cellCount, inputSize)
    self.threshold = threshold
    self.activeCells = np.empty(0, dtype='uint32')
    self.activeSegments = np.empty(0, dtype='uint32')

  def associate(self, activeCells, activeInput):
    self.activeCells = activeCells
    self.activeSegments = self.proximalConnections.createSegments(
      activeCells)
    self.proximalConnections.matrix.setZerosOnOuter(
      self.activeSegments, activeInput, 1.0)

  def infer(self, activeInput):
    overlaps = self.proximalConnections.computeActivity(activeInput)
    self.activeSegments = np.where(overlaps >= self.threshold)[0]
    self.activeCells = self.proximalConnections.mapSegmentsToCells(
      self.activeSegments)
    self.activeCells.sort()


class PoolOfPairsLocation1DExperiment(object):
  """
  There are a lot of ways this experiment could choose to associate "operands"
  with results -- e.g. we could just do it randomly. This particular experiment
  assumes there are an equal number of "operand1", "operand2", and "result"
  values. It assigns each operand/result an index, and it relates these via:

    result = (operand1 + operand2) % numLocations

  Note that this experiment would be fundamentally no different if it used
  subtraction:

    result = (operand1 - operand2) % numLocations

  The resulting network would be identical, it's just our interpretation of the
  SDRs that would change.

  This experiment intentionally mimics a 1D space with wraparound, with
  operands/results representing 1D locations and offsets. You can think of this
  as:

    location2 = location1 + offset
    offset = location2 - location1
  """

  def __init__(self,
               numLocations=25,
               numMinicolumns=15,
               numActiveMinicolumns=10,
               poolingThreshold=8,
               cellsPerColumn=8,
               segmentedProximal=True,
               segmentedPooling=True,
               minicolumnSDRs=None):

    self.numOperandCells = 100
    self.numActiveOperandCells = 4
    self.numResultCells = 100
    self.numActiveResultCells = 4
    self.numLocations = numLocations
    self.numActiveMinicolumns = numActiveMinicolumns

    self.contextOperandSDRs = createEvenlySpreadSDRs(
      numLocations, self.numOperandCells, self.numActiveOperandCells)
    self.resultSDRs = createEvenlySpreadSDRs(
      numLocations, self.numResultCells, self.numActiveResultCells)
    self.drivingOperandSDRs = createEvenlySpreadSDRs(
      numLocations, self.numOperandCells, self.numActiveOperandCells)

    if minicolumnSDRs is None:
      self.minicolumnSDRs = createEvenlySpreadSDRs(
        self.numLocations, numMinicolumns, numActiveMinicolumns)
    else:
      assert len(minicolumnSDRs) >= self.numLocations
      self.minicolumnSDRs = list(minicolumnSDRs)
      random.shuffle(self.minicolumnSDRs)

    self.minicolumnParams = {
      "cellCount": numMinicolumns,
      "inputSize": self.numOperandCells,
      "threshold": self.numActiveOperandCells,
    }
    if segmentedProximal:
      self.pairLayerProximalConnections = SegmentedForwardModel(
        **self.minicolumnParams)
    else:
      self.pairLayerProximalConnections = ForwardModel(**self.minicolumnParams)

    self.pairParams = {
      "columnCount": numMinicolumns,
      "initialPermanence": 1.0,
      "cellsPerColumn": cellsPerColumn,
      "basalInputSize": self.numOperandCells,
      "activationThreshold": self.numActiveOperandCells,
      "minThreshold": self.numActiveOperandCells,
    }
    self.pairLayer = ApicalTiebreakPairMemory(**self.pairParams)

    self.poolingParams = {
      "cellCount": self.numResultCells,
      "inputSize": self.pairLayer.numberOfCells(),
      "threshold": poolingThreshold,
    }
    if segmentedPooling:
      self.poolingLayer = SegmentedForwardModel(**self.poolingParams)
    else:
      self.poolingLayer = ForwardModel(**self.poolingParams)


  def train(self):
    """
    Train the pair layer and pooling layer.
    """
    for iDriving, cDriving in enumerate(self.drivingOperandSDRs):
      minicolumnSDR = self.minicolumnSDRs[iDriving]
      self.pairLayerProximalConnections.associate(minicolumnSDR, cDriving)
      for iContext, cContext in enumerate(self.contextOperandSDRs):
        iResult = (iContext + iDriving) % self.numLocations
        cResult = self.resultSDRs[iResult]
        self.pairLayer.compute(minicolumnSDR, basalInput=cContext)
        cPair = self.pairLayer.getWinnerCells()
        self.poolingLayer.associate(cResult, cPair)


  def trainWithSpecificPairSDRs(self, pairLayerContexts):
    """
    Train the pair layer and pooling layer, manually choosing which contexts
    each cell will encode (i.e. the pair layer's distal connections).

    @param pairLayerContexts (list of lists of lists of ints)
    iContext integers for each cell, grouped by minicolumn. For example,
      [[[1, 3], [2,4]],
       [[1, 2]]]
    would specify that cell 0 connects to location 1 and location 3, while cell
    1 connects to locations 2 and 4, and cell 2 (in the second minicolumn)
    connects to locations 1 and 2.
    """
    # Grow basal segments in the pair layer.
    for iMinicolumn, contextsByCell in enumerate(pairLayerContexts):
      for iCell, cellContexts in enumerate(contextsByCell):
        iCellAbsolute = iMinicolumn*self.pairLayer.getCellsPerColumn() + iCell
        for context in cellContexts:
          segments = self.pairLayer.basalConnections.createSegments(
            [iCellAbsolute])
          self.pairLayer.basalConnections.growSynapses(
            segments, self.contextOperandSDRs[context], 1.0)

    # Associate the pair layer's minicolumn SDRs with offset cell SDRs,
    # and associate the pooling layer's location SDRs with a pool of pair SDRs.
    for iDriving, cDriving in enumerate(self.drivingOperandSDRs):
      minicolumnSDR = self.minicolumnSDRs[iDriving]
      self.pairLayerProximalConnections.associate(minicolumnSDR, cDriving)

      for iContext, cContext in enumerate(self.contextOperandSDRs):
        iResult = (iContext + iDriving) % self.numLocations
        cResult = self.resultSDRs[iResult]
        cPair = [
          iMinicolumn*self.pairLayer.getCellsPerColumn() + iCell
          for iMinicolumn in minicolumnSDR
          for iCell, cellContexts in enumerate(pairLayerContexts[iMinicolumn])
          if iContext in cellContexts]
        assert len(cPair) == len(minicolumnSDR)

        self.poolingLayer.associate(cResult, cPair)


  def testInferenceOnUnions(self, unionSize, numTests=300):
    """
    Select a random driving operand and a random union of context operands.
    Test how well outputs a union of results.

    Perform the test multiple times with different random selections.
    """
    additionalSDRCounts = []

    for _ in xrange(numTests):
      iContexts = random.sample(xrange(self.numLocations), unionSize)
      iDriving = random.choice(xrange(self.numLocations))
      cDriving = self.drivingOperandSDRs[iDriving]

      cContext = np.unique(np.concatenate(
        [self.contextOperandSDRs[iContext]
         for iContext in iContexts]))
      cResultExpected = np.unique(np.concatenate(
        [self.resultSDRs[(iContext + iDriving) % self.numLocations]
         for iContext in iContexts]))

      self.pairLayerProximalConnections.infer(cDriving)
      minicolumnSDR = self.pairLayerProximalConnections.activeCells
      assert minicolumnSDR.size == self.numActiveMinicolumns

      self.pairLayer.compute(minicolumnSDR, basalInput=cContext, learn=False)
      self.poolingLayer.infer(self.pairLayer.getActiveCells())

      assert np.all(np.in1d(cResultExpected, self.poolingLayer.activeCells))

      additionalSDRCounts.append(
        np.setdiff1d(self.poolingLayer.activeCells,
                     cResultExpected).size / self.numActiveResultCells
      )

    return additionalSDRCounts


def runExperiment(n, w, threshold, cellsPerColumn, folder, numTrials=5,
                  cleverTMSDRs=False):
  """
  Run a PoolOfPairsLocation1DExperiment various union sizes.
  """
  if not os.path.exists(folder):
    try:
      os.makedirs(folder)
    except OSError:
      # Multiple parallel tasks might create the folder. That's fine.
      pass

  filename = "{}/n_{}_w_{}_threshold_{}_cellsPerColumn_{}.json".format(
    folder, n, w, threshold, cellsPerColumn)
  if len(glob.glob(filename)) == 0:
    print("Starting: {}/n_{}_w_{}_threshold_{}_cellsPerColumn_{}".format(
      folder, n, w, threshold, cellsPerColumn))

    result = defaultdict(list)
    for _ in xrange(numTrials):

      exp = PoolOfPairsLocation1DExperiment(**{
        "numMinicolumns": n,
        "numActiveMinicolumns": w,
        "poolingThreshold": threshold,
        "cellsPerColumn": cellsPerColumn,
        "minicolumnSDRs": generateMinicolumnSDRs(n=n, w=w, threshold=threshold),
      })

      if cleverTMSDRs:
        exp.trainWithSpecificPairSDRs(carefullyCollideContexts(
          numContexts=25, numCells=cellsPerColumn, numMinicolumns = n))
      else:
        exp.train()

      for unionSize in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]:
        additionalSDRCounts = exp.testInferenceOnUnions(unionSize)

        result[unionSize] += additionalSDRCounts

    with open(filename, "w") as fOut:
      json.dump(sorted(result.items(), key=lambda x: x[0]),
                fOut)
      print("Wrote:", filename)


def doGenerateMinicolumnSDRs(args):
  return generateMinicolumnSDRs(*args)


def doRunExperiment(kwargs):
  return runExperiment(**kwargs)


def run(allParams, parallel=True):
  if parallel:
    # Some experiments might use the same minicolumn SDRs, so generate them
    # first.
    multiprocessing.Pool().map(doGenerateMinicolumnSDRs,
                               set((params["n"], params["w"],
                                    params["threshold"])
                                   for params in allParams))
    multiprocessing.Pool().map(doRunExperiment, allParams)
  else:
    for params in allParams:
      runExperiment(**params)



if __name__ == "__main__":
  allParams = (
    # Especially good parameters for <= 100 cells.
    [{"n": 16,
      "w": 11,
      "threshold": 9,
      "cellsPerColumn": 6,
      "folder": "data/default"}]

    # Good parameters for ~100, 150, ~200, and ~250 cells.
    +
    [{"n": 15,
      "w": 10,
      "threshold": 8,
      "cellsPerColumn": cellsPerColumn,
      "folder": "data/default"}
     for cellsPerColumn in [7, 10, 13, 16]]

    # Vary the density.
    +
    [{"n": 15,
      "w": w,
      "threshold": threshold,
      "cellsPerColumn": 10,
      "folder": "data/default"}
     for w, threshold in [(3, 2), (4, 3), (5, 4), (10, 8)]]

    # Experiment with improving the TM SDRs
    +
    [{"n": 15,
      "w": 10,
      "threshold": 8,
      "cellsPerColumn": cellsPerColumn,
      "cleverTMSDRs": True,
      "folder": "data/improved-tm-sdrs"}
     for cellsPerColumn in [7, 10, 13, 16]]
  )

  run(allParams, parallel=True)

  print("View these results by opening 'charts.ipynb' in Jupyter Notebook.")
