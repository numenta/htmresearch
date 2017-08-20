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
Functions that generate and cache SDRs on the filesystem, and a standalone
SDR generator that can run from the command line.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import json
import multiprocessing
import os
import random
import time

import numpy as np

from htmresearch_core.experimental import enumerateDistantSDRsBruteForce


def generateMinicolumnSDRs(n, w, threshold):
  """
  Wraps enumerateDistantSDRsBruteForce, caching its result on the filesystem.
  """
  if not os.path.exists("sdrs"):
    os.makedirs("sdrs")

  filename = "sdrs/{}_{}_{}.json".format(n, w, threshold)

  if len(glob.glob(filename)) > 0:
    with open(filename, "r") as fIn:
      sdrs = json.load(fIn)
  else:
    begin = time.time()
    sdrs = enumerateDistantSDRsBruteForce(n, w, threshold)
    end = time.time()
    with open(filename, "w") as fOut:
      json.dump([sdr.tolist() for sdr in sdrs], fOut)
      print("Saved", filename)
    print("Elapsed time: {:.2f} seconds".format(end - begin))

  return sdrs


def createEvenlySpreadSDRs(numSDRs, n, w):
  """
  Return a set of ~random SDRs that use every available bit
  an equal number of times, +- 1.
  """
  assert w <= n

  available = np.arange(n)
  np.random.shuffle(available)

  SDRs = []

  for _ in xrange(numSDRs):
    selected = available[:w]
    available = available[w:]

    if available.size == 0:
      remainderSelected = np.random.choice(
        np.setdiff1d(np.arange(n), selected),
        size=(w - selected.size),
        replace= False)
      selected = np.append(selected, remainderSelected)

      available = np.setdiff1d(np.arange(n), remainderSelected)
      np.random.shuffle(available)

    selected.sort()
    SDRs.append(selected)

  return SDRs


def carefullyCollideContexts(numContexts, numCells, numMinicolumns):
  """
  Use a greedy algorithm to choose how each minicolumn should distribute
  contexts between its cells.

  @return (list of lists of lists of ints)
  iContext integers for each cell, grouped by minicolumn. For example,
    [[[1, 3], [2,4]],
     [[1, 2]]]
  would specify that cell 0 connects to location 1 and location 3, while cell
  1 connects to locations 2 and 4, and cell 2 (in the second minicolumn)
  connects to locations 1 and 2.
  """
  minicolumns = []

  for _ in xrange(numMinicolumns):
    contextsForCell = [set() for _ in xrange(numCells)]

    contexts = range(numContexts)
    random.shuffle(contexts)

    while len(contexts) > 0:
      eligibleCells = range(len(contextsForCell))
      while len(contexts) > 0 and len(eligibleCells) > 0:
        candidateAdditions = [(context, cell)
                              for context in contexts
                              for cell in eligibleCells]

        # How many new duplicate collisions will come from this addition?
        #
        # For every other context in on this cell, check how many times this
        # pair occurs elsewhere.
        badness = [sum(sum(1 if (context in otherCellContexts and
                                 otherContext in otherCellContexts) else 0
                           for minicolumn in minicolumns
                           for otherCellContexts in minicolumn)
                       for otherContext in contextsForCell[cell])
                   for context, cell in candidateAdditions]

        selectedContext, selectedCell = candidateAdditions[
          badness.index(min(badness))]

        contextsForCell[selectedCell].add(selectedContext)
        eligibleCells.remove(selectedCell)
        contexts.remove(selectedContext)

    minicolumns.append(contextsForCell)

  return minicolumns


def doGenerateMinicolumnSDRs(kwargs):
  print("Starting:", kwargs)
  sdrs = generateMinicolumnSDRs(**kwargs)
  print("Finished: {}. Result: {} SDRs".format(kwargs, len(sdrs)))


if __name__ == "__main__":
  # Change these, then run "python generate_sdrs.py" to generate SDRs and print
  # the number of generated SDRs.
  allParams = [
    {"n": 15, "w": 10, "threshold": 8},
    {"n": 16, "w": 11, "threshold": 9},
  ]

  multiprocessing.Pool().map(doGenerateMinicolumnSDRs, allParams)
