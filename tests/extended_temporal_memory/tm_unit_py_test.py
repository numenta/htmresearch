#!/usr/bin/env python
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

import copy
import unittest

from htmresearch.algorithms.extended_temporal_memory import (
    ExtendedTemporalMemory)
from tm_unit_test_base import TemporalMemoryUnitTest


try:
  import capnp
except ImportError:
  capnp = None
if capnp:
  from nupic.proto import TemporalMemoryProto_capnp


class TemporalMemoryUnitTestPy(TemporalMemoryUnitTest, unittest.TestCase):


  def getTMClass(self):
    return ExtendedTemporalMemory


  def testConnectionsNeverChangeWhenLearningDisabled(self):
    """
    Only do this test on the Python ETM because deepcopy doesn't work on SWIG
    objects.
    """
    tm = self.constructTM(
      columnDimensions=[32],
      cellsPerColumn=4,
      activationThreshold=3,
      initialPermanence=.2,
      connectedPermanence=.50,
      minThreshold=2,
      maxNewSynapseCount=4,
      permanenceIncrement=.10,
      permanenceDecrement=.10,
      predictedSegmentDecrement=0.02,
      seed=42)

    prevActiveColumns = [0]
    prevActiveCells = [0, 1, 2, 3]
    activeColumns = [1, 2] #1 is predicted, 2 is bursting
    prevInactiveCell = 81
    expectedActiveCells = [4]

    correctActiveSegment = tm.basalConnections.createSegment(expectedActiveCells[0])
    tm.basalConnections.createSynapse(correctActiveSegment, prevActiveCells[0], .5)
    tm.basalConnections.createSynapse(correctActiveSegment, prevActiveCells[1], .5)
    tm.basalConnections.createSynapse(correctActiveSegment, prevActiveCells[2], .5)

    wrongMatchingSegment = tm.basalConnections.createSegment(43)
    tm.basalConnections.createSynapse(wrongMatchingSegment, prevActiveCells[0], .5)
    tm.basalConnections.createSynapse(wrongMatchingSegment, prevActiveCells[1], .5)
    tm.basalConnections.createSynapse(wrongMatchingSegment, prevInactiveCell, .5)

    before = copy.deepcopy(tm.basalConnections)

    tm.compute(prevActiveColumns, learn=False)
    tm.compute(activeColumns, learn=False)

    self.assertEqual(before, tm.basalConnections)
