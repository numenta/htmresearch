#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
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

import unittest
import numpy
from nupic.research.spatial_pooler import SpatialPooler
from sensorimotor.spatial_pooler_monitor_mixin import (
  SpatialPoolerMonitorMixin)

class MonitoredSpatialPooler(SpatialPoolerMonitorMixin, SpatialPooler): pass

class SpatialPoolerMonitorMixinTest(unittest.TestCase):

  VERBOSITY = 1


  def setUp(self):
    # Initialize the spatial pooler
    self.sp = MonitoredSpatialPooler(inputDimensions=(15,),
                       columnDimensions=(4,),
                       potentialRadius=15,
                       numActiveColumnsPerInhArea=1,
                       globalInhibition=True,
                       synPermActiveInc=0.03,
                       potentialPct=1.0)

  def testGetActiveColumn(self):
    # test whether the active column indices are correctly stored

    cat = numpy.array( [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='uint8')
    dog = numpy.array( [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype='uint8')
    rat = numpy.array( [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype='uint8')
    bat = numpy.array( [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype='uint8')

    output = numpy.zeros((4,), dtype="int")
    activeColumns = numpy.zeros((400,4), dtype="int")
    for i in xrange(100):
        self.sp.compute(cat, learn=True, activeArray=output)
        activeColumns[4*i+0] = output
        self.sp.compute(dog, learn=True, activeArray=output)
        activeColumns[4*i+1] = output
        self.sp.compute(rat, learn=True, activeArray=output)
        activeColumns[4*i+2] = output
        self.sp.compute(bat, learn=True, activeArray=output)
        activeColumns[4*i+3] = output

    for i in range(400):
        self.assertTrue(activeColumns[i][self.sp.mmGetTraceActiveCells().data[i]]==1)

    self.assertTrue(self.sp.mmGetTraceConnectionCounts().data[-1]==3*4)


  def testGetActiveDutyCycles(self):
    # test whether active duty cycle are calculated correctly

    cat = numpy.array( [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='uint8')
    dog = numpy.array( [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype='uint8')
    rat = numpy.array( [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype='uint8')
    bat = numpy.array( [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype='uint8')

    output = numpy.zeros((4,), dtype="int")
    for i in xrange(100):
        self.sp.compute(cat, learn=True, activeArray=output)
        self.sp.compute(dog, learn=True, activeArray=output)
        self.sp.compute(rat, learn=True, activeArray=output)
        self.sp.compute(bat, learn=True, activeArray=output)


    self.assertTrue(numpy.sum(self.sp.mmGetTraceActiveDutyCycles().data) == 400)
    self.sp.mmClearHistory()

    output = numpy.zeros((4,), dtype="int")
    for i in xrange(100):
        self.sp.compute(cat, learn=False, activeArray=output)
        self.sp.compute(dog, learn=False, activeArray=output)
        self.sp.compute(rat, learn=False, activeArray=output)
        self.sp.compute(bat, learn=False, activeArray=output)






  def _printInformation(self):
    print self.sp.mmGetTraceActiveDutyCycles().data
    pass

if __name__ == "__main__":
  unittest.main()
