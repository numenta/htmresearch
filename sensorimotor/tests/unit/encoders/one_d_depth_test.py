#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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

import numpy
import unittest

from sensorimotor.encoders.one_d_depth import OneDDepthEncoder



class OneDDepthEncoderTest(unittest.TestCase):
  """Unit tests for OneDDepthEncoder class"""

  def setUp(self):
    self.encoder = OneDDepthEncoder(name="one_d_depth",
                                    positions=range(36),
                                    radius=3,
                                    wrapAround=False,
                                    nPerPosition=57,
                                    wPerPosition=3,
                                    minVal=0,
                                    maxVal=1)


  def testEncodeUniform(self):
    inputData = numpy.array([0.5] * len(self.encoder.positions))
    outputData = self.encoder.encode(inputData)
    self._printInputOutput(inputData, outputData)


  def testEncodeSingleSpikeCenter(self):
    inputData = numpy.array([0.0] * len(self.encoder.positions))
    inputData[int(len(self.encoder.positions) / 2)] = 0.75
    outputData = self.encoder.encode(inputData)
    self._printInputOutput(inputData, outputData)


  def testEncodeSingleSpikeLeft(self):
    inputData = numpy.array([0.0] * len(self.encoder.positions))
    inputData[0] = 0.75
    outputData = self.encoder.encode(inputData)
    self._printInputOutput(inputData, outputData)


  def testEncodeSingleSpikeLeftWrapAround(self):
    self.encoder.wrapAround = True
    inputData = numpy.array([0.0] * len(self.encoder.positions))
    inputData[0] = 0.75
    outputData = self.encoder.encode(inputData)
    self._printInputOutput(inputData, outputData)


  def _printInputOutput(self, inputData, outputData):
    print self.id()
    print "=================================="
    print "Input data:"
    print inputData.tolist()
    print
    print "Output data:"
    self._printOutputData(outputData)
    print


  def _printOutputData(self, outputData):
    shape = len(self.encoder.positions), self.encoder.scalarEncoder.getWidth()
    for i in outputData.reshape(shape).tolist():
      print i



if __name__ == "__main__":
  unittest.main()
