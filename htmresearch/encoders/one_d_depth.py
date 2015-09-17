# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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
from nupic.encoders.base import Encoder
from nupic.encoders.scalar import ScalarEncoder



class OneDDepthEncoder(Encoder):
  """
  Given an array of numbers, each representing distance to the closest object,
  returns an SDR representation of that depth data.

  At each given position, computes the closest distance within radius 3, and
  encodes that distance with a scalar encoder. The concatenation of all these
  scalar encodings is the final encoding.
  """

  def __init__(self,
               positions=range(36),
               radius=3,
               wrapAround=False,
               nPerPosition=57,
               wPerPosition=3,
               minVal=0,
               maxVal=1,
               name=None,
               verbosity=0):
    """
    See `nupic.encoders.base.Encoder` for more information.

    @param positions    (list) Positions at which to encode distance
    @param radius       (int)  Radius of positions over which to consider to get closest distance for encoding
    @param wrapAround   (bool) Whether radius should wrap around the sides of the input array
    @param nPerPosition (int)  Number of bits available for scalar encoder when encoding each position
    @param wPerPosition (int)  Number of bits active for scalar encoder when encoding each position
    @param minVal       (int)  Minimum distance that can be encoded
    @param maxVal       (int)  Maximum distance that can be encoded
    """
    self.positions = positions
    self.radius = radius
    self.wrapAround = wrapAround
    self.scalarEncoder = ScalarEncoder(wPerPosition, minVal, maxVal,
                                       n=nPerPosition,
                                       forced=True)
    self.verbosity = verbosity
    self.encoders = None

    self.n = len(self.positions) * nPerPosition
    self.w = len(self.positions) * wPerPosition

    if name is None:
      name = "[%s:%s]" % (self.n, self.w)
    self.name = name


  def getWidth(self):
    """See `nupic.encoders.base.Encoder` for more information."""
    return self.n


  def getDescription(self):
    """See `nupic.encoders.base.Encoder` for more information."""
    return [('data', 0)]


  def getScalars(self, inputData):
    """See `nupic.encoders.base.Encoder` for more information."""
    return numpy.array([0]*len(inputData))


  def encodeIntoArray(self, inputData, output):
    """
    See `nupic.encoders.base.Encoder` for more information.

    @param inputData (tuple) Contains depth data (numpy.array)
    @param output (numpy.array) Stores encoded SDR in this numpy array
    """
    output[:] = 0

    for i, position in enumerate(self.positions):
      indices = range(position-self.radius, position+self.radius+1)
      mode = 'wrap' if self.wrapAround else 'clip'
      values = inputData.take(indices, mode=mode)
      start = i * self.scalarEncoder.getWidth()
      end = (i + 1) * self.scalarEncoder.getWidth()
      output[start:end] = self.scalarEncoder.encode(max(values))


  def dump(self):
    print "OneDDepthEncoder:"
    print "  w:   %d" % self.w
    print "  n:   %d" % self.n


  @classmethod
  def read(cls, proto):
    encoder = object.__new__(cls)
    encoder.w = proto.w
    encoder.n = proto.n
    encoder.radius = proto.radius
    encoder.verbosity = proto.verbosity
    encoder.name = proto.name
    return encoder


  def write(self, proto):
    proto.w = self.w
    proto.n = self.n
    proto.radius = self.radius
    proto.verbosity = self.verbosity
    proto.name = self.name
