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

Utility functions

"""

from __future__ import print_function

import numpy as np

from nupic.encoders.base import defaultDtype
from nupic.encoders.coordinate import CoordinateEncoder


def createLocationEncoder(t, w=15):
  """
  A default coordinate encoder for encoding locations into sparse
  distributed representations.
  """
  encoder = CoordinateEncoder(name="positionEncoder", n=t.l6CellCount, w=w)
  return encoder


def encodeLocation(encoder, x, y, output, radius=5):
  # Radius of 7 or 8 gives an overlap of about 8 or 9 with neighoring pixels
  # Radius of 5 about 3
  encoder.encodeIntoArray((np.array([x * radius, y * radius]), radius), output)
  return output.nonzero()[0]


def trainThalamusLocations(t, encoder):
  print("Training TRN cells on location SDRs")
  output = np.zeros(encoder.getWidth(), dtype=defaultDtype)

  # Train the TRN cells to respond to SDRs representing locations
  for y in range(0, t.trnHeight):
    for x in range(0, t.trnWidth):
      t.learnL6Pattern(encodeLocation(encoder, x, y, output),
                       [(x, y)])


def getUnionLocations(encoder, x, y, r, step=1):
  """
  Return a union of location encodings that correspond to the union of all locations
  within the specified circle.
  """
  output = np.zeros(encoder.getWidth(), dtype=defaultDtype)
  locations = set()
  for dx in range(-r, r+1, step):
    for dy in range(-r, r+1, step):
      if dx*dx + dy*dy <= r*r:
        e = encodeLocation(encoder, x+dx, y+dy, output)
        locations = locations.union(set(e))

  return locations


def trainThalamusLocationsTMP(t, encoder, windowSize=5):
  print("Training TRN cells on location SDRs")
  output = np.zeros(encoder.getWidth(), dtype=defaultDtype)

  # Train the TRN cells to respond to SDRs representing locations
  for wy in range(0, t.trnHeight):
    print(wy)
    for wx in range(0, t.trnWidth):
      e = encodeLocation(encoder, wx, wy, output)
      for x in range(wx-windowSize, wx+windowSize):
        for y in range(wy - windowSize, wy + windowSize):
          if x >= 0 and x < t.trnWidth and y >= 0 and y < t.trnHeight:
            t.learnL6Pattern(e, [(x, y)])

