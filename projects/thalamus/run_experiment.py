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

from __future__ import print_function

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from htmresearch.frameworks.thalamus.thalamus import Thalamus

from nupic.encoders.base import defaultDtype
from nupic.encoders.coordinate import CoordinateEncoder


def trainThalamus(t):
  # Learn
  t.learnL6Pattern([0, 1, 2, 3, 4, 5], [(0, 0), (2, 3)])
  t.learnL6Pattern([6, 7, 8, 9, 10], [(1, 1), (3, 4)])


def getLocationSDR(encoder, x, y, output):
  radius = 5
  encoder.encodeIntoArray((np.array([x * radius, y * radius]), radius), output)
  return output.nonzero()[0]


def trainThalamusLocations(t):
  encoder = CoordinateEncoder(name="positionEncoder", n=t.l6CellCount, w=15)
  output = np.zeros(encoder.getWidth(), dtype=defaultDtype)

  # Train the TRN cells to respond to SDRs representing locations
  for y in range(0, t.trnHeight):
    for x in range(0, t.trnWidth):
      t.learnL6Pattern(getLocationSDR(encoder, x, y, output),
                       [(x, y)])


def plotActivity(activity, filename):
  plt.imshow(activity, vmin=0.0, vmax=2.0)
  plt.colorbar()
  plt.savefig(filename)
  plt.close()


def testThalamus(t, l6Input, ffInput):
  """

  :param t:
  :param l6Input:
  :param ffInput: a numpy array of 0's and 1's
  :return:
  """
  print("\n-----------")
  t.reset()
  t.deInactivateCells(l6Input)
  t.computeFeedForwardActivity(ffInput)
  print("L6 input:", l6Input)
  print("Active TRN cells: ", t.activeTRNCellIndices)
  print("Burst ready relay cells: ", t.burstReadyCellIndices)


def locationsTest():
  t = Thalamus()

  trainThalamusLocations(t)

  encoder = CoordinateEncoder(name="positionEncoder", n=t.l6CellCount, w=15)
  output = np.zeros(encoder.getWidth(), dtype=defaultDtype)


  ff = np.zeros((32, 32))
  ff[10:20, 10:20] = 1
  plotActivity(ff, "ff_input.jpg")
  testThalamus(t, getLocationSDR(encoder, 16, 16, output), ff)
  plotActivity(ff, "relay_output1.jpg")

  ff = np.zeros((32, 32))
  ff[10:20, 10:20] = 1
  testThalamus(t, getLocationSDR(encoder, 17, 17, output), ff)
  plotActivity(ff, "relay_output2.jpg")

  ff = np.zeros((32, 32))
  ff[10:20, 10:20] = 1
  testThalamus(t, getLocationSDR(encoder, 18, 18, output), ff)
  plotActivity(ff, "relay_output3.jpg")


def basicTest():
  t = Thalamus()

  trainThalamus(t)

  ff = np.zeros((32,32))
  ff.reshape(-1)[[8, 9, 98, 99]] = 1.0
  testThalamus(t, [0, 1, 2, 3, 4, 5], ff)

  encoder = CoordinateEncoder(name="positionEncoder", n=1024, w=15)
  output = np.zeros(encoder.getWidth(), dtype=defaultDtype)

  # Positions go from 0 to 1000 in both x and y directions
  encoder.encodeIntoArray((np.array([100, 200]), 10), output)

  # ff = np.zeros((32,32))
  # testThalamus(t, [6, 7, 8, 9, 10], ff)
  #
  # # Should do nothing
  # ff = np.zeros((32,32))
  # testThalamus(t, [1, 2, 3, 6, 7, 8], ff)
  #

if __name__ == '__main__':

  # basicTest()
  locationsTest()
