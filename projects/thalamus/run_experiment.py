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

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from htmresearch.frameworks.thalamus.thalamus import Thalamus
from htmresearch.frameworks.thalamus.thalamus_utils import *


def plotActivity(activity, filename,
                 title="",
                 cmap="Greys"):
  plt.imshow(activity, vmin=0.0, vmax=2.0, origin="upper", cmap=cmap)
  plt.title(title)
  plt.colorbar()
  plt.savefig(os.path.join("images", filename))
  plt.close()


def inferThalamus(t, l6Input, ffInput):
  """
  Compute the effect of this feed forward input given the specific L6 input.

  :param t: instance of Thalamus
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

  encoder = createLocationEncoder(t)

  trainThalamusLocations(t, encoder)

  output = np.zeros(encoder.getWidth(), dtype=defaultDtype)
  ff = np.zeros((32, 32))
  for x in range(10,20):
    ff[:] = 0
    ff[10:20, 10:20] = 1
    plotActivity(ff, "square_ff_input.jpg", title="Feed forward input")
    inferThalamus(t, encodeLocation(encoder, x, x, output), ff)
    plotActivity(ff, "square_relay_output_" + str(x) + ".jpg",
                 title="Relay cell activity",
                 cmap="coolwarm")

  # Show attention with an A
  ff = np.zeros((32, 32))
  for x in range(10,20):
    ff[:] = 0
    ff[10, 10:20] = 1
    ff[15, 10:20] = 1
    ff[10:20, 10] = 1
    ff[10:20, 20] = 1
    plotActivity(ff, "A_ff_input.jpg", title="Feed forward input")
    inferThalamus(t, encodeLocation(encoder, x, x, output), ff)
    plotActivity(t.burstReadyCells, "relay_burstReady_" + str(x) + ".jpg",
                 title="Burst-ready cells (x,y)=({},{})".format(x, x),
                 )
    plotActivity(ff, "A_relay_output_" + str(x) + ".jpg",
                 title="Relay cell activity",
                 cmap="coolwarm")


# Simple tests for debugging
def trainThalamus(t):
  # Learn
  t.learnL6Pattern([0, 1, 2, 3, 4, 5], [(0, 0), (2, 3)])
  t.learnL6Pattern([6, 7, 8, 9, 10], [(1, 1), (3, 4)])


def basicTest():
  t = Thalamus()

  trainThalamus(t)

  ff = np.zeros((32,32))
  ff.reshape(-1)[[8, 9, 98, 99]] = 1.0
  inferThalamus(t, [0, 1, 2, 3, 4, 5], ff)

  encoder = createLocationEncoder(t)
  output = np.zeros(encoder.getWidth(), dtype=defaultDtype)

  # Positions go from 0 to 1000 in both x and y directions
  encoder.encodeIntoArray((np.array([100, 200]), 10), output)


if __name__ == '__main__':

  # basicTest()
  locationsTest()
