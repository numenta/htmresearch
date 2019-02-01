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

def trainThalamus(t):
  # Learn
  t.learnL6Pattern([0, 1, 2, 3, 4, 5], [(0, 0), (2, 3)])
  t.learnL6Pattern([6, 7, 8, 9, 10], [(1, 1), (3, 4)])


def plotActivity(activity, filename):
  plt.imshow(activity)
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


def basicTest():
  t = Thalamus()

  trainThalamus(t)

  ff = np.zeros((32,32))
  ff.reshape(-1)[[8, 9, 98, 99]] = 1.0
  testThalamus(t, [0, 1, 2, 3, 4, 5], ff)
  plotActivity(ff, "temp.jpg")

  # ff = np.zeros((32,32))
  # testThalamus(t, [6, 7, 8, 9, 10], ff)
  #
  # # Should do nothing
  # ff = np.zeros((32,32))
  # testThalamus(t, [1, 2, 3, 6, 7, 8], ff)
  #

if __name__ == '__main__':

  basicTest()
