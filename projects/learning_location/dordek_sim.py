# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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

"""Use Hebbian learning over place cell inputs to learn grid cells."""

import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF as PCA

from nupic.algorithms.spatial_pooler import SpatialPooler


def getNewLocation(x, y, width, viewDistance, wrap):
  options = []
  if x > viewDistance or wrap:
    options.append(((x - 1) % width, y))
  if x < width - viewDistance - 1 or wrap:
    options.append(((x + 1) % width, y))
  if y > viewDistance or wrap:
    options.append((x, (y - 1) % width))
  if y < width - viewDistance - 1 or wrap:
    options.append((x, (y + 1) % width))
  return options[int(random.random() * len(options))]


def getActive(world, x, y):
  active = set()
  for i in xrange(x - 2, x + 2 + 1):
    for j in xrange(y - 2, y + 2 + 1):
      active.add(world[i % 25, j % 25])
  return active


def main():
  x = 10
  y = 10
  steps = 10000
  history = []
  world = np.array([i for i in xrange(625)])
  world.resize((25, 25))
  sp = SpatialPooler(
      inputDimensions=(625,),
      columnDimensions=(25,),
      potentialRadius=625,
      numActiveColumnsPerInhArea=1,
  )
  output = np.zeros((25,), dtype=np.uint32)
  for _ in xrange(steps):
    active = getActive(world, x, y)
    assert len(active) == 25, "{}, {}: {}".format(x, y, active)
    activeInput = np.zeros((625,), dtype=np.uint32)
    for v in active:
      activeInput[v] = 1
    history.append(active)
    sp.compute(activeInput, True, output)
    x, y = getNewLocation(x, y, 25, 2, False)

  for i in xrange(25):
    permanence = np.zeros((625,))
    sp.getPermanence(i, permanence)
    plt.imshow(permanence.reshape((25, 25)), cmap="hot", interpolation="nearest")
    plt.show()


if __name__ == "__main__":
  main()
