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

"""Use Hebbian learning over place cell inputs fed through
center-surround cells to learn grid cells. The center-surround output
looks very similar to the input so it doesn't have much impact.
"""

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


def generateCenterSurroundFields():
  fields = []
  for i in xrange(25 - 5 + 1):
    for j in xrange(25 - 5 + 1):
      center = np.zeros((25, 25), dtype=np.bool)
      center[i:i+5, j:j+5] = 1
      sr = 3
      surround = np.zeros((25, 25), dtype=np.bool)
      surround[max(0, i-sr):i+5+sr+1, max(0, j-sr):j+5+sr+1] = 1
      surround[i:i+5, j:j+5] = 0
      fields.append((center.flatten(), surround.flatten()))
  return fields


def processCenterSurround(fields, activeInput):
  return np.array(
      [activeInput[c].sum() > activeInput[s].sum()
       for c, s in fields], dtype=np.uint32)


def main():
  x = 10
  y = 10
  steps = 10000
  world = np.array([i for i in xrange(625)])
  world.resize((25, 25))
  spInputSize = 21*21
  sp = SpatialPooler(
      inputDimensions=(spInputSize,),
      columnDimensions=(25,),
      potentialRadius=spInputSize,
      numActiveColumnsPerInhArea=1,
      synPermActiveInc=0.1,
      synPermInactiveDec=0.5,
      boostStrength=1.0,
  )
  csFields = generateCenterSurroundFields()
  output = np.zeros((25,), dtype=np.uint32)
  for _ in xrange(steps):
    active = getActive(world, x, y)
    assert len(active) == 25, "{}, {}: {}".format(x, y, active)
    activeInput = np.zeros((625,), dtype=np.uint32)
    for v in active:
      activeInput[v] = 1
    centerSurround = processCenterSurround(csFields, activeInput)
    print centerSurround

    sp.compute(centerSurround, True, output)
    x, y = getNewLocation(x, y, 25, 2, False)

  for i in xrange(25):
    permanence = np.zeros((spInputSize,))
    sp.getPermanence(i, permanence)
    plt.imshow(permanence.reshape((21, 21)), cmap="hot", interpolation="nearest")
    plt.show()


if __name__ == "__main__":
  main()
