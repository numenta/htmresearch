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

"""Use principle components of correlation matrix to find grid cell
weights to place cell inputs.
"""

import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def computeCorrelation(history):
  correlation = np.zeros((625, 625))
  for active in history:
    for element in active:
      for coactive in active:
        correlation[element][coactive] += 1
  return correlation


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
  #active = set(world[x-2:x+2,y-2:y+2].flatten())
  active = set()
  for i in xrange(x - 2, x + 2 + 1):
    for j in xrange(y - 2, y + 2 + 1):
      active.add(world[i % 25, j % 25])
  return active


def main():
  x = 10
  y = 10
  steps = 100000
  history = []
  world = np.array([i for i in xrange(625)])
  world.resize((25, 25))
  for _ in xrange(steps):
    active = getActive(world, x, y)
    assert len(active) == 25, "{}, {}: {}".format(x, y, active)
    history.append(active)
    x, y = getNewLocation(x, y, 25, 2, False)
  correlation = computeCorrelation(history)

  #plt.imshow(correlation, cmap="hot", interpolation="nearest")
  #plt.show()

  pca = PCA(n_components=50)
  pca.fit(correlation)
  print 'components'
  print pca.components_
  #negativeMask = (pca.components_ < 0)
  #pca.components_[negativeMask] = 0
  print 'transform:'
  transform = pca.transform(correlation)
  #negativeMask = (transform < 0)
  #transform[negativeMask] = 0
  print transform.shape

  for i in [i * 2 for i in xrange(25)]:
    plt.imshow(transform[:,i].reshape((25, 25)), cmap="hot", interpolation="nearest")
    plt.show()


if __name__ == "__main__":
  main()
