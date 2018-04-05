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

import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from sklearn.decomposition import PCA

from nupic.algorithms.spatial_pooler import SpatialPooler

CWD = os.path.dirname(os.path.realpath(__file__))


def computeCorrelation(history, ws):
  correlation = np.zeros((ws, ws))
  for active in history:
    for element in active:
      for coactive in active:
        correlation[element][coactive] += 1
  return correlation


def getNewLocation(x, y, width, viewDistance, wrap, locationHeatmap, stepSize, jumpProb, direction, directionStability):
  if random.random() < jumpProb:
    if wrap:
      border = 0
    else:
      border = viewDistance
    options = []
    for i in xrange(locationHeatmap.shape[0]):
      for j in xrange(locationHeatmap.shape[1]):
        if (i >= viewDistance and i < (width - viewDistance) and
            j >= viewDistance and j < (width - viewDistance)):
          options.append((locationHeatmap[i][j], (i, j)))
    options.sort()
    coord = options[0][1]
    return coord + [direction]

  options = set()
  if x - stepSize >= viewDistance or wrap:
    options.add("left")
  if x + stepSize < width - viewDistance or wrap:
    options.add("right")
  if y - stepSize >= viewDistance or wrap:
    options.add("up")
  if y + stepSize < width - viewDistance or wrap:
    options.add("down")

  while True:
    # This is a random value between (-180, +180) scaled by directionStability
    dirChange = (((random.random() * 360.0) - 180.0) *
                 (1.0 - directionStability))
    direction = (direction + dirChange) % 360.0
    if direction >= 135.0 and direction < 225.0:
      movement = "left"
    elif direction >= 315.0 or direction < 45.0:
      movement = "right"
    elif direction >=45.0 and direction < 135.0:
      movement = "up"
    elif direction >= 225.0 and direction < 315.0:
      movement = "down"
    else:
      raise Exception("Invalid angle!!!")

    if movement in options:
      break

  if movement == "left":
    return ((x - stepSize) % width, y, direction)
  elif movement == "right":
    return ((x + stepSize) % width, y, direction)
  elif movement == "up":
    return (x, (y - stepSize) % width, direction)
  else:
    return (x, (y + stepSize) % width, direction)


def getActive(world, ww, x, y, rr):
  active = set()
  for i in xrange(x - rr, x + rr + 1):
    for j in xrange(y - rr, y + rr + 1):
      active.add(world[i % ww, j % ww])
  return active


def scale(m, zero=False):
  if zero:
    minVal = m.min()
    return (m - minVal) / (m.max() - minVal)

  return m / m.max()


def frequencyAnalysis(m):
  fs = np.fft.fft2(m)
  plt.imshow(m)
  plt.show()
  #plt.imshow(np.log(np.abs(np.fft.fftshift(fs))**2))
  plt.imshow(np.abs(np.fft.fftshift(fs)))
  plt.show()


def runTrial(ww, numColumns, potentialPct, inc, dec, mpo, dutyCycle, boost, steps, rr, spW, stimulusThreshold, connected, stepSize, jumpProb, directionStability):
  ws = ww ** 2
  x = 10
  y = 10
  locationHeatmap = np.zeros((ww, ww))
  history = []
  world = np.array([i for i in xrange(ws)])
  world.resize((ww, ww))
  sp = SpatialPooler(
      inputDimensions=(ws,),
      columnDimensions=(numColumns,),
      potentialRadius=ws,
      potentialPct=potentialPct,
      numActiveColumnsPerInhArea=spW,
      stimulusThreshold=stimulusThreshold,
      synPermActiveInc=inc,
      synPermInactiveDec=dec,
      synPermConnected=connected,
      minPctOverlapDutyCycle=mpo,
      dutyCyclePeriod=dutyCycle,
      boostStrength=boost,
      seed=1936,
      globalInhibition=True,
  )
  output = np.zeros((numColumns,), dtype=np.uint32)
  direction = 0
  for i in xrange(steps):
    locationHeatmap[x][y] += 1
    active = getActive(world, ww, x, y, rr)
    history.append(active)
    activeInput = np.zeros((ws,), dtype=np.uint32)
    for v in active:
      activeInput[v] = 1
    sp.compute(activeInput, True, output)
    x, y, direction = getNewLocation(x, y, ww, rr, wrap=True, locationHeatmap=locationHeatmap, stepSize=stepSize, jumpProb=jumpProb, direction=direction, directionStability=directionStability)

    if (i + 1) % 100 == 0:
      saveImage(history, ws, ww, numColumns, locationHeatmap, potentialPct, inc, dec, mpo, dutyCycle, boost, rr, spW, i+1, sp)

  saveImage(history, ws, ww, numColumns, locationHeatmap, potentialPct, inc, dec, mpo, dutyCycle, boost, rr, spW, steps, sp)


def saveImage(history, ws, ww, numColumns, locationHeatmap, potentialPct, inc, dec, mpo, dutyCycle, boost, rr, spW, steps, sp):
  autoCorrelation = computeCorrelation(history, ws)

  pca = PCA(n_components=30)
  pca.fit(autoCorrelation)
  transform = pca.transform(autoCorrelation)

  #for i in xrange(25):
  #  plt.imshow(transform[:,i].reshape((ww, ww)), cmap="hot", interpolation="nearest")
  #  plt.show()

  principleComponents = [
      np.concatenate(
          [scale(transform[:,i*2].reshape((ww, ww)), zero=True)
           for i in xrange(5)]),
      np.concatenate(
          [scale(transform[:,(5+i)*2].reshape((ww, ww)), zero=True)
           for i in xrange(5)]),
      np.concatenate(
          [scale(transform[:,(10+i)*2].reshape((ww, ww)), zero=True)
           for i in xrange(5)]),
  ]

  #frequencyAnalysis(transform[:,13].reshape((ww, ww)))
  #frequencyAnalysis(transform[:,14].reshape((ww, ww)))
  #frequencyAnalysis(transform[:,15].reshape((ww, ww)))
  #import sys; sys.exit()

  #plt.imshow(locationHeatmap, cmap="hot", interpolation="nearest")
  #plt.show()
  reconstructions = []
  for i in xrange(numColumns):
    permanence = np.zeros((ws,))
    sp.getPermanence(i, permanence)
    reconstructions.append(permanence.reshape((ww, ww)))

  maxVal = max([m.max() for m in reconstructions])
  reconstructions = [m / maxVal for m in reconstructions]

  zoomFactor = float(ww) / float(autoCorrelation.shape[0])
  autoCorrelation = scipy.ndimage.zoom(autoCorrelation, zoomFactor)
  autoCorrelation = autoCorrelation / autoCorrelation.max()
  locationHeatmap = locationHeatmap / locationHeatmap.max()
  padding = np.zeros((ww, ww))
  header = np.concatenate([autoCorrelation, locationHeatmap] +
                          ([padding] * 3))
  rows = [np.concatenate(reconstructions[i*5:(i+1)*5]) for i in xrange(numColumns / 5)]
  allPermanences = np.concatenate([header] + principleComponents + rows, axis=1)
  #allPermanences = np.concatenate(rows, axis=1)
  plt.imshow(allPermanences, cmap="hot", interpolation="nearest")
  name = "cols_{}_ww_{}_pot_{}_inc_{}_dec_{}_mpo_{}_dc_{}_bs_{}_rr_{}_spW_{}_steps_{}.png".format(numColumns, ww, potentialPct, inc, dec, mpo, dutyCycle, boost, rr, spW, steps)
  plt.savefig(os.path.join(CWD, "output", name))


def main():
  start = time.time()
  stimulusThreshold = 0
  connected = 0.5
  totalTrials = 0
  jumpProb = 0.0
  directionStability = 0.9
  stepSize = 1
  for ww in (30,):
    for rr in (1,):
      for steps in (5000000,):
        for numColumns in (25,):
          for spW in (1,):
            for potentialPct in (1.0,):
              for inc in (0.03,):
                for dec in (0.009,):
                  for mpo in (0.001,):
                    for dutyCycle in (3,):
                      for boost in (15.0,):
                        runTrial(ww=ww, numColumns=numColumns, potentialPct=potentialPct, inc=inc, dec=dec, mpo=mpo, dutyCycle=dutyCycle, boost=boost, steps=steps, rr=rr, spW=spW, stimulusThreshold=stimulusThreshold, connected=connected, stepSize=stepSize, jumpProb=jumpProb, directionStability=directionStability)
                        totalTrials += 1
  print "Total trials: {}".format(totalTrials)
  print "Time: {}".format(time.time() - start)


if __name__ == "__main__":
  main()
