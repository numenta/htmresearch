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

from collections import defaultdict
import operator

import numpy

from unity_client.fetcher import Fetcher
from nupic.encoders.coordinate import CoordinateEncoder
from nupic.encoders.scalar import ScalarEncoder
from sensorimotor.general_temporal_memory import GeneralTemporalMemory
from nupic.research.monitor_mixin.temporal_memory_monitor_mixin import (
  TemporalMemoryMonitorMixin)
class MonitoredGeneralTemporalMemory(TemporalMemoryMonitorMixin,
                                     GeneralTemporalMemory): pass



SCALE = 5
RADIUS = 7



def run(plotEvery=1):
  encoder = CoordinateEncoder(n=1024,
                              w=21)
  motorEncoder = ScalarEncoder(21, -1, 1,
                               n=1024)
  fetcher = Fetcher()
  plotter = Plotter(encoder)
  tm = MonitoredGeneralTemporalMemory(
    columnDimensions=[2048],
    cellsPerColumn=1,
    initialPermanence=0.5,
    connectedPermanence=0.6,
    permanenceIncrement=0.1,
    permanenceDecrement=0.02,
    minThreshold=35,
    activationThreshold=35,
    maxNewSynapseCount=40)

  lastState = None
  lastAction = None

  while True:
    outputData = fetcher.sync()

    if outputData is None:
      continue

    if fetcher.skippedTimesteps > 0:
      print ("Warning: skipped {0} timesteps, "
             "now at {1}").format(fetcher.skippedTimesteps, fetcher.timestep)

    if not ("reset" in outputData and
            "location" in outputData and
            "steer" in outputData):
      print ("Warning: Missing data on timestep {0}: {1}".format(
             fetcher.timestep, outputData))
      continue

    if outputData["reset"]:
      print "Reset."
      tm.reset()
      tm.mmClearHistory()

    location = outputData["location"]
    steer = outputData["steer"]

    x = int(location["x"] * SCALE)
    z = int(location["z"] * SCALE)
    coordinate = numpy.array([x, z])
    encoding = encoder.encode((coordinate, RADIUS))

    motorEncoding = motorEncoder.encode(steer)

    # if lastState is not None:
    #   print (lastState & encoding).sum()

    sensorPattern = set(encoding.nonzero()[0])
    motorPattern = set(motorEncoding.nonzero()[0])

    tm.compute(sensorPattern,
               activeExternalCells=motorPattern,
               formInternalConnections=True)

    print tm.mmPrettyPrintMetrics(tm.mmGetDefaultMetrics())

    lastState = encoding
    lastAction = steer



class Plotter(object):

  def __init__(self, encoder):
    self.encoder = encoder

    self.sensor = []
    self.encoding = []
    self.steer = []
    self.reward = []
    self.value = []
    self.qValues = defaultdict(lambda: [])
    self.bestAction = []

    import matplotlib.pyplot as plt
    self.plt = plt
    import matplotlib.cm as cm
    self.cm = cm

    from pylab import rcParams
    rcParams.update({'figure.figsize': (6, 9)})
    # rcParams.update({'figure.autolayout': True})
    rcParams.update({'figure.facecolor': 'white'})

    self.plt.ion()
    self.plt.show()


  def update(self, sensor, encoding, steer, reward, value, qValues):
    self.sensor.append(sensor)
    self.encoding.append(encoding)
    self.steer.append(steer)
    self.reward.append(reward)
    self.value.append(value)

    for key, value in qValues.iteritems():
      self.qValues[key].append(value)

    bestAction = int(max(qValues.iteritems(), key=operator.itemgetter(1))[0])
    self.bestAction.append(bestAction)


  def render(self):
    self.plt.figure(1)

    self.plt.clf()

    n = 7

    self.plt.subplot(n,1,1)
    self._plot(self.steer, "Steer over time")

    self.plt.subplot(n,1,2)
    self._plot(self.reward, "Reward over time")

    self.plt.subplot(n,1,3)
    self._plot(self.value, "Value over time")

    self.plt.subplot(n,1,4)
    shape = len(self.encoder.positions), self.encoder.scalarEncoder.getWidth()
    encoding = numpy.array(self.encoding[-1]).reshape(shape).transpose()
    self._imshow(encoding, "Encoding at time t")

    self.plt.subplot(n,1,5)
    data = self.encoding
    w = self.encoder.w
    overlaps = [sum(a & b) / float(w) for a, b in zip(data[:-1], data[1:])]
    self._plot(overlaps, "Encoding overlaps between consecutive times")

    # for i, action in enumerate(ACTIIONS):
    #   self.plt.subplot(n,1,4+i)
    #   self._plot(self.qValues[action], "Q value: {0}".format(action))

    # self.plt.subplot(n,1,7)
    # self._plot(self.bestAction, "Best action")

    self.plt.draw()


  def _plot(self, data, title):
    self.plt.title(title)
    self.plt.xlim(0, len(data))
    self.plt.plot(range(len(data)), data)


  def _imshow(self, data, title):
    self.plt.title(title)
    self.plt.imshow(data,
                    cmap=self.cm.Greys,
                    interpolation="nearest",
                    aspect='auto',
                    vmin=0,
                    vmax=1)



if __name__ == "__main__":
  run()
