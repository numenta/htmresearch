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
from nupic.research.monitor_mixin.trace import CountsTrace
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
  plotter = Plotter(tm)

  lastState = None
  lastAction = None

  while True:
    outputData = fetcher.sync()

    if outputData is None:
      continue

    if fetcher.skippedTimesteps > 0:
      print ("Warning: skipped {0} timesteps, "
             "now at {1}").format(fetcher.skippedTimesteps, fetcher.timestep)

    if not ("location" in outputData and
            "steer" in outputData):
      print ("Warning: Missing data on timestep {0}: {1}".format(
             fetcher.timestep, outputData))
      plotter.render()
      continue

    if outputData.get("reset"):
      print "Reset."
      tm.reset()

    location = outputData["location"]
    steer = outputData["steer"]

    x = int(location["x"] * SCALE)
    z = int(location["z"] * SCALE)
    coordinate = numpy.array([x, z])
    encoding = encoder.encode((coordinate, RADIUS))

    motorEncoding = motorEncoder.encode(steer)

    sensorPattern = set(encoding.nonzero()[0])
    motorPattern = set(motorEncoding.nonzero()[0])

    tm.compute(sensorPattern,
               activeExternalCells=motorPattern,
               formInternalConnections=True)

    print tm.mmPrettyPrintMetrics(tm.mmGetDefaultMetrics())

    overlap = 0
    if lastState is not None:
      overlap = (lastState & encoding).sum()

    plotter.update(overlap)

    # if fetcher.timestep % plotEvery == 0 and outputData["reset"]:
    #   plotter.render()

    lastState = encoding
    lastAction = steer



class Plotter(object):

  def __init__(self, tm):
    self.tm = tm
    self.overlaps = []

    import matplotlib.pyplot as plt
    self.plt = plt
    import matplotlib.cm as cm
    self.cm = cm

    from pylab import rcParams
    rcParams.update({'figure.figsize': (6, 9)})
    rcParams.update({'figure.autolayout': True})
    rcParams.update({'figure.facecolor': 'white'})

    self.plt.ion()
    self.plt.show()


  def update(self, overlap):
    self.overlaps.append(overlap)


  def render(self):
    self.plt.figure(1)

    self.plt.clf()

    traces = self.tm.mmGetDefaultTraces()
    traces = [trace for trace in traces if type(trace) is CountsTrace]

    n = len(traces) + 1

    for i in xrange(len(traces)):
      trace = traces[i]
      self.plt.subplot(n, 1, i+1)
      self._plot(trace.data, trace.title)

    self.plt.subplot(n, 1, n)
    self._plot(self.overlaps, "Overlap between encoding at t and t-1")

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
