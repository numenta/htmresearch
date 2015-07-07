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
import time

import numpy

from unity_client.server import Server
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



class Agent(object):

  def __init__(self):
    self.encoder = CoordinateEncoder(n=1024,
                                w=21)
    self.motorEncoder = ScalarEncoder(21, -1, 1,
                                 n=1024)
    self.tm = MonitoredGeneralTemporalMemory(
      columnDimensions=[2048],
      cellsPerColumn=1,
      initialPermanence=0.5,
      connectedPermanence=0.6,
      permanenceIncrement=0.1,
      permanenceDecrement=0.02,
      minThreshold=35,
      activationThreshold=35,
      maxNewSynapseCount=40)
    self.plotter = Plotter(self.tm)

    self.lastState = None
    self.lastAction = None


  def sync(self, outputData):
    if not ("location" in outputData and
            "steer" in outputData):
      print "Warning: Missing data:", outputData
      return

    if outputData.get("reset"):
      print "Reset."
      self.tm.reset()

    location = outputData["location"]
    steer = outputData["steer"]

    x = int(location["x"] * SCALE)
    z = int(location["z"] * SCALE)
    coordinate = numpy.array([x, z])
    encoding = self.encoder.encode((coordinate, RADIUS))

    motorEncoding = self.motorEncoder.encode(steer)

    sensorPattern = set(encoding.nonzero()[0])
    motorPattern = set(motorEncoding.nonzero()[0])

    self.tm.compute(sensorPattern,
                    activeExternalCells=motorPattern,
                    formInternalConnections=True)

    print self.tm.mmPrettyPrintMetrics(self.tm.mmGetDefaultMetrics())

    overlap = 0
    if self.lastState is not None:
      overlap = (self.lastState & encoding).sum()

    self.plotter.update(overlap)

    if outputData.get("reset"):
      self.plotter.render()

    self.lastState = encoding
    self.lastAction = steer



class Plotter(object):

  def __init__(self, tm):
    self.tm = tm
    self.overlaps = []
    self.numSegmentsPerCell = []
    self.numSynapsesPerSegment = []

    import matplotlib.pyplot as plt
    self.plt = plt
    import matplotlib.cm as cm
    self.cm = cm

    from pylab import rcParams
    rcParams.update({'figure.figsize': (6, 12)})
    rcParams.update({'figure.autolayout': True})
    rcParams.update({'figure.facecolor': 'white'})
    rcParams.update({'ytick.labelsize': 8})

    # self.plt.ion()
    # self.plt.show()


  def update(self, overlap):
    self.overlaps.append(overlap)

    # TODO: Deal with empty segments / unconnected synapses
    numSegmentsPerCell = [len(segments) for segments in
                          self.tm.connections._segmentsForCell.values()]
    self.numSegmentsPerCell.append(numpy.array(numSegmentsPerCell))

    numSynapsesPerSegment = [len(synapses) for synapses in
                             self.tm.connections._synapsesForSegment.values()]
    self.numSynapsesPerSegment.append(numpy.array(numSynapsesPerSegment))


  def render(self):
    self.plt.figure(1)

    self.plt.clf()

    traces = self.tm.mmGetDefaultTraces()
    traces = [trace for trace in traces if type(trace) is CountsTrace]

    n = len(traces) + 3

    for i in xrange(len(traces)):
      trace = traces[i]
      self.plt.subplot(n, 1, i+1)
      self._plot(trace.data, trace.title)

    self.plt.subplot(n, 1, n-2)
    self._plotDistributions(self.numSegmentsPerCell, "# segments per cell")

    self.plt.subplot(n, 1, n-1)
    self._plotDistributions(self.numSynapsesPerSegment, "# synapses per segment")

    self.plt.subplot(n, 1, n)
    self._plot(self.overlaps, "Overlap between encoding at t and t-1")

    self.plt.draw()
    self.plt.savefig("sm-{0}.png".format(time.time()))


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


  def _plotDistributions(self, data, title):
    self.plt.title(title)
    self.plt.xlim(0, len(data))

    means = [numpy.mean(x) if len(x) else 0 for x in data]
    maxs = [numpy.max(x) if len(x) else 0 for x in data]
    self.plt.plot(range(len(data)), means, label='mean')
    self.plt.plot(range(len(data)), maxs, label='max')
    self.plt.legend(loc='lower right')



if __name__ == "__main__":
  agent = Agent()
  Server(agent)
