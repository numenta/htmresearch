#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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

from collections import defaultdict
import operator
import time

import numpy

from unity_client.server import Server
from nupic.encoders.coordinate import CoordinateEncoder
from nupic.encoders.scalar import ScalarEncoder
from nupic.algorithms.monitor_mixin.trace import CountsTrace
from sensorimotor.extended_temporal_memory import ExtendedTemporalMemory
from htmresearch.support.etm_monitor_mixin import (
  ExtendedTemporalMemoryMonitorMixin)
class MonitoredExtendedTemporalMemory(ExtendedTemporalMemoryMonitorMixin,
                                     ExtendedTemporalMemory): pass



SCALE = 5
RADIUS = 10



class Agent(object):

  def __init__(self):
    self.encoder = CoordinateEncoder(n=1024,
                                w=21)
    self.motorEncoder = ScalarEncoder(21, -1, 1,
                                 n=1024)
    self.tm = MonitoredExtendedTemporalMemory(
      columnDimensions=[2048],
      basalInputDimensions: (999999,) # Dodge input checking.
      cellsPerColumn=1,
      initialPermanence=0.5,
      connectedPermanence=0.6,
      permanenceIncrement=0.1,
      permanenceDecrement=0.02,
      minThreshold=35,
      activationThreshold=35,
      maxNewSynapseCount=40)
    self.plotter = Plotter(self.tm, showOverlaps=False, showOverlapsValues=False)

    self.lastState = None
    self.lastAction = None
    self.prevMotorPattern = ()


  def sync(self, outputData):
    if not ("location" in outputData and
            "steer" in outputData):
      print "Warning: Missing data:", outputData
      return

    reset = outputData.get("reset") or False

    if reset:
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
                    activeCellsExternalBasal=motorPattern,
                    reinforceCandidatesExternalBasal=self.prevMotorPattern,
                    growthCandidatesExternalBasal=self.prevMotorPattern)

    print self.tm.mmPrettyPrintMetrics(self.tm.mmGetDefaultMetrics())

    self.plotter.update(encoding, reset)

    if reset:
      self.plotter.render()

    self.lastState = encoding
    self.lastAction = steer
    self.prevMotorPattern = motorPattern



class Plotter(object):

  def __init__(self, tm, showOverlaps=False, showOverlapsValues=False):
    self.tm = tm

    self.showOverlaps = showOverlaps
    self.showOverlapsValues = showOverlapsValues

    self.encodings = []
    self.resets = []
    self.numSegmentsPerCell = []
    self.numSynapsesPerSegment = []

    import matplotlib.pyplot as plt
    self.plt = plt
    import matplotlib.cm as cm
    self.cm = cm

    from pylab import rcParams

    if self.showOverlaps and self.showOverlapsValues:
      rcParams.update({'figure.figsize': (20, 20)})
    else:
      rcParams.update({'figure.figsize': (6, 12)})

    rcParams.update({'figure.autolayout': True})
    rcParams.update({'figure.facecolor': 'white'})
    rcParams.update({'ytick.labelsize': 8})


  def update(self, encoding, reset):
    self.encodings.append(encoding)
    self.resets.append(reset)

    # TODO: Deal with empty segments / unconnected synapses
    numSegmentsPerCell = [len(segments) for segments in
                          self.tm.connections._segmentsForCell.values()]
    self.numSegmentsPerCell.append(numpy.array(numSegmentsPerCell))

    numSynapsesPerSegment = [len(synapses) for synapses in
                             self.tm.connections._synapsesForSegment.values()]
    self.numSynapsesPerSegment.append(numpy.array(numSynapsesPerSegment))


  def render(self):
    timestamp = int(time.time())

    self.plt.figure(1)
    self.plt.clf()
    self._renderMetrics(timestamp)

    if self.showOverlaps:
      self.plt.figure(2)
      self.plt.clf()
      self._renderOverlaps(timestamp)


  def _renderMetrics(self, timestamp):
    traces = self.tm.mmGetDefaultTraces()
    traces = [trace for trace in traces if type(trace) is CountsTrace]

    t = len(traces)
    n = t + 2

    for i in xrange(t):
      trace = traces[i]
      self.plt.subplot(n, 1, i+1)
      self._plot(trace.data, trace.title)

    self.plt.subplot(n, 1, t+1)
    self._plotDistributions(self.numSegmentsPerCell, "# segments per cell")

    self.plt.subplot(n, 1, t+2)
    self._plotDistributions(self.numSynapsesPerSegment, "# synapses per segment")

    self.plt.draw()
    self.plt.savefig("sm-{0}_A.png".format(timestamp))


  def _renderOverlaps(self, timestamp):
    self.plt.subplot(1, 1, 1)

    overlaps = self._computeOverlaps()
    self._imshow(overlaps, "Overlaps", aspect=None)

    for i in self._computeResetIndices():
      self.plt.axvline(i, color='black', alpha=0.5)
      self.plt.axhline(i, color='black', alpha=0.5)

    if self.showOverlapsValues:
      for i in range(len(overlaps)):
        for j in range(len(overlaps[i])):
          overlap = "%.1f" % overlaps[i][j]
          self.plt.annotate(overlap, xy=(i, j), fontsize=6, color='red', verticalalignment='center', horizontalalignment='center')

    self.plt.draw()
    self.plt.savefig("sm-{0}_B.png".format(timestamp))


  def _computeOverlaps(self):
    overlaps = []
    encodings = self.encodings

    for i in range(len(encodings)):
      row = []

      for j in range(len(encodings)):
        n = max(encodings[i].sum(), encodings[j].sum())
        overlap = (encodings[i] & encodings[j]).sum() / float(n)
        row.append(overlap)

      overlaps.append(row)

    return overlaps


  def _computeResetIndices(self):
    return numpy.array(self.resets).nonzero()[0]


  def _plot(self, data, title):
    self.plt.plot(range(len(data)), data)
    self._finishPlot(data, title)


  def _finishPlot(self, data, title):
    self.plt.title(title)
    self.plt.xlim(0, len(data))

    for i in self._computeResetIndices():
      self.plt.axvline(i, color='black', alpha=0.5)


  def _imshow(self, data, title, aspect='auto'):
    self.plt.title(title)
    self.plt.imshow(data,
                    cmap=self.cm.Greys,
                    interpolation="nearest",
                    aspect=aspect,
                    vmin=0,
                    vmax=1)


  def _plotDistributions(self, data, title):
    means = [numpy.mean(x) if len(x) else 0 for x in data]
    maxs = [numpy.max(x) if len(x) else 0 for x in data]
    self.plt.plot(range(len(data)), means, label='mean')
    self.plt.plot(range(len(data)), maxs, label='max')
    self.plt.legend(loc='lower right')
    self._finishPlot(data, title)



if __name__ == "__main__":
  agent = Agent()
  Server(agent)
