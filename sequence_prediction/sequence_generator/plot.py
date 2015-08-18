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

import pickle
import sys
import time

from matplotlib import pyplot
import numpy



def movingAverage(a, n):
  weights = numpy.repeat(1.0, n)/n
  return numpy.convolve(a, weights, 'valid')



def plotMovingAverage(data, window, label=None):
  movingData = movingAverage(data, min(len(data), window))
  style = 'ro' if len(data) < window else ''
  pyplot.plot(range(len(movingData)), movingData, style, label=label)



def plotAccuracy(correct, window=100, label=None):
  pyplot.title("High-order prediction")
  pyplot.xlabel("# of sequences seen")
  pyplot.ylabel("High-order prediction accuracy over last {0} sequences".format(window))
  plotMovingAverage(correct, window, label=label)



def plotTMStats(numPredictedActiveCells, numPredictedInactiveCells, numUnpredictedActiveColumns,
                window=100):
  pyplot.subplot(3, 1, 1)
  pyplot.title("# predicted => active cells over window={0}".format(window))
  plotMovingAverage(numPredictedActiveCells, window)

  pyplot.subplot(3, 1, 2)
  pyplot.title("# predicted => inactive cells over window={0}".format(window))
  plotMovingAverage(numPredictedInactiveCells, window)

  pyplot.subplot(3, 1, 3)
  pyplot.title("# unpredicted => active cells over window={0}".format(window))
  plotMovingAverage(numUnpredictedActiveColumns, window)



def plotTraces(tm, timestamp=int(time.time()), window=500):
  """
  Have to make the following change in NuPIC for this to work:

  --- a/nupic/research/TP_shim.py
  +++ b/nupic/research/TP_shim.py
  @@ -27,10 +27,13 @@ for use with OPF.
   import numpy

   from nupic.research.temporal_memory import TemporalMemory
  +from nupic.research.monitor_mixin.temporal_memory_monitor_mixin import (
  +  TemporalMemoryMonitorMixin)
  +class MonitoredTemporalMemory(TemporalMemoryMonitorMixin, TemporalMemory): pass



  -class TPShim(TemporalMemory):
  +class TPShim(MonitoredTemporalMemory):
  """
  traces = tm.mmGetDefaultTraces()
  traces = [trace for trace in traces if type(trace) is CountsTrace]

  t = len(traces)

  for i in xrange(t):
    trace = traces[i]
    pyplot.subplot(t, 1, i+1)
    pyplot.title(trace.title)
    pyplot.xlim(max(len(trace.data)-window, 0), len(trace.data))
    pyplot.plot(range(len(trace.data)), trace.data)

  pyplot.draw()
  # pyplot.savefig("tm-{0}.png".format(timestamp))



if __name__ == "__main__":
  from pylab import rcParams

  rcParams.update({'figure.autolayout': True})
  rcParams.update({'figure.facecolor': 'white'})
  rcParams.update({'ytick.labelsize': 8})
  rcParams.update({'figure.figsize': (12, 6)})


  with open(sys.argv[1]) as infile:
    results = pickle.load(infile)

    for (numPredictions, accuracy) in results:
      plotAccuracy(accuracy, label="{0} possible predictions per sequence".format(numPredictions))

    # TODO: Fix below
    # pyplot.figure(2)
    # plotTMStats(numPredictedActiveCells, numPredictedInactiveCells, numUnpredictedActiveColumns)

  pyplot.legend()
  pyplot.show()
