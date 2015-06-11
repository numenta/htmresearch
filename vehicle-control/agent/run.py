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
from sensorimotor.encoders.one_d_depth import OneDDepthEncoder
from sensorimotor.q_learner import QLearner



ACTIIONS = ["-1", "0", "1"]



def run(positions, plotEvery=1):
  encoder = OneDDepthEncoder(positions=positions,
                             radius=5,
                             wrapAround=True,
                             nPerPosition=28,
                             wPerPosition=3,
                             minVal=0,
                             maxVal=1)
  fetcher = Fetcher()
  plotter = Plotter(encoder)
  learner = QLearner(ACTIIONS, n=1008)

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
            "ForwardsSweepSensor" in outputData and
            "steer" in outputData):
      print ("Warning: Missing data on timestep {0}: {1}".format(
             fetcher.timestep, outputData))
      continue

    if outputData["reset"]:
      print "Reset."

    sensor = outputData["ForwardsSweepSensor"]
    steer = outputData["steer"]
    reward = outputData.get("reward") or 0

    encoding = encoder.encode(numpy.array(sensor))

    if lastState is not None:
      learner.update(lastState, str(lastAction), encoding, str(steer), reward)

    value = learner.value(encoding)

    qValues = {}
    for action in ACTIIONS:
      qValues[action] = learner.qValue(encoding, action)

    plotter.update(sensor, encoding, steer, reward, value, qValues)

    if fetcher.timestep % plotEvery == 0:
      plotter.render()

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

    # self.plt.subplot(n,1,4)
    # shape = len(self.encoder.positions), self.encoder.scalarEncoder.getWidth()
    # encoding = numpy.array(self.encoding[-1]).reshape(shape).transpose()
    # self._imshow(encoding, "Encoding at time t")

    # self.plt.subplot(n,1,5)
    # data = self.encoding
    # w = self.encoder.w
    # overlaps = [sum(a & b) / float(w) for a, b in zip(data[:-1], data[1:])]
    # self._plot(overlaps, "Encoding overlaps between consecutive times")

    for i, action in enumerate(ACTIIONS):
      self.plt.subplot(n,1,4+i)
      self._plot(self.qValues[action], "Q value: {0}".format(action))

    self.plt.subplot(n,1,7)
    self._plot(self.bestAction, "Best action")

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
  # complete uniform
  # positions = [i*20 for i in range(36)]

  # forward uniform
  positions = [i*10 for i in range(18, 54)]

  run(positions)
