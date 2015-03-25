import random

import numpy

from sensorimotor.general_temporal_memory import GeneralTemporalMemory
from nupic.research.monitor_mixin.temporal_memory_monitor_mixin import (
  TemporalMemoryMonitorMixin)
class MonitoredGeneralTemporalMemory(TemporalMemoryMonitorMixin,
                                     GeneralTemporalMemory): pass

from nupic.encoders.random_distributed_scalar import (
  RandomDistributedScalarEncoder)

PLOT_EVERY = 25



class Road(object):

  def get(self, distance, field):
    raise NotImplementedError



class StraightRoad(Road):

  def get(self, distance, field):
    return (float(field.width) / 2, 25)



class ZigZagRoad(Road):

  def __init__(self, width=25, zigZagEvery=60):
    super(ZigZagRoad, self).__init__()

    self.width = width
    self.zigZagEvery = zigZagEvery


  def get(self, distance, field):
    zig = int(distance / self.zigZagEvery)
    zag = int((distance + self.zigZagEvery) / self.zigZagEvery)

    zigDistance = distance % self.zigZagEvery

    numpy.random.seed(zig)
    zigPosition = numpy.random.randint(0, field.width)
    numpy.random.seed(zag)
    zagPosition = numpy.random.randint(0, field.width)
    separation = zagPosition - zigPosition

    zigPercent = float(zigDistance) / self.zigZagEvery
    position = zigPosition + (separation * zigPercent)

    return (position, self.width)


  def containsVehicle(self, vehicle):
    if vehicle.position > 50:
      return True

    return False



class Field(object):

  def __init__(self, road, width=100):
    self.width = width
    self.road = road



class Sensor(object):

  def __init__(self, noise=(0.0, 0.0)):
    self.noise = noise


  def sense(self, field, vehicle):
    raise NotImplementedError



class NoOpSensor(Sensor):

  def sense(self, field, vehicle):
    return None



class PositionSensor(Sensor):

  def sense(self, field, vehicle):
    return vehicle.position



class Motor(object):

  def __init__(self, noise=(0.0, 0.0)):
    self.noise = noise


  def move(self, motorValue, vehicle):
    raise NotImplementedError



class AccelerationMotor(Motor):

  def __init__(self, frictionCoefficient=0.1, noise=(0.0, 0.0)):
    self.frictionCoefficient = frictionCoefficient
    self.noise = noise


  def move(self, motorValue, vehicle):
    acceleration = motorValue + random.gauss(*self.noise)
    friction = vehicle.velocity * self.frictionCoefficient
    vehicle.velocity += acceleration - friction
    vehicle.position += vehicle.velocity
    vehicle.position = vehicle.position % vehicle.field.width



class PositionMotor(Motor):

  def __init__(self, noise=(0.0, 0.0)):
    self.noise = noise


  def move(self, motorValue, vehicle):
    vehicle.position += motorValue + random.gauss(*self.noise)



class Vehicle(object):

  def __init__(self, field, sensor, motor):
    self.field = field
    self.sensor = sensor
    self.motor = motor

    self.position = 0
    self.distance = 0
    self.velocity = 0

    self.sensorValue = None
    self.motorValue = None


  def move(self):
    raise NotImplementedError()


  def tick(self):
    self.sensorValue = self.sensor.sense(self.field, self)
    self.motorValue = self.move()
    self.motor.move(self.motorValue, self)
    self.distance += 1



class NoOpVehicle(Vehicle):

  def move(self):
    return 0



class HumanVehicle(Vehicle):

  def setGraphics(self, graphics):
    self.graphics = graphics


  def move(self):
    if self.graphics.currentKey is not None:
      key = self.graphics.currentKey - 48
      if key >= 1 and key <= 9:
        return key - 5

    return 0



class RandomVehicle(Vehicle):

  def __init__(self, field, sensor, motor, motorValues=range(-4, 4+1)):
    super(RandomVehicle, self).__init__(field, sensor, motor)
    self.motorValues = motorValues


  def move(self):
    return random.choice(self.motorValues)



class Model(object):

  def update(self, sensorValue, motorValue):
    pass


  def predict(self):
    raise NotImplementedError



class HTMPositionModel(Model):

  def __init__(self, sparsity=0.02, encoderResolution=0.05, tmParams=None):
    tmParams = tmParams or {}
    self.tm = MonitoredGeneralTemporalMemory(mmName="TM", **tmParams)
    self.n = self.tm.numberOfColumns()
    self.w = int(self.n * sparsity) + 1
    self.sensorEncoder = RandomDistributedScalarEncoder(encoderResolution,
                                                        w=self.w,
                                                        n=self.n)
    self.motorEncoder = RandomDistributedScalarEncoder(encoderResolution,
                                                       w=self.w,
                                                       n=self.n)


  def update(self, sensorValue, motorValue):
    sensorPattern = set(self.sensorEncoder.encode(sensorValue).nonzero()[0])
    motorPattern = set(self.motorEncoder.encode(motorValue).nonzero()[0])

    self.tm.compute(sensorPattern,
                    activeExternalCells=motorPattern,
                    formInternalConnections=True,
                    learn=True)


  def predict(self):
    print self.tm.predictedCells
    return set()  # TODO



class Graphics(object):

  def __init__(self, field, vehicle, scorer, model, size=(400, 600)):
    import pygame

    self.field = field
    self.vehicle = vehicle
    self.scorer = scorer
    self.model = model
    self.size = size
    self.pygame = pygame

    self.currentKey = None
    self.screen = None

    self.setup()


  def setup(self):
    self.pygame.init()
    self.screen = self.pygame.display.set_mode(self.size)


  def update(self):
    self.currentKey = None

    for event in self.pygame.event.get():
      if event.type == self.pygame.KEYDOWN:
        self.currentKey = event.key


  def render(self):
    self.renderBackground()
    self.renderRoad()
    self.renderVehicle()

    self.pygame.display.flip()


  def renderBackground(self):
    black = (0, 0, 0)
    self.screen.fill(black)


  def renderVehicle(self):
    color = (0, 255, 0)
    x = self._scale(self.vehicle.position)
    y = float(self.size[1]) / 2
    self.pygame.draw.rect(self.screen,
                          color,
                          self.pygame.Rect(x, y, 10, 10))


  def renderRoad(self):
    color = (255, 0, 0)

    for y in range(self.size[1]):
      distance = self.vehicle.distance + y - float(self.size[1]) / 2

      if distance < 0:
        continue

      position, width = self.field.road.get(distance, self.field)
      x = self._scale(float(position - width) / 2)
      w = self._scale(width)

      self.pygame.draw.rect(self.screen,
                            color,
                            self.pygame.Rect(x, self.size[1] - y, w, 2))


  def _scale(self, x):
    return float(x * self.size[0]) / self.field.width



class Plots(object):

  def __init__(self, field, vehicle, scorer, model):
    self.field = field
    self.vehicle = vehicle
    self.scorer = scorer
    self.model = model

    import matplotlib.pyplot as plt
    self.plt = plt
    # import matplotlib.cm as cm
    # self.cm = cm

    self.plt.ion()
    self.plt.show()

    self.positions = []
    self.sensorValues = []
    self.motorValues = []
    self.scores = []


  def update(self):
    self.positions.append(self.vehicle.position)
    self.sensorValues.append(self.vehicle.sensorValue)
    self.motorValues.append(self.vehicle.motorValue)
    self.scores.append(self.scorer.score)


  def render(self):
    rows = 4
    cols = 1
    self.plt.clf()

    self.plt.subplot(rows, cols, 1)
    self.plt.ylabel("Position")
    self.plt.ylim([0, self.field.width])
    self.plt.plot(range(len(self.positions)), self.positions)

    self.plt.subplot(rows, cols, 2)
    self.plt.ylabel("Sensor value")
    self.plt.plot(range(len(self.sensorValues)), self.sensorValues)

    self.plt.subplot(rows, cols, 3)
    self.plt.ylabel("Motor value")
    self.plt.plot(range(len(self.motorValues)), self.motorValues)

    self.plt.subplot(rows, cols, 4)
    self.plt.ylabel("Score")
    self.plt.plot(range(len(self.scores)), self.scores)

    self.plt.draw()



class HTMPlots(Plots):

  def render(self):
    self.plt.figure(1)
    super(HTMPlots, self).render()

    # self.plt.figure(2)
    # self.model.tm.mmGetCellActivityPlot()

    self.plt.figure(2)
    self.plt.clf()
    rows = 4
    cols = 1

    data = self.model.tm.mmGetTraceActiveColumns().data
    overlaps = [len(a & b) for a, b in zip(data[:-1], data[1:])]
    self.plt.subplot(rows, cols, 1)
    self.plt.ylabel("Active columns overlap with t-1")
    self.plt.plot(range(len(overlaps)), overlaps)

    data = self.model.tm.mmGetTraceUnpredictedActiveColumns().makeCountsTrace().data
    self.plt.subplot(rows, cols, 2)
    self.plt.ylabel("Unpredicted active columns")
    self.plt.plot(range(len(data)), data)

    data = self.model.tm.mmGetTracePredictedActiveColumns().makeCountsTrace().data
    self.plt.subplot(rows, cols, 3)
    self.plt.ylabel("Predicted active columns")
    self.plt.plot(range(len(data)), data)

    data = self.model.tm.mmGetTracePredictedInactiveColumns().makeCountsTrace().data
    self.plt.subplot(rows, cols, 4)
    self.plt.ylabel("Predicted inactive columns")
    self.plt.plot(range(len(data)), data)



class Logs(object):

  def __init__(self, field, vehicle, scorer, model):
    self.field = field
    self.vehicle = vehicle
    self.scorer = scorer
    self.model = model


  def log(self):
    print self.scorer.score



class Scorer(object):

  def __init__(self, field, vehicle):
    self.field = field
    self.vehicle = vehicle

    self.score = 0
    self.scoreDelta = 0


  def getChange(self):
    raise NotImplementedError


  def update(self):
    self.scoreDelta = self.getChange()
    self.score += self.scoreDelta



class StayOnRoadScorer(Scorer):

  def getChange(self):
    position, width = self.field.road.get(self.vehicle.distance, self.field)
    left = float(position - width) / 2
    right = left + width

    if self.vehicle.position >= left and self.vehicle.position <= right:
      return 1
    else:
      return -1



class Game(object):

  def __init__(self, field, vehicle, scorer, model,
               logs=None, plots=None, graphics=None):
    self.field = field
    self.vehicle = vehicle
    self.scorer = scorer
    self.model = model
    self.logs = logs
    self.plots = plots
    self.graphics = graphics
    self.logs = logs
    self.graphics = graphics


  def run(self):
    i = 0
    while True:
      if self.logs is not None:
        self.logs.log()

      if self.plots is not None:
        self.plots.update()

        if i % PLOT_EVERY == 0:
          self.plots.render()

      if self.graphics is not None:
        self.graphics.update()
        self.graphics.render()

      self.vehicle.tick()
      self.scorer.update()
      self.model.update(self.vehicle.sensorValue, self.vehicle.motorValue)

      i += 1
