import random

import numpy



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

  def __init__(self, noise=(0.0, 0.0), n=1024):
    super(NoOpSensor, self).__init__(noise=noise)
    self.n = n


  def sense(self, field, vehicle):
    return numpy.zeros(self.n)



class Motor(object):

  def __init__(self, noise=(0.0, 0.0)):
    self.noise = noise


  def move(self, command, vehicle):
    raise NotImplementedError



class AccelerationMotor(Motor):

  def __init__(self, frictionCoefficient=0.1, noise=(0.0, 0.0)):
    self.frictionCoefficient = frictionCoefficient
    self.noise = noise


  def move(self, command, vehicle):
    acceleration = command + random.gauss(*self.noise)
    friction = vehicle.velocity * self.frictionCoefficient
    vehicle.velocity += acceleration - friction
    vehicle.position += vehicle.velocity
    vehicle.position = vehicle.position % vehicle.field.width



class PositionMotor(Motor):

  def __init__(self, noise=(0.0, 0.0)):
    self.noise = noise


  def move(self, command, vehicle):
    vehicle.position += command + random.gauss(*self.noise)



class Vehicle(object):

  def __init__(self, field, sensor, motor):
    self.field = field
    self.sensor = sensor
    self.motor = motor

    self.position = 0
    self.distance = 0
    self.velocity = 0

    self.command = None


  def sense(self):
    return self.sensor.sense()


  def move(self):
    raise NotImplementedError()


  def tick(self):
    self.command = self.move()
    self.motor.move(self.command, self)
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



class Graphics(object):

  def __init__(self, field, vehicle, scorer, size=(400, 600)):
    import pygame

    self.field = field
    self.vehicle = vehicle
    self.scorer = scorer
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

  def __init__(self, field, vehicle, scorer):
    import matplotlib.pyplot as plt
    # import matplotlib.cm as cm

    self.field = field
    self.vehicle = vehicle
    self.scorer = scorer
    self.plt = plt
    # self.cm = cm

    self.plt.ion()
    self.plt.show()

    self.positions = []
    self.commands = []
    self.scores = []


  def update(self):
    self.positions.append(self.vehicle.position)
    self.commands.append(self.vehicle.command)
    self.scores.append(self.scorer.score)


  def render(self):
    rows = 2
    cols = 2
    self.plt.clf()

    self.plt.subplot(rows, cols, 1)
    self.plt.ylim([0, self.field.width])
    self.plt.plot(range(len(self.positions)), self.positions)

    self.plt.subplot(rows, cols, 2)
    self.plt.plot(range(len(self.commands)), self.commands)

    self.plt.subplot(rows, cols, 3)
    self.plt.plot(range(len(self.scores)), self.scores)

    self.plt.draw()



class Logs(object):

  def __init__(self, field, vehicle, scorer):
    self.field = field
    self.vehicle = vehicle
    self.scorer = scorer


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

  def __init__(self, field, vehicle, scorer,
               logs=True, plots=True, graphics=True):
    self.field = field
    self.vehicle = vehicle
    self.scorer = scorer

    self.plots = None
    if plots:
      self.plots = Plots(field, vehicle, scorer)

    self.logs = None
    if logs:
      self.logs = Logs(field, vehicle, scorer)

    self.graphics = None
    if graphics:
      self.graphics = Graphics(field, vehicle, scorer)


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

      i += 1
