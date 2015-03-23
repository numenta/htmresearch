import random

import numpy



class Road(object):

  def get(self, distance):
    raise NotImplementedError



class StraightRoad(Road):

  def get(self, distance):
    return (0.0, 10.0)



class Field(object):

  def __init__(self, width=100):
    self.width = width



class Sensor(object):

  def __init__(self, noise=(0.0, 0.0)):
    self.noise = noise


  def sense(self, field, vehicle):
    raise NotImplementedError



class NoOpSensor(object):

  def __init__(self, n=1024):
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



class Vehicle(object):

  def __init__(self, field, sensor, motor):
    self.field = field
    self.sensor = sensor
    self.motor = motor

    self.position = 0
    self.distance = 0
    self.velocity = 0


  def sense(self):
    return self.sensor.sense()


  def move(self):
    raise NotImplementedError()


  def tick(self):
    command = self.move()
    self.motor.move(command, self)
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
      if key >= 0 and key <= 9:
        return key - 5

    return 0



class Graphics(object):

  def __init__(self, field, vehicle):
    import pygame

    self.field = field
    self.vehicle = vehicle
    self.currentKey = None

    self.pygame = pygame
    self.screen = None

    self.setup()


  def setup(self, size=(320, 240)):
    self.pygame.init()
    self.screen = self.pygame.display.set_mode(size)


  def update(self):
    self.currentKey = None

    for event in self.pygame.event.get():
      if event.type == self.pygame.KEYDOWN:
        self.currentKey = event.key


  def render(self):
    pass



class Plots(object):

  def __init__(self, field, vehicle):
    import matplotlib.pyplot as plt
    # import matplotlib.cm as cm

    self.field = field
    self.vehicle = vehicle
    self.plt = plt
    # self.cm = cm

    self.plt.ion()
    self.plt.show()

    self.positions = []


  def update(self):
    self.positions.append(self.vehicle.position)


  def render(self):
    rows = 1
    cols = 1
    self.plt.clf()

    self.plt.subplot(rows, cols, 1)
    self.plt.ylim([0, self.field.width])
    self.plt.plot(range(len(self.positions)), self.positions)

    self.plt.draw()



class Logs(object):

  def __init__(self, field, vehicle):
    self.field = field
    self.vehicle = vehicle


  def log(self):
    pass



class Game(object):

  def __init__(self, field, vehicle, logs=True, plots=True, graphics=True):
    self.field = field
    self.vehicle = vehicle

    self.plots = None
    if plots:
      self.plots = Plots(field, vehicle)

    self.logs = None
    if logs:
      self.logs = Logs(field, vehicle)

    self.graphics = None
    if graphics:
      self.graphics = Graphics(field, vehicle)


  def run(self):
    while True:
      if self.logs is not None:
        self.logs.log()

      if self.plots is not None:
        self.plots.update()
        self.plots.render()

      if self.graphics is not None:
        self.graphics.update()
        self.graphics.render()

      self.vehicle.tick()
