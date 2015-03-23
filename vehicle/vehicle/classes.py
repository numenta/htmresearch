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

  def __init__(self, frictionCoefficient=0.1):
    self.frictionCoefficient = frictionCoefficient


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
    self.move()
    self.distance += 1



class NoOpVehicle(Vehicle):

  def move(self):
    return 0



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
    print self.vehicle.position



class Game(object):

  def __init__(self, field, vehicle, log=True, plot=True):
    self.field = field
    self.vehicle = vehicle

    self.plots = None
    if plot:
      self.plots = Plots(field, vehicle)

    self.logs = None
    if log:
      self.logs = Logs(field, vehicle)


  def run(self):
    while True:
      if self.plots is not None:
        self.plots.update()
        self.plots.render()

      if self.logs is not None:
        self.logs.log()

      self.vehicle.tick()
