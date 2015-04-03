import random



class Vehicle(object):

  def __init__(self, field, sensor, motor,
               motorValues=range(-4, 4+1),
               startPosition=0):
    self.field = field
    self.sensor = sensor
    self.motor = motor
    self.motorValues = motorValues

    self.position = startPosition
    self.distance = 0
    self.velocity = 0
    self.acceleration = 0
    self.jerk = 0

    self.sensorValue = None
    self.motorValue = None


  def move(self):
    raise NotImplementedError()


  def tick(self):
    if self.motorValue is not None:
      self.motor.move(self.motorValue, self)
    self.sensorValue = self.sensor.sense(self.field, self)
    self.setMotorValue(self.move())
    self.distance += 1


  def setMotorValue(self, motorValue):
    self.motorValue = motorValue



class NoOpVehicle(Vehicle):

  def move(self):
    return 0



class HumanVehicle(Vehicle):

  def setGraphics(self, graphics):
    self.graphics = graphics


  def move(self):
    if self.graphics.currentKey is not None:
      key = self.graphics.currentKey - 48
      motorValue = key - 5
      if motorValue in self.motorValues:
        return motorValue

    return 0



class RandomVehicle(Vehicle):

  def move(self):
    return random.choice(self.motorValues)



class LoopVehicle(Vehicle):

  def __init__(self, field, sensor, motor, startPosition=0,
               motorValues=range(-4, 4+1)):
    super(LoopVehicle, self).__init__(field, sensor, motor,
                                      startPosition=startPosition)
    self.motorValues = motorValues
    self.idx = 0


  def move(self):
    value = self.motorValues[self.idx]
    self.idx += 1
    self.idx %= len(self.motorValues)
    return value
