import random



class Sensor(object):

  def __init__(self, noise=(0.0, 0.0)):
    self.noise = noise
    self.noiseAmount = 0.0


  def sense(self, field, vehicle):
    raise NotImplementedError



class NoOpSensor(Sensor):

  def sense(self, field, vehicle):
    return None



class PositionSensor(Sensor):

  def sense(self, field, vehicle):
    self.noiseAmount = random.gauss(*self.noise)
    position = vehicle.position + self.noiseAmount
    position = position % field.width
    return position
