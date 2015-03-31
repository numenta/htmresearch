import random



class Motor(object):

  def __init__(self, noise=(0.0, 0.0)):
    self.noise = noise
    self.noiseAmount = 0.0


  def move(self, motorValue, vehicle):
    raise NotImplementedError



class AccelerationMotor(Motor):

  def __init__(self, frictionCoefficient=0.1, noise=(0.0, 0.0)):
    super(AccelerationMotor, self).__init__(noise=noise)
    self.frictionCoefficient = frictionCoefficient


  def move(self, motorValue, vehicle):
    self.noiseAmount = random.gauss(*self.noise)
    acceleration = motorValue + self.noiseAmount
    friction = vehicle.velocity * self.frictionCoefficient
    vehicle.velocity += acceleration - friction
    vehicle.position += vehicle.velocity
    vehicle.position = vehicle.position % vehicle.field.width



class PositionMotor(Motor):

  def move(self, motorValue, vehicle):
    self.noiseAmount = random.gauss(*self.noise)
    vehicle.position += motorValue + self.noiseAmount
    vehicle.position = vehicle.position % vehicle.field.width
