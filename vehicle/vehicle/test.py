#!/usr/bin/env python

from vehicle.classes import (
 Field,
 NoOpVehicle,
 NoOpSensor,
 AccelerationMotor,
 Game)



if __name__ == "__main__":
  field = Field()
  sensor = NoOpSensor()
  motor = AccelerationMotor()
  vehicle = NoOpVehicle(field, sensor, motor)

  game = Game(field, vehicle)
  game.run()
