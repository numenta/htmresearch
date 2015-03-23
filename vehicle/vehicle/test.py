#!/usr/bin/env python

from vehicle.classes import (
 Field,
 HumanVehicle,
 NoOpSensor,
 AccelerationMotor,
 Game)



if __name__ == "__main__":
  field = Field()
  sensor = NoOpSensor()
  motor = AccelerationMotor()
  vehicle = HumanVehicle(field, sensor, motor)

  game = Game(field, vehicle)
  vehicle.setGraphics(game.graphics)
  game.run()
