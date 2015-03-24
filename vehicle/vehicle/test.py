#!/usr/bin/env python

import argparse

from vehicle.classes import (
 ZigZagRoad,
 Field,
 HumanVehicle,
 NoOpSensor,
 AccelerationMotor,
 Game)



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--plots', action='store_true',
                      help="Enable plots")

  args = parser.parse_args()

  road = ZigZagRoad()
  field = Field(road)
  sensor = NoOpSensor()
  motor = AccelerationMotor()
  vehicle = HumanVehicle(field, sensor, motor)

  game = Game(field, vehicle, plots=args.plots)
  vehicle.setGraphics(game.graphics)
  game.run()
