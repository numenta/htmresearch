#!/usr/bin/env python

import argparse

from vehicle.classes import (
 StraightRoad, ZigZagRoad,
 Field,
 HumanVehicle,
 NoOpSensor,
 AccelerationMotor,
 PositionMotor,
 StayOnRoadScorer,
 Game)



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--plots', action='store_true',
                      help="Enable plots")
  parser.add_argument('--motor', choices=["acceleration", "position"],
                      default="acceleration")
  parser.add_argument('--road', choices=["straight", "zigzag"],
                      default="zigzag")
  parser.add_argument('--sensorNoise', type=float,
                      default=0)
  parser.add_argument('--motorNoise', type=float,
                      default=0)

  args = parser.parse_args()

  if args.road == "straight":
    road = StraightRoad()
  elif args.road == "zigzag":
    road = ZigZagRoad()

  field = Field(road)

  sensorNoise = (0, args.sensorNoise)
  sensor = NoOpSensor(noise=sensorNoise)

  motorNoise = (0, args.motorNoise)
  if args.motor == "acceleration":
    motor = AccelerationMotor(noise=motorNoise)
  elif args.motor == "position":
    motor = PositionMotor(noise=motorNoise)

  vehicle = HumanVehicle(field, sensor, motor)

  scorer = StayOnRoadScorer(field, vehicle)
  game = Game(field, vehicle, scorer, plots=args.plots)
  vehicle.setGraphics(game.graphics)
  game.run()
