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

  args = parser.parse_args()

  if args.road == "straight":
    road = StraightRoad()
  elif args.road == "zigzag":
    road = ZigZagRoad()

  field = Field(road)
  sensor = NoOpSensor()

  if args.motor == "acceleration":
    motor = AccelerationMotor()
  elif args.motor == "position":
    motor = PositionMotor()

  vehicle = HumanVehicle(field, sensor, motor)

  scorer = StayOnRoadScorer(field, vehicle)
  game = Game(field, vehicle, scorer, plots=args.plots)
  vehicle.setGraphics(game.graphics)
  game.run()
