#!/usr/bin/env python

import argparse

from vehicle.classes import (
 StraightRoad, ZigZagRoad,
 Field,
 HumanVehicle,
 PositionSensor,
 AccelerationMotor,
 PositionMotor,
 StayOnRoadScorer,
 HTMPositionModel,
 Logs,
 Graphics,
 Plots,
 HTMPlots,
 Game)



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--plots', choices=[None, "default", "htm"],
                      default=None,
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
  sensor = PositionSensor(noise=sensorNoise)

  motorNoise = (0, args.motorNoise)
  if args.motor == "acceleration":
    motor = AccelerationMotor(noise=motorNoise)
  elif args.motor == "position":
    motor = PositionMotor(noise=motorNoise)

  vehicle = HumanVehicle(field, sensor, motor)
  scorer = StayOnRoadScorer(field, vehicle)
  model = HTMPositionModel()

  plots = None
  if args.plots == "htm":
    plots = HTMPlots(field, vehicle, scorer, model)
  if args.plots == "default":
    plots = Plots(field, vehicle, scorer, model)

  logs = Logs(field, vehicle, scorer, model)
  graphics = Graphics(field, vehicle, scorer, model)
  game = Game(field, vehicle, scorer, model,
              logs=logs, plots=plots, graphics=graphics)

  vehicle.setGraphics(game.graphics)
  game.run()
