#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
import csv
from math import sqrt
from matplotlib import pyplot



FILE_PREFIX = "data"



# From http://rosettacode.org/wiki/Runge-Kutta_method
def RK4(f):
    return lambda t, y, dt: (
            lambda dy1: (
            lambda dy2: (
            lambda dy3: (
            lambda dy4: (dy1 + 2*dy2 + 2*dy3 + dy4)/6
            )( dt * f( t + dt  , y + dy3   ) )
      )( dt * f( t + dt/2, y + dy2/2 ) )
      )( dt * f( t + dt/2, y + dy1/2 ) )
      )( dt * f( t       , y         ) )



def run():
  T = []
  Y = []

  a = 0.2
  b = 0.1
  c = 10
  tau = 17
  yi = 0.8

  t, y, dt = 0., yi, .1

  def fn(t, y):
    # TODO: clean up below logic
    i = int(tau / dt)
    py = Y[-i] if len(Y) > i else yi
    return ((a * py) / (1 + py**c)) - (b * y)

  dy = RK4(fn)

  with open(FILE_PREFIX + "_all.csv", 'wb') as allFile:
    with open(FILE_PREFIX + "_train.csv", 'wb') as trainFile:
      with open(FILE_PREFIX + "_test.csv", 'wb') as testFile:
        allWriter = csv.writer(allFile)
        trainWriter = csv.writer(trainFile)
        testWriter = csv.writer(testFile)

        for writer in (allWriter, trainWriter, testWriter):
          writer.writerow(["y"])
          writer.writerow(["float"])
          writer.writerow([""])

        while True:
          T.append(t)
          Y.append(y)

          allWriter.writerow([y])

          if t >= 200 and t < 3200:
            trainWriter.writerow([y])
          elif t >= 5000 and t < 5500:
            testWriter.writerow([y])

          if abs(round(t) - t) < 1e-5:
            print("y(%2.1f)\t= %4.6f \t" % ( t, y ))
          t, y = t + dt, y + dy( t, y, dt )

          if t > 5500:
            break

  # pyplot.plot(T, Y)
  # pyplot.xlim(5000, 5250)
  # pyplot.show()



if __name__ == "__main__":
  run()
