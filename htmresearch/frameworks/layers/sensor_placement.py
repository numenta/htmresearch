# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import random



def greedySensorPositions(numSensors, numLocations):
  """
  Returns an infinite sequence of sensor placements.

  Each return value is a tuple of locations, one location per sensor.

  Positions are selected using a simple greedy algorithm. The first priority of
  the algorithm is "touch every position an equal number of times". The second
  priority is "each individual sensor should touch each position an equal number
  of times".

  @param numSensors (int)
  The number of sensors

  @param numLocations (int)
  The number of locations

  @return (generator of tuples)
  The next locations for each sensor. The tuple's length is `numSensors`.
  """

  locationViewCounts = [0] * numLocations
  locationViewCountsBySensor = [[0] * numLocations
                                for _ in xrange(numSensors)]

  placement = random.sample(xrange(numLocations), numSensors)

  while True:
    yield tuple(placement)

    # Update statistics.
    for sensor, location in enumerate(placement):
      locationViewCounts[location] += 1
      locationViewCountsBySensor[sensor][location] += 1

    # Choose the locations with the lowest view counts. Break ties randomly.
    nextLocationsRanked = sorted(xrange(numLocations),
                                 key=lambda x: (locationViewCounts[x],
                                                random.random()))
    nextLocations = nextLocationsRanked[:numSensors]

    # For each sensor (in random order), choose the location that has touched
    # the least, breaking ties randomly.
    sensors = range(numSensors)
    random.shuffle(sensors)
    for sensor in sensors:
      viewCount = min(locationViewCountsBySensor[sensor][location]
                      for location in nextLocations)
      location = random.choice([x for x in nextLocations
                                if locationViewCountsBySensor[sensor][x]
                                == viewCount])
      nextLocations.remove(location)
      placement[sensor] = location
