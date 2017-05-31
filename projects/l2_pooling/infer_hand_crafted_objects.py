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

from collections import defaultdict
import itertools
import random
import os

import matplotlib.pyplot as plt

from htmresearch.frameworks.layers.l2_l4_inference import L4L2Experiment
from htmresearch.frameworks.layers.sensor_placement import greedySensorPositions



FEATURES = ("A", "B")

LOCATIONS = tuple(xrange(9))

# Every object shares 3 feature-locations with every other object.
OBJECTS = {"Object 1": ("A", "A", "A",
                        "A", "A", "A",
                        "A", "A", "A"),
           "Object 2": ("A", "A", "A",
                        "B", "B", "B",
                        "B", "B", "B"),
           "Object 3": ("B", "B", "B",
                        "A", "A", "A",
                        "B", "B", "B"),
           "Object 4": ("B", "B", "B",
                        "B", "B", "B",
                        "A", "A", "A"),}

TIMESTEPS_PER_SENSATION = 3

def experiment(numColumns, sampleSize):
  locationSDRsByColumn = [dict((name,
                                set(random.sample(xrange(1024), 40)))
                               for name in LOCATIONS)
                          for _ in xrange(numColumns)]

  featureSDRsByColumn = [dict((name,
                               set(random.sample(xrange(1024), 40)))
                              for name in FEATURES)
                         for _ in xrange(numColumns)]

  exp = L4L2Experiment(
    "Hello",
    numCorticalColumns=numColumns,
    L2Overrides={
      "sampleSizeDistal": sampleSize,
    },
    seed=random.randint(2048, 4096)
  )

  exp.learnObjects(dict((objectName,
                         [dict((column,
                                (locationSDRsByColumn[column][location],
                                 featureSDRsByColumn[column][features[location]]))
                               for column in xrange(numColumns))
                          for location in LOCATIONS])
                        for objectName, features in OBJECTS.iteritems()))

  objectName = "Object 1"
  features = OBJECTS[objectName]
  inferredL2 = exp.objectL2Representations[objectName]

  touchCount = 0

  for sensorPositions in greedySensorPositions(numColumns, len(LOCATIONS)):
    sensation = dict(
      (column,
       (locationSDRsByColumn[column][sensorPositions[column]],
        featureSDRsByColumn[column][features[sensorPositions[column]]]))
      for column in xrange(numColumns))
    exp.infer([sensation]*TIMESTEPS_PER_SENSATION,
              reset=False, objectName=objectName)

    touchCount += 1

    if exp.getL2Representations() == inferredL2:
      print "Inferred object after %d touches" % touchCount
      return touchCount

    if touchCount >= 60:
      print "Never inferred object"
      return None


def go():
  numColumnsOptions = range(1, len(LOCATIONS) + 1)
  configs = (
    ("Placeholder 13", 13),
    ("Placeholder 20", 20),
    ("Placeholder 30", 30),
    ("Placeholder everything", -1),
  )

  numTouchesLog = defaultdict(list)

  for config in configs:
    _, sampleSize = config
    print "sampleSize %d" % sampleSize

    for numColumns in numColumnsOptions:
      print "%d columns" % numColumns

      for _ in xrange(10):
        numTouches = experiment(numColumns, sampleSize)
        numTouchesLog[(numColumns, config)].append(numTouches)

  averages = dict((k,
                   sum(numsTouches) / float(len(numsTouches)))
                  for k, numsTouches in numTouchesLog.iteritems())

  plt.figure()
  colorList = dict(zip(configs,
                       ('r', 'k', 'g', 'b')))
  markerList = dict(zip(configs,
                        ('o', '*', 'D', 'x')))

  for config in configs:
    plt.plot(numColumnsOptions,
             [averages[(numColumns, config)]
              for numColumns in numColumnsOptions],
             color=colorList[config],
             marker=markerList[config])

  plt.legend([description for description, _ in configs],
             loc="upper right")
  plt.xlabel("Columns")
  plt.xticks(numColumnsOptions)
  plt.ylabel("Number of touches")
  plt.yticks([0, 1, 2, 3, 4, 5])
  plt.title("Touches until inference")

  plotPath = os.path.join("plots", "infer_hand_crafted_objects.pdf")
  plt.savefig(plotPath)
  plt.close()


if __name__ == "__main__":
  go()
