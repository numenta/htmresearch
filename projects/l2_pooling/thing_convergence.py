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

"""
This file is used to debug specific Thing experiments.
"""

import os
import numpy as np
import pprint
from htmresearch.support.logging_decorator import LoggingDecorator
from htmresearch.frameworks.layers.l2_l4_inference import (
  L4L2Experiment, rerunExperimentFromLogfile)

def loadThingObjects():
  objDataPath = '../../htmresearch/data/thing/'
  objFiles = [f for f in os.listdir(objDataPath)
               if os.path.isfile(os.path.join(objDataPath, f))]

  thingObjects = {}
  for f in objFiles:
    objName = f.split('.')[0]
    objFile = open('{}/{}'.format(objDataPath, f))

    sensationList = []
    for line in objFile.readlines():
      sense = line.split('=>')[1].strip(' ').strip('\n')
      location = sense.split('],[')[0].strip('[')
      feature = sense.split('],[')[1].strip(']')
      location = np.fromstring(location, sep=',', dtype=np.uint8)
      feature = np.fromstring(feature, sep=',', dtype=np.uint8)

      sensationList.append({0: [location.tolist(), feature.tolist()]})

    thingObjects[objName] = sensationList
  return thingObjects


def getObjectPair(objectName, pointNumber):
  """
  Returns the location/feature pair for point pointNumber in object objName

  """
  return thingObjects[objectName][pointNumber][0]


def createExperiment(logFilename):
  # Typically this would be done by Thing
  exp = L4L2Experiment("shared_features", logCalls=True)
  exp.learnObjects(thingObjects)

  LoggingDecorator.save(exp.callLog, logFilename)


def debugExperiment(logFile, thingObjects):
  """
  Debug a thing experiment given a logFile
  """

  exp = rerunExperimentFromLogfile(logFile)
  exp.logCalls = False

  L2Representations = exp.objectL2Representations
  print "Learned object representations:"
  pprint.pprint(L2Representations, width=400)
  print "=========================="

  objects = thingObjects.keys()
  for i in range(len(objects)):
    print "\nRun inference with a point on the {}".format(objects[i])
    sensationList = [
      {0: getObjectPair(objects[i], 0)},
      {0: getObjectPair(objects[i], 1)},
      {0: getObjectPair(objects[i], 2)},
    ]
    exp.infer(sensationList, reset= False)
    print "Output for {}: {}".format(objects[i], exp.getL2Representations())
    for i in range(len(objects)):
      print "Intersection with {}:{}".format(
        objects[i],
        len(exp.getL2Representations()[0] & L2Representations[objects[i]][0]))
    exp.sendReset()



if __name__ == "__main__":
  thingObjects = loadThingObjects()

  # Mimic thing, which will create a log file
  createExperiment("callLog.pkl")

  # Recreate class from log and debug experiment
  debugExperiment("callLog.pkl", thingObjects)


