#!/usr/bin/env python
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

"""
This file demonstrate convergence of single vs multiple columns
"""

import random
import matplotlib.pyplot as plt
import matplotlib as mpl

from htmresearch.frameworks.layers.l2_l4_inference import L4L2Experiment
from htmresearch.frameworks.layers.object_machine_factory import (
  createObjectMachine
)

mpl.rcParams['pdf.fonttype'] = 42


plt.ion()
plt.close('all')


def plotActivity(l2ActiveCellsMultiColumn):
  fig, axs = plt.subplots(1, 3)
  for c in range(len(l2ActiveCellsMultiColumn[0])):
    for step in range(len(l2ActiveCellsMultiColumn)):
      activeCellList = list(l2ActiveCellsMultiColumn[step][c])
      for activeCells in activeCellList:
        axs[c].plot(step, activeCells, '-ko')
    axs[c].set_title('column {}'.format(c))
    axs[c].set_xlabel('Time')
  axs[0].set_ylabel('Neuron #')
  return fig



if __name__ == "__main__":
  numColumns = [1, 3]
  numFeatures = 3
  numPoints = 10
  numLocations = 10
  numObjects = 10

  objectMachine = createObjectMachine(
    machineType="simple",
    numInputBits=20,
    sensorInputSize=1024,
    externalInputSize=1024,
    numCorticalColumns=3,
    seed=37,
  )
  objectMachine.createRandomObjects(numObjects, numPoints=numPoints,
                              numLocations=numLocations,
                              numFeatures=numFeatures)


  objects = objectMachine.provideObjectsToLearn()

  # single-out the inputs to the column #1
  objectsSingleColumn = {}
  for i in range(numObjects):
    featureLocations = []
    for j in range(numLocations):
      featureLocations.append({0: objects[i][j][0]})
    objectsSingleColumn[i] = featureLocations

  # we will run two experiments side by side, with either single column
  # or 3 columns
  exp3 = L4L2Experiment(
    'three_column',
    numCorticalColumns=3,
    seed=1
  )

  exp1 = L4L2Experiment(
    'single_column',
    numCorticalColumns=1,
    seed=1
  )

  print "train single column "
  exp1.learnObjects(objectsSingleColumn)
  print "train multi-column "
  exp3.learnObjects(objects)

  objectId = 0
  obj = objectMachine[objectId]

  # Create sequence of sensations for this object for all columns
  numColumns = 3
  objectSensations = {}
  for c in range(numColumns):
    objectCopy = [pair for pair in obj]
    random.shuffle(objectCopy)
    # stay multiple steps on each sensation
    sensations = []
    for pair in objectCopy:
        sensations.append(pair)
    objectSensations[c] = sensations

  sensationStepsSingleColumn = []
  sensationStepsMultiColumn = []
  for step in xrange(len(objectSensations[0])):
    pairs = [
      objectSensations[col][step] for col in xrange(numColumns)
      ]
    sdrs = objectMachine._getSDRPairs(pairs)
    sdrSingleColumn = {0: sdrs[0]}
    sensationStepsMultiColumn.append(sdrs)
    sensationStepsSingleColumn.append(sdrs)

  print "inference: multi-columns "
  exp3.sendReset()
  l2ActiveCellsMultiColumn = []
  L2ActiveCellNVsTimeMultiColumn = []
  for sensation in sensationStepsMultiColumn:
    exp3.infer([sensation], objectName=objectId, reset=False)
    l2ActiveCellsMultiColumn.append(exp3.getL2Representations())
    activeCellNum = 0
    for c in range(numColumns):
      activeCellNum += len(exp3.getL2Representations()[c])
    L2ActiveCellNVsTimeMultiColumn.append(activeCellNum/numColumns)

  print "inference: single column "
  exp1.sendReset()
  l2ActiveCellsSingleColumn = []
  L2ActiveCellNVsTimeSingleColumn = []
  for sensation in sensationStepsSingleColumn:
    exp1.infer([sensation], objectName=objectId, reset=False)
    l2ActiveCellsSingleColumn.append(exp1.getL2Representations())
    L2ActiveCellNVsTimeSingleColumn.append(len(exp1.getL2Representations()[0]))

  plt.figure()
  plt.plot(L2ActiveCellNVsTimeSingleColumn, label='single column')
  plt.plot(L2ActiveCellNVsTimeMultiColumn, label='3 columns')
  plt.ylabel(' L2 active bits ')
  plt.ylim([0, max(L2ActiveCellNVsTimeSingleColumn)])
  plt.legend()
  plt.savefig('plots/L2_active_cell_vs_time.pdf')

  numSteps = len(l2ActiveCellsSingleColumn)
  numL2Cells = exp1.config["L2Params"]['cellCount']

  fig1 = plotActivity(l2ActiveCellsMultiColumn)
  st = fig1.suptitle("Single Cortical Column")
  plt.savefig('plots/L2_active_cell_single_column.pdf')

  fig3 = plotActivity(l2ActiveCellsSingleColumn)
  st = fig1.suptitle("Three Cortical Column")
  plt.savefig('plots/L2_active_cell_multi_column.pdf')