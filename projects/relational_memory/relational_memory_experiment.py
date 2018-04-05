# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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

"""Algorithm and experiment for sensorimotor learning using grid cells."""

import argparse
import collections
import contextlib
import itertools
import json
import os
import time

import capnp
import numpy as np

from htmresearch.algorithms.apical_tiebreak_temporal_memory import (
    ApicalTiebreakPairMemory as TemporalMemory)
from htmresearch.algorithms.column_pooler import ColumnPooler
from nupic.algorithms.connections import Connections
from nupic.algorithms.knn_classifier import KNNClassifier

from relational_memory_log_capnp import RelationalMemoryLog


@contextlib.contextmanager
def dummyContextMgr():
    yield None


def bind(cell1, cell2, moduleDimensions):
  """Return transform index for given cells.

  Convert to coordinate space, calculate transform, and convert back to an
  index. In coordinate space, the transform represents `C2 - C1`.
  """
  cell1Coords = np.unravel_index(cell1, moduleDimensions)
  cell2Coords = np.unravel_index(cell2, moduleDimensions)
  transformCoords = [(c2 - c1) % m
                     for c1, c2, m in itertools.izip(cell1Coords, cell2Coords,
                                                     moduleDimensions)]
  return np.ravel_multi_index(transformCoords, moduleDimensions)


def unbind(cell1, transform, moduleDimensions):
  """Return the cell index corresponding to the other half of the transform.

  Assumes that `transform = bind(cell1, cell2)` and, given `cell1` and
  `transform`, returns `cell2`.
  """
  cell1Coords = np.unravel_index(cell1, moduleDimensions)
  transformCoords = np.unravel_index(transform, moduleDimensions)
  cell2Coords = [(t + c1) % m
                 for c1, t, m in itertools.izip(cell1Coords, transformCoords,
                                                moduleDimensions)]
  return np.ravel_multi_index(cell2Coords, moduleDimensions)


def pathIntegrate(cellIdx, moduleDimensions, delta):
  cellCoords = np.unravel_index(cellIdx, moduleDimensions)
  updatedCoords = [(c + d) % m
                   for c, d, m in itertools.izip(cellCoords, delta,
                                                 moduleDimensions)]
  return np.ravel_multi_index(updatedCoords, moduleDimensions)


def getGlobalIndices(modules, moduleSize):
  globalIndices = []
  for m, activeCells in enumerate(modules):
    globalIndices.extend([c + (m * moduleSize) for c in activeCells])
  return globalIndices


class RelationalMemory(object):

  def __init__(self, l4N, l4W, numModules, moduleDimensions,
               maxActivePerModule, l6ActivationThreshold):
    self.numModules = numModules
    self.moduleDimensions = moduleDimensions
    self._cellsPerModule = np.prod(moduleDimensions)
    self.maxActivePerModule = maxActivePerModule
    self.l4N = l4N
    self.l4W = l4W
    self.l6ActivationThreshold = l6ActivationThreshold

    self.l4TM = TemporalMemory(
        columnCount=l4N,
        basalInputSize=numModules*self._cellsPerModule,
        cellsPerColumn=4,
        #activationThreshold=int(numModules / 2) + 1,
        #reducedBasalThreshold=int(numModules / 2) + 1,
        activationThreshold=1,
        reducedBasalThreshold=1,
        initialPermanence=1.0,
        connectedPermanence=0.5,
        minThreshold=1,
        sampleSize=numModules,
        permanenceIncrement=1.0,
        permanenceDecrement=0.0,
    )
    self.l6Connections = [Connections(numCells=self._cellsPerModule)
                          for _ in xrange(numModules)]

    self.pooler = ColumnPooler(
      inputWidth=self.numModules*self._cellsPerModule,
    )

    self.classifier = KNNClassifier(k=1, distanceMethod="rawOverlap")
    #self.classifier = KNNClassifier(k=1, distanceMethod="norm")

    # Active state
    self.activeL6Cells = [[] for _ in xrange(numModules)]
    self.activeL5Cells = [[] for _ in xrange(numModules)]
    self.predictedL6Cells = [set([]) for _ in xrange(numModules)]

    # Debug state
    self.activeL6BeforeMotor = [[] for _ in xrange(numModules)]
    self.l6ToL4Map = collections.defaultdict(list)

  def reset(self):
    self.activeL6Cells = [[] for _ in xrange(self.numModules)]
    self.activeL5Cells = [[] for _ in xrange(self.numModules)]
    self.predictedL6Cells = [set([]) for _ in xrange(self.numModules)]
    self.l4TM.reset()
    self.pooler.reset()

  def trainFeatures(self, sensoryInputs):
    # Randomly assign bilateral connections and zero others
    for sense in sensoryInputs:
      # Choose L6 cells randomly
      activeL6Cells = [[np.random.randint(self._cellsPerModule)]
                       for _ in xrange(self.numModules)]
      l4BasalInput = getGlobalIndices(activeL6Cells, self._cellsPerModule)

      # Learn L6->L4 connections
      self.l4TM.compute(activeColumns=sense, basalInput=l4BasalInput, learn=True)
      self.l4TM.compute(activeColumns=sense, basalInput=l4BasalInput, learn=True)
      self.l4TM.compute(activeColumns=sense, basalInput=l4BasalInput, learn=True)
      self.l4TM.compute(activeColumns=sense, basalInput=l4BasalInput, learn=True)
      activeL4Cells = self.l4TM.getActiveCells()
      # Debug: store the map
      for l6Cell in itertools.chain(*activeL6Cells):
        self.l6ToL4Map[l6Cell].extend(activeL4Cells)
      # Learn L4->L6 connections
      for l6Cells, connections in zip(activeL6Cells, self.l6Connections):
        # Assumes one cell active per L6 module when training features
        segment = connections.createSegment(l6Cells[0])
        for l4Cell in activeL4Cells:
          connections.createSynapse(segment, l4Cell, 1.0)

  def compute(self, ff, motor, objClass, outputFile):
    """Run one iteration of the online sensorimotor algorithm.

    This function has three stages:

    - The FEEDFORWARD pass drives

    Prerequisites: `trainFeatures` must have been run already

    :param ff: feedforward sensory input
    :param motor: the motor command for next move, in the form of delta
        coordinates
    :param objClass: the object class to train the classifier, or None
        if not learning
    """
    delta = motor

    # FEEDFORWARD

    # Determine active feature representation in l4, using lateral input
    # from l6 previous step feedback
    l4BasalInput = getGlobalIndices(self.predictedL6Cells, self._cellsPerModule)
    self.l4TM.compute(activeColumns=ff, basalInput=l4BasalInput,
                      learn=False)
    predictedL4Cells = self.l4TM.getPredictedCells()
    activeL4Cells = self.l4TM.getActiveCells()

    # Drive L6 activation from l4
    for m, connections in enumerate(self.l6Connections):
      newCells = []
      activeConnectedPerSegment = connections.computeActivity(activeL4Cells, 0.5)[0]
      for flatIdx, activeConnected in enumerate(activeConnectedPerSegment):
        if activeConnected >= self.l6ActivationThreshold:
          cellIdx = connections.segmentForFlatIdx(flatIdx).cell
          newCells.append(cellIdx)

      #for cell in newCells:
      #  print connections.segmentsForCell(cell)
      #print newCells
      #assert len(newCells) <= 1

      self.activeL6Cells[m].insert(0, newCells)
      # TODO: This is the number of steps, not necessarily the number of cells
      lenBefore = len(self.activeL6Cells[m])
      del self.activeL6Cells[m][self.maxActivePerModule:]
      lenAfter = len(self.activeL6Cells[m])
      #assert lenBefore == lenAfter, "Debug assert to check that we aren't hitting limit on L6 activity. Can remove when we set max active low enough relative to object size (times number of train/test iterations)"

    self.activeL6BeforeMotor = [list(itertools.chain(*l6Module))
                                for l6Module in self.activeL6Cells]

    # Replace l5 activity with new transforms
    self.activeL5Cells = []
    for activeL6Module in self.activeL6Cells:
      transforms = set()
      for newCell in activeL6Module[0]:
        for prevCell in itertools.chain(*activeL6Module[1:]):
          if newCell == prevCell:
            continue
          # Transform from prev to new
          t1 = bind(prevCell, newCell, self.moduleDimensions)
          transforms.add(t1)
          # Transform from new to prev
          t2 = bind(newCell, prevCell, self.moduleDimensions)
          transforms.add(t2)
      self.activeL5Cells.append(list(transforms))


    # Pool into object representation
    classifierLearn = True if objClass is not None else False
    globalL5ActiveCells = sorted(getGlobalIndices(self.activeL5Cells, self._cellsPerModule))
    self.pooler.compute(feedforwardInput=globalL5ActiveCells,
                        learn=classifierLearn,
                        predictedInput=globalL5ActiveCells)

    # Classifier
    classifierInput = np.zeros((self.pooler.numberOfCells(),), dtype=np.uint32)
    classifierInput[self.pooler.getActiveCells()] = 1
    #print classifierInput.nonzero()
    #print self.pooler.getActiveCells()
    #print
    self.prediction = self.classifier.infer(classifierInput)
    if objClass is not None:
      self.classifier.learn(classifierInput, objClass)

    # MOTOR

    # Update L6 based on motor command
    numActivePerModuleBefore = [sum([len(cells) for cells in active]) for active in self.activeL6Cells]

    self.activeL6Cells = [
        [[pathIntegrate(c, self.moduleDimensions, delta)
          for c in steps]
         for steps in prevActiveCells]
        for prevActiveCells in self.activeL6Cells]

    numActivePerModuleAfter = [sum([len(cells) for cells in active]) for active in self.activeL6Cells]
    assert numActivePerModuleAfter == numActivePerModuleBefore

    # FEEDBACK

    # Get all transforms associated with object
    # TODO: Get transforms from object in addition to current activity
    predictiveTransforms = [l5Active for l5Active in self.activeL5Cells]

    # Get set of predicted l6 representations (including already active)
    # and store them for next step l4 compute
    self.predictedL6Cells = []
    for l6, l5 in itertools.izip(self.activeL6Cells, predictiveTransforms):
      predictedCells = []
      for activeL6Cell in set(itertools.chain(*l6)):
        for activeL5Cell in l5:
          predictedCell = unbind(activeL6Cell, activeL5Cell, self.moduleDimensions)
          predictedCells.append(predictedCell)
      self.predictedL6Cells.append(set(
          list(itertools.chain(*l6)) + predictedCells))

    # Log this step
    if outputFile:
      log = RelationalMemoryLog.new_message()
      log.ts = time.time()
      sensationProto = log.init("sensation", len(ff))
      for i in xrange(len(ff)):
        sensationProto[i] = int(ff[i])
      predictedL4Proto = log.init("predictedL4", len(predictedL4Cells))
      for i in xrange(len(predictedL4Cells)):
        predictedL4Proto[i] = int(predictedL4Cells[i])
      activeL4Proto = log.init("activeL4", len(activeL4Cells))
      for i in xrange(len(activeL4Cells)):
        activeL4Proto[i] = int(activeL4Cells[i])
      activeL6HistoryProto = log.init("activeL6History", len(self.activeL6Cells))
      for i in xrange(len(self.activeL6Cells)):
        activeL6ModuleProto = activeL6HistoryProto.init(i, len(self.activeL6Cells[i]))
        for j in xrange(len(self.activeL6Cells[i])):
          activeL6ModuleStepProto = activeL6ModuleProto.init(j, len(self.activeL6Cells[i][j]))
          for k in xrange(len(self.activeL6Cells[i][j])):
            activeL6ModuleStepProto[k] = int(self.activeL6Cells[i][j][k])
      activeL5Proto = log.init("activeL5", len(self.activeL5Cells))
      for i in xrange(len(self.activeL5Cells)):
        activeL5ModuleProto = activeL5Proto.init(i, len(self.activeL5Cells[i]))
        for j in xrange(len(self.activeL5Cells[i])):
          activeL5ModuleProto[j] = int(self.activeL5Cells[i][j])

      classifierResults = [(i, distance)
                           for i, distance in enumerate(self.prediction[2])
                           if distance is not None]
      classifierResultsProto = log.init("classifierResults",
                                        len(classifierResults))
      for i in xrange(len(classifierResults)):
        classifierResultProto = classifierResultsProto[i]
        classifierResultProto.label = classifierResults[i][0]
        classifierResultProto.distance = float(classifierResults[i][1])

      motorDeltaProto = log.init("motorDelta", len(delta))
      for i in xrange(len(delta)):
        motorDeltaProto[i] = int(delta[i])
      predictedL6Proto = log.init("predictedL6", len(self.predictedL6Cells))
      for i in xrange(len(self.predictedL6Cells)):
        predictedL6ModuleProto = predictedL6Proto.init(i, len(self.predictedL6Cells[i]))
        for j, c in enumerate(self.predictedL6Cells[i]):
          predictedL6ModuleProto[j] = int(c)

      json.dump(log.to_dict(), outputFile)
      outputFile.write("\n")


def runExperiment(numObjects, numFeatures, testNoise, l6ActivationThreshold, outputPath):
  objectDims = (4, 4)

  numTrainingPasses = 5
  numTestingPasses = 3
  maxActivePerModule = 10
  skipFirst = 0
  numModules = 2
  moduleDims = (10, 10)

  l4N = 1024
  l4W = 20

  # Create a network
  net = RelationalMemory(l4N=l4N, l4W=l4W, numModules=numModules,
             moduleDimensions=moduleDims,
             maxActivePerModule=maxActivePerModule,
             l6ActivationThreshold=l6ActivationThreshold)

  # Create set of sensory inputs and objects
  sensoryIndices = np.array(range(l4N), dtype=np.uint32)
  sensoryInputs = []
  for _ in xrange(numFeatures):
    np.random.shuffle(sensoryIndices)
    sensoryInputs.append(np.copy(sensoryIndices[:l4W]))

  objects = []
  for objClass in xrange(numObjects):
    obj = np.zeros(objectDims + (l4W,), dtype=np.uint32)
    for i in xrange(np.prod(objectDims)):
      coords = np.unravel_index(i, objectDims)
      sensationIdx = int(np.random.random() * len(sensoryInputs))
      obj[coords][:] = sensoryInputs[sensationIdx]
    objects.append((objClass, obj))

  # Batch train on the features
  net.trainFeatures(sensoryInputs)

  # Train with the objects
  #if outputPath is not None:
  #  outputFile = open(outputPath, "w")
  with open(outputPath, "w") if outputPath else dummyContextMgr() as outputFile:
    for objClass, obj in objects:
      net.reset()
      objSensations = 0
      for _ in xrange(numTrainingPasses):
        locationSequence = [np.unravel_index(i, objectDims)
                            for i in xrange(np.prod(objectDims))]
        np.random.shuffle(locationSequence)
        inputSequence = []
        for i in xrange(len(locationSequence)):
          coords = locationSequence[i]
          if i + 1 < len(locationSequence):
            nextCoords = locationSequence[i+1]
          else:
            nextCoords = coords
          delta = [n - c for n, c in itertools.izip(nextCoords, coords)]
          inputSequence.append((coords, delta))
        for coords, nextMove in inputSequence:
          sensation = obj[coords]
          objSensations += 1
          if objSensations <= skipFirst:
            oClass = None
          else:
            oClass = objClass
          net.compute(sensation, nextMove, oClass, outputFile)
          #assert len(net.activeL6BeforeMotor) == 1
          #print i+1, len(net.activeL6BeforeMotor[0])
          #assert len(net.activeL6BeforeMotor[0]) == (i+1), (
          #    "expected {} but got {}".format(
          #        i+1,
          #        len(net.activeL6BeforeMotor[0])
          #    ))
          #assert len(net.activeL6BeforeMotor[0][-1]) == i, len(net.activeL6BeforeMotor[0])
      if (objClass + 1) % 100 == 0:
        print "Completed training {} objects".format(objClass + 1)

  # Test by doing one pass over objects and checking best guess from classifier against actual object
  # TODO: Try multiple passes and take best guess
  correct = 0
  total = 0
  averagePredictedL4 = []
  averageActiveL4 = []
  for objClass, obj in objects:
    net.reset()
    objSensations = 0
    numActiveL4 = 0
    numPredictedL4 = 0
    objClassifications = collections.defaultdict(int)
    for _ in xrange(numTestingPasses):
      locationSequence = [np.unravel_index(i, objectDims)
                          for i in xrange(np.prod(objectDims))]
      np.random.shuffle(locationSequence)

      # Add noise
      noisePositions = np.array(range(len(locationSequence)), dtype=np.uint32)
      np.random.shuffle(noisePositions)

      inputSequence = []
      for i in xrange(len(locationSequence)):
        coords = locationSequence[i]
        if i + 1 < len(locationSequence):
          nextCoords = locationSequence[i+1]
        else:
          nextCoords = coords
        delta = [n - c for n, c in itertools.izip(nextCoords, coords)]
        inputSequence.append((coords, delta))
      for i, (coords, nextMove) in enumerate(inputSequence):
        if i in noisePositions[:testNoise]:
          sensation = sensoryInputs[np.random.randint(len(sensoryInputs))]
        else:
          sensation = obj[coords]
        net.compute(sensation, nextMove, None, None)
        objSensations += 1
        activeTMCells = net.l4TM.getActiveCells()
        numActiveL4 += len(activeTMCells)
        if objSensations > skipFirst:
          objClassifications[net.prediction[0]] += 1
          predictedActiveTMCells = net.l4TM.getPredictedActiveCells()
          numPredictedL4 += len(predictedActiveTMCells)
    bestGuess = sorted(objClassifications.iteritems(), key=lambda x: x[1])[-1][0]
    if bestGuess == objClass:
      correct += 1
    total += 1
    if total % 100 == 0:
      print "Completed testing {} objects".format(total)

    averageActiveL4.append(float(numActiveL4) / float(objSensations))
    averagePredictedL4.append(float(numPredictedL4) / float(objSensations - skipFirst))

  print "Average active L4 cells: {}".format(sum(averageActiveL4) / len(averageActiveL4))
  print "Average predicted L4 cells: {}".format(sum(averagePredictedL4) / len(averagePredictedL4))
  accuracy = float(correct) / float(total)
  print "Test accuracy after {}: {}".format(total, accuracy)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--objects", default=1000, required=True, type=int)
  parser.add_argument("--features", required=True, type=int)
  parser.add_argument("--noise", default=0, required=True, type=int)
  parser.add_argument("--output", default=None, help="path relative to cwd to save log to")
  parser.add_argument("--l6thresh", default=6, type=int, help="path relative to cwd to save log to")
  args = parser.parse_args()

  numObjects = args.objects
  numFeatures = args.features
  testNoise = args.noise
  if args.output is not None:
    outputPath = os.path.join(os.getcwd(), args.output)
  else:
    outputPath = None

  runExperiment(numObjects, numFeatures, testNoise, args.l6thresh, outputPath)
