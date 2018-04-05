# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for self.software code, the
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
# along with self.program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""Attempt to learn motor mappings using sensory input to detect cycles."""

import collections
import csv

import numpy as np

MEMORY_LENGTH = 10000



class Algorithm(object):

  def __init__(self, locationWidth, motorWidth):
    self.locWidth = locationWidth
    self.numMotorCommands = (motorWidth ** 2) - 1
    self.numOffsets = (locationWidth ** 2) - 1
    self.numLocations = locationWidth ** 2

    # Hard-coded list of possible location offsets identified by index
    coordinates = []
    for i in xrange(-(locationWidth / 2), (locationWidth / 2) + 1):
      for j in xrange(-(locationWidth / 2), (locationWidth / 2) + 1):
        if i != 0 or j != 0:
          coordinates.append((i, j))
    self.locOffsetValues = coordinates

    # LEARNED ALGORITHM STATE

    # Learned mapping from motor command ID to current offset ID
    self.motorMap = {}
    for m in xrange(self.numMotorCommands):
      self.motorMap[m] = np.random.randint(self.numOffsets)

    self.iteration = 0

    # WORKING ALGORITHM STATE

    # Current location cell
    self.loc = [0, 0]

    # Map from feature to most recent location
    self.memory = {}
    # Map from feature to the iteration that it was seen
    self.memoryIteration = {}
    # How many instances of each feature are in the history
    self.count = collections.defaultdict(int)
    # List of features in short term memory
    self.featHistory = []
    # List of motor commands in short term memory
    self.motorHistory = []

    # Learning coordination state
    # Maps motor command to [total, correct] cycle history
    self.motorStats = dict([(i, [0, 0]) for i in xrange(self.numMotorCommands)])
    self.clearMotorStats()
    # Keep track of when we started a mini-goal
    self.miniGoalStart = 0
    # None for random exploration, or the motor command that we are remapping
    self.miniGoal = None
    self.miniGoalCurrent = None
    # Remaining mappings to try
    self.miniGoalRemaining = []
    # Remember how good each possible mapping was (map from possible command to % correct)
    self.miniGoalStats = {}


  def clearMotorStats(self):
    for i in self.motorStats.keys():
      self.motorStats[i] = [0, 0]


  def getMotorAccuracy(self, m):
    total, correct = self.motorStats[m]
    return float(correct) / float(total)


  def updateLoc(self, currentMotor):
    offset = self.motorMap[currentMotor]
    xd, yd = self.locOffsetValues[offset]
    self.loc = [v % self.locWidth for v in (self.loc[0] + xd, self.loc[1] + yd)]


  def updateHistory(self, feat):
    self.motorHistory.append(feat)
    self.featHistory.append(feat)
    if len(self.motorHistory) > MEMORY_LENGTH:
      self.count[self.featHistory[0]] -= 1
      del self.motorHistory[0]
      del self.featHistory[0]
    self.count[feat] += 1
    self.memory[feat] = self.loc
    self.memoryIteration[feat] = self.iteration


  def updateMotorStats(self, feat, consistentCycle):
    cycleStart = self.memoryIteration[feat]
    relCycleStart = cycleStart - self.iteration
    motorCommands = self.motorHistory[relCycleStart:]
    for m in motorCommands:
      total, correct = self.motorStats.get(m, [0, 0])
      total += 1
      correct += int(consistentCycle)
      self.motorStats[m] = [total, correct]


  def run(self, feat, m):
    # BASIC SETTLING

    # Correct location if necessary
    if feat in self.memory:
      # We have a cycle

      if self.memory[feat] == self.loc:
        # Consistent cycle in location!

        self.updateMotorStats(feat, True)
      else:
        # No location layer cycle!

        self.updateMotorStats(feat, False)
        self.loc = self.memory[feat]

    # Update feat/loc history
    self.updateHistory(feat)

    # MINI-GOAL

    if (self.iteration - self.miniGoalStart) % 10000 == 0:
      if self.miniGoal is None:
        # Choose command to update
        #self.miniGoal = self.getWorstMotor()
        self.miniGoal = np.random.randint(self.numMotorCommands)
        self.miniGoalRemaining = range(self.numOffsets)
        np.random.shuffle(self.miniGoalRemaining)
        self.miniGoalCurrent = self.miniGoalRemaining.pop()
        print "changing mapping {} to test offset {}".format(self.miniGoal, self.miniGoalCurrent)
        self.motorMap[self.miniGoal] = self.miniGoalCurrent
      else:
        self.miniGoalStats[self.miniGoalCurrent] = self.getMotorAccuracy(self.miniGoalCurrent)
        if self.miniGoalRemaining:
          self.miniGoalCurrent = self.miniGoalRemaining.pop()
          print "changing mapping {} to test offset {}".format(self.miniGoal, self.miniGoalCurrent)
          self.motorMap[self.miniGoal] = self.miniGoalCurrent
        else:
          # Set the motor mapping to the best option from our mini goal experiments
          sortedResults = sorted(self.miniGoalStats.iteritems(),
                                 key=lambda x: x[1], reverse=True)
          totalCorrectness = 0.0
          for o, v in sortedResults:
            totalCorrectness += v
          randomIndex = (np.random.random() ** 3) * totalCorrectness
          accum = 0.0
          for o, v in sortedResults:
            accum += v
            if accum >= randomIndex:
              bestOffset = o
              break
          else:
            print "WTF?!"
          # TODO: hack
          bestOffset = sortedResults[0][0]
          print self.miniGoalStats
          print "changing mapping {} to best offset {}".format(self.miniGoal, bestOffset)
          self.motorMap[self.miniGoal] = bestOffset
          # Spend some time with new mapping before we choose the next to update
          self.miniGoal = None
          self.miniGoalStats = {}

        self.clearMotorStats()

      self.miniGoalStart = self.iteration

    # BASIC UPDATE

    # Update location based on new motor command
    self.updateLoc(m)

    # HOUSEKEEPING
    self.iteration += 1


  @staticmethod
  def computeBasisTransform(locOffset1, motorOffset1, locOffset2, motorOffset2):
    locOffsetX1, locOffsetY1 = locOffset1
    motorOffsetX1, motorOffsetY1 = motorOffset1
    locOffsetX2, locOffsetY2 = locOffset2
    motorOffsetX2, motorOffsetY2 = motorOffset2

    # Solve system of equations to determine basis transform matrix
    coefficients = [
      [locOffsetX1, 0, locOffsetY1, 0],
      [0, locOffsetX1, 0, locOffsetY1],
      [locOffsetX2, 0, locOffsetY2, 0],
      [0, locOffsetX2, 0, locOffsetY2],
    ]
    ordinates = [motorOffsetX1, motorOffsetY1, motorOffsetX2, motorOffsetY2]
    try:
      basisValues = np.linalg.solve(coefficients, ordinates)
    except np.linalg.linalg.LinAlgError:
      return None
    basisTransform = basisValues.reshape((2, 2))
    assert basisValues[0] == basisTransform[0][0]
    assert basisValues[1] == basisTransform[0][1]
    assert basisValues[2] == basisTransform[1][0]
    assert basisValues[3] == basisTransform[1][1]

    return basisTransform


  def measureConsistency(self, worldMotorMap, worldOffsetValues):
    best = 0.0
    bestTransform = None
    for m1 in worldMotorMap.keys():
      for m2 in worldMotorMap.keys():
        # Find the basis transform for these two location/world offset pairs
        locOffset1 = self.locOffsetValues[self.motorMap[m1]]
        motorOffset1 = worldOffsetValues[worldMotorMap[m1]]
        locOffset2 = self.locOffsetValues[self.motorMap[m2]]
        motorOffset2 = worldOffsetValues[worldMotorMap[m2]]
        basisTransform = self.computeBasisTransform(locOffset1, motorOffset1, locOffset2, motorOffset2)
        if basisTransform is None:
          continue

        # Check how many mappings match this transform
        n = 0
        consistent = 0
        for m in worldMotorMap.keys():
          n += 1
          worldOffset = worldOffsetValues[worldMotorMap[m]]
          locOffset = self.locOffsetValues[self.motorMap[m]]
          transformed = np.dot(locOffset, basisTransform)
          shift = int(self.locWidth / 2)
          transformed = [((v + shift) % self.locWidth) - shift for v in transformed]
          if np.allclose(transformed, worldOffset):
            consistent += 1
        score = float(consistent) / float(n)
        if score > best:
          best = score
          bestTransform = basisTransform

    return best, bestTransform



class Experiment(object):


  def __init__(self, locationWidth, motorWidth, worldWidth):
    self.locWidth = locationWidth
    self.motorWidth = motorWidth
    self.worldWidth = worldWidth

    self.numMotorCommands = (motorWidth ** 2) - 1

    self.algorithm = Algorithm(locationWidth, motorWidth)

    # Hard-coded list of possible location offsets identified by index
    offsets = []
    for i in xrange(-(locationWidth / 2), (locationWidth / 2) + 1):
      for j in xrange(-(locationWidth / 2), (locationWidth / 2) + 1):
        if i != 0 or j != 0:
          offsets.append((i, j))
    self.worldOffsetValues = offsets

    # Hard-coded mapping from motor command ID to real-world change
    self.motorMap = dict([(v, v) for v in xrange(self.numMotorCommands)])
    print "world motor map:"
    for m, o in self.motorMap.iteritems():
      print "{}: {}".format(m, ",".join(str(v) for v in self.worldOffsetValues[o]))

    # WORLD STATE

    # Current world position
    self.pos = [0, 0]

    # Build a world
    sensoryInputs = range(worldWidth ** 2)
    np.random.shuffle(sensoryInputs)
    world = np.array(sensoryInputs, dtype=int)
    world.resize((worldWidth, worldWidth))
    self.world = world

    # EXPERIMENT STATE

    self.i = 0


  def getNextMotor(self):
    candidates = []
    for m, offset in self.motorMap.iteritems():
      xd, yd = self.worldOffsetValues[offset]
      x = self.pos[0] + xd
      y = self.pos[1] + yd
      if x >= 0 and y >= 0 and x < len(self.world[0]) and y < len(self.world):
        #weight = self.offsetMap[m][1] ** 2
        #total, correct = self.motorStats
        #weight = (float(correct) / float(total))
        weight = 1.0
        candidates.append((weight, m))

    # Select a point in the cumulative weights
    totalWeights = 0.0
    for w, _ in candidates:
      totalWeights += w
    point = np.random.random() * totalWeights

    # Find which motor command this corresponds to and return it
    cum = 0.0
    for w, m in candidates:
      cum = cum + w
      if cum > point:
        return m
    raise ValueError("Should never get here...")


  def updatePos(self, currentMotor):
    xd, yd = self.worldOffsetValues[self.motorMap[currentMotor]]
    self.pos = (self.pos[0] + xd, self.pos[1] + yd)


  def measureConsistency(self):
    return self.algorithm.measureConsistency(self.motorMap, self.worldOffsetValues)


  def runOne(self):
    currentMotor = self.getNextMotor()

    feat = self.world[self.pos[0]][self.pos[1]]
    self.algorithm.run(feat, currentMotor)

    self.updatePos(currentMotor)

    self.i += 1



def testParams(locationWidth, motorWidth, worldWidth):
  iterationsToPerfect = []
  numTrials = 10
  for trial in xrange(numTrials):
    exp = Experiment(
      locationWidth=locationWidth,
      motorWidth=motorWidth,
      worldWidth=worldWidth,
    )
    for i in xrange(100000000):
      exp.runOne()
      if (i + 1) % 10000 == 0:
        consistency, basisT = exp.measureConsistency()
        print "consistency: ", consistency
        if consistency > 0.99:
          print "got perfect!!!"
          iterationsToPerfect.append(i)
          break
    else:
      print "never got perfect"
      iterationsToPerfect.append(i)
  assert len(iterationsToPerfect) == numTrials
  total = 0.0
  for iterations in iterationsToPerfect:
    total += float(iterations)
  return total / float(numTrials)



if __name__ == "__main__":
  #paramSet = [
  #  (3, 10, 0.05, 0.05, 0.01),
  #  (3, 10, 0.05, 0.05, 0.02),
  #]
  #for params in paramSet:
  #  print "params: ", params
  #  averageIterationsToPerfect = testParams(*params)
  #  print "avg iterations: ", averageIterationsToPerfect
  #  print
  testParams(3, 3, 5)
