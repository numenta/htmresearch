# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
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

import numpy



class BehaviorMemory(object):

  def __init__(self,
               numMotorColumns=1024,
               numSensorColumns=1024,
               numCellsPerSensorColumn=32,
               goalToBehaviorLearningRate=0.3,
               behaviorToMotorLearningRate=0.3,
               motorToBehaviorLearningRate=0.3,
               behaviorDecayRate=0.10):
    self.numMotorColumns = numMotorColumns
    self.numSensorColumns = numSensorColumns
    self.numCellsPerSensorColumn = numCellsPerSensorColumn
    self.goalToBehaviorLearningRate = goalToBehaviorLearningRate
    self.behaviorToMotorLearningRate = behaviorToMotorLearningRate
    self.motorToBehaviorLearningRate = motorToBehaviorLearningRate
    self.behaviorDecayRate = behaviorDecayRate

    self.numMotorCells = numMotorColumns
    self.numGoalCells = numSensorColumns

    self.motor = numpy.zeros(self.numMotorCells)
    self.learningBehavior = numpy.zeros([self.numSensorColumns,
                                         self.numCellsPerSensorColumn])
    self.activeBehavior = numpy.zeros([self.numSensorColumns,
                                       self.numCellsPerSensorColumn])
    self.goal = numpy.zeros(self.numGoalCells)

    self.goalToBehavior = self._initWeights([self.numGoalCells,
                                             self.numSensorColumns,
                                             self.numCellsPerSensorColumn])
    self.behaviorToMotor = self._initWeights([self.numSensorColumns,
                                              self.numCellsPerSensorColumn,
                                              self.numMotorCells])
    self.motorToBehavior = self._initWeights([self.numMotorCells,
                                              self.numSensorColumns,
                                              self.numCellsPerSensorColumn])

    # For debugging
    self.activeMotorColumns = set()
    self.activeSensorColumns = set()
    self.activeGoalColumns = set()

    self.prevActiveSensorColumns = set()

    self.reconstructedBehavior = numpy.zeros([self.numSensorColumns,
                                              self.numCellsPerSensorColumn])
    self.reconstructedMotor = numpy.zeros(self.numMotorCells)


  @staticmethod
  def _initWeights(shape):
    weights = numpy.random.normal(0.2, 0.1, shape)
    numpy.clip(weights, 0, 1, out=weights)

    return weights


  @staticmethod
  def _makeArray(s, length):
    arr = numpy.zeros(length)
    arr[list(s)] = 1
    return arr


  @classmethod
  def _reinforce(cls, weights, active, learningRate):
    delta = active * learningRate
    return cls._addAndNormalize(weights, delta)


  @staticmethod
  def _addAndNormalize(numbers, delta):
    total = numbers.sum()
    numbers += delta
    numpy.clip(numbers, 0, 1, out=numbers)
    numbers /= (numbers.sum() / total)
    return numbers


  def compute(self, activeMotorColumns, activeSensorColumns, activeGoalColumns):
    self.prevActiveSensorColumns = self.activeSensorColumns
    self.activeMotorColumns = activeMotorColumns
    self.activeSensorColumns = activeSensorColumns
    self.activeGoalColumns = activeGoalColumns

    motorPattern = self._makeArray(activeMotorColumns, self.numMotorColumns)
    sensorPattern = self._makeArray(activeSensorColumns, self.numSensorColumns)
    goalPattern = self._makeArray(activeGoalColumns, self.numSensorColumns)

    self.motor = motorPattern
    self.goal = goalPattern if len(activeGoalColumns) else sensorPattern

    prevSensorPattern = self._makeArray(self.prevActiveSensorColumns,
                                        self.numSensorColumns)
    self.reconstructedBehavior = self._computeBehaviorFromGoal(
      self.goal, prevSensorPattern)
    self.reconstructedMotor = self._computeMotorFromBehavior(
      self.reconstructedBehavior)

    if len(activeGoalColumns):
      self.activeBehavior = self._computeBehaviorFromGoal(self.goal,
                                                          sensorPattern)
      self.motor = self._computeMotorFromBehavior(self.activeBehavior)
    else:
      self._reinforceGoalToBehavior(self.goal, self.learningBehavior)
      self.activeBehavior = self._computeBehaviorFromMotor(self.motor,
                                                           sensorPattern)
      self.learningBehavior = self._computeLearningBehavior(
        self.learningBehavior, self.activeBehavior)
      self._reinforceBehaviorToMotor(self.activeBehavior, self.motor)
      self._reinforceMotorToBehavior(self.motor, self.activeBehavior)


  def numBehaviorCells(self):
    return self.numSensorColumns * self.numCellsPerSensorColumn


  def goalToBehaviorFlat(self):
    return self.goalToBehavior.reshape([self.numGoalCells,
                                        self.numBehaviorCells()])


  def motorToBehaviorFlat(self):
    return self.motorToBehavior.reshape([self.numMotorCells,
                                         self.numBehaviorCells()])


  def behaviorToMotorFlat(self):
    return self.behaviorToMotor.reshape([self.numBehaviorCells(),
                                         self.numMotorCells])


  def _reinforceGoalToBehavior(self, goal, behavior):
    for column in goal.nonzero()[0]:
      weights = self.goalToBehavior[column]
      self._reinforce(weights,
                      behavior,
                      self.goalToBehaviorLearningRate)


  def _computeBehaviorFromMotor(self, motor, sensorPattern):
    activity = numpy.dot(motor, self.motorToBehaviorFlat())
    activity = activity.reshape([self.numSensorColumns,
                                self.numCellsPerSensorColumn])
    winnerCells = numpy.argmax(activity, axis=1)

    behavior = numpy.zeros([self.numSensorColumns,
                            self.numCellsPerSensorColumn])

    for column in sensorPattern.nonzero()[0]:
      winnerCell = winnerCells[column]
      behavior[column][winnerCell] = 1

    return behavior


  def _computeLearningBehavior(self, learningBehavior, activeBehavior):
    behavior = learningBehavior * (1 - self.behaviorDecayRate)
    behavior += activeBehavior

    winnerCells = behavior.argmax(axis=1)
    sparseBehavior = numpy.zeros(behavior.shape)

    for column in range(len(winnerCells)):
      winnerCell = winnerCells[column]
      sparseBehavior[column][winnerCell] = behavior[column][winnerCell]

    numpy.clip(sparseBehavior, 0, 1, out=sparseBehavior)
    return sparseBehavior


  def _reinforceBehaviorToMotor(self, behavior, motor):
    for cell in numpy.transpose(behavior.nonzero()):
      weights = self.behaviorToMotor[cell[0], cell[1]]
      self._reinforce(weights,
                      motor,
                      self.behaviorToMotorLearningRate)


  def _reinforceMotorToBehavior(self, motor, behavior):
    for cell in motor.nonzero()[0]:
      weights = self.motorToBehavior[cell]
      self._reinforce(weights,
                      behavior,
                      self.motorToBehaviorLearningRate)


  def _computeBehaviorFromGoal(self, goal, sensorPattern):
    activity = numpy.dot(goal, self.goalToBehaviorFlat())
    activity = activity.reshape([self.numSensorColumns,
                                self.numCellsPerSensorColumn])
    behavior = numpy.zeros(activity.shape)

    for column in sensorPattern.nonzero()[0]:
      behavior[column][:] = activity[column]

    return behavior


  def _computeMotorFromBehavior(self, behavior):
    motor = numpy.dot(behavior.flatten(),
                      self.behaviorToMotorFlat())
    motor /= self.motor.sum()
    return motor
