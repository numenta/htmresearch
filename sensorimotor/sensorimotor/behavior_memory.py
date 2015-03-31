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

import matplotlib.pyplot as plt
import matplotlib.cm as cm



DEBUG = False



class BehaviorMemory(object):

  def __init__(self,
               numMotorColumns=1024,
               numSensorColumns=1024,
               numCellsPerSensorColumn=32,
               goalToBehaviorLearningRate=0.3,
               behaviorToMotorLearningRate=0.3,
               motorToBehaviorLearningRate=0.3,
               behaviorDecayRate=0.33):
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

    self.activeMotorColumns = set()
    self.activeSensorColumns = set()
    self.activeGoalColumns = set()

    if DEBUG:
      plt.ion()
      plt.show()


  @staticmethod
  def _initWeights(shape):
    weights = numpy.random.normal(0.5, 0.5, shape)
    weights[weights < 0] = 0
    weights[weights > 1] = 1

    return weights


  @staticmethod
  def _makeArray(s, length):
    arr = numpy.zeros(length)
    arr[list(s)] = 1
    return arr


  @staticmethod
  def _reinforce(weights, active, learningRate):
    delta = active * learningRate
    total = weights.sum()
    weights += delta
    weights /= (weights.sum() / total)


  def compute(self, activeMotorColumns, activeSensorColumns, activeGoalColumns):
    self.activeMotorColumns = activeMotorColumns
    self.activeSensorColumns = activeSensorColumns
    self.activeGoalColumns = activeGoalColumns

    motorPattern = self._makeArray(activeMotorColumns, self.numMotorColumns)
    sensorPattern = self._makeArray(activeSensorColumns, self.numSensorColumns)

    self.motor = motorPattern
    self.goal = sensorPattern

    if len(activeGoalColumns):
      goalPattern = self._makeArray(activeGoalColumns, self.numSensorColumns)
      self.goal = goalPattern
      self._updateActiveBehaviorFromGoal(sensorPattern)
      self._updateMotorFromActiveBehavior()
    else:
      self._reinforceGoalToBehavior()
      self._updateActiveBehaviorFromMotor(sensorPattern)
      self._updateLearningBehavior()
      self._reinforceBehaviorToMotor()
      self._reinforceMotorToBehavior()

      if DEBUG:
        plt.clf()
        plt.figure(1)
        numBehaviorCells = self.numSensorColumns * self.numCellsPerSensorColumn
        plt.imshow(self.goalToBehavior.reshape(self.numGoalCells, numBehaviorCells), cmap=cm.Greys, interpolation="nearest")
        # plt.imshow(self.activeBehavior, cmap=cm.Greys, interpolation="nearest")
        # plt.imshow(self.learningBehavior, cmap=cm.Greys, interpolation="nearest")
        # plt.imshow(self.behaviorToMotor.reshape(self.numSensorColumns, self.numCellsPerSensorColumn * self.numMotorCells), cmap=cm.Greys, interpolation="nearest")
        # plt.imshow(self.motorToBehavior.reshape(self.numCellsPerSensorColumn * self.numMotorCells, self.numSensorColumns), cmap=cm.Greys, interpolation="nearest")
        plt.draw()


  def _reinforceGoalToBehavior(self):
    for column in self.goal.nonzero()[0]:
      weights = self.goalToBehavior[column]
      self._reinforce(weights,
                      self.learningBehavior,
                      self.goalToBehaviorLearningRate)


  def _updateActiveBehaviorFromMotor(self, sensorPattern):
    numBehaviorCells = self.numSensorColumns * self.numCellsPerSensorColumn
    motorToBehaviorFlat = self.motorToBehavior.reshape([self.numMotorCells,
                                                        numBehaviorCells])
    activity = numpy.dot(self.motor, motorToBehaviorFlat)
    activity = activity.reshape([self.numSensorColumns,
                                self.numCellsPerSensorColumn])
    winnerCells = numpy.argmax(activity, axis=1)

    self.activeBehavior.fill(0)
    for column in sensorPattern.nonzero()[0]:
      winnerCell = winnerCells[column]
      self.activeBehavior[column][winnerCell] = 1


  def _updateLearningBehavior(self):
    self.learningBehavior = self.learningBehavior * self.behaviorDecayRate
    self.learningBehavior += self.activeBehavior


  def _reinforceBehaviorToMotor(self):
    for cell in numpy.transpose(self.activeBehavior.nonzero()):
      weights = self.behaviorToMotor[cell[0], cell[1]]
      self._reinforce(weights,
                      self.motor,
                      self.behaviorToMotorLearningRate)


  def _reinforceMotorToBehavior(self):
    for cell in self.motor.nonzero()[0]:
      weights = self.motorToBehavior[cell]
      self._reinforce(weights,
                      self.activeBehavior,
                      self.motorToBehaviorLearningRate)


  def _updateActiveBehaviorFromGoal(self, sensorPattern):
    pass


  def _updateMotorFromActiveBehavior(self):
    pass
