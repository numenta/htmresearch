import numpy

from sensorimotor.extended_temporal_memory import ApicalTiebreakPairMemory

from htmresearch.support.apical_tm_pair_monitor_mixin import (
  ApicalTMPairMonitorMixin)

class MonitoredApicalTiebreakPairMemory(
  ApicalTMPairMonitorMixin, ApicalTiebreakPairMemory):
  pass
from sensorimotor.behavior_memory import BehaviorMemory

from nupic.encoders.coordinate import CoordinateEncoder



class Model(object):

  def __init__(self, motorValues=range(-4, 4+1)):
    self.motorValues = motorValues

    self.currentGoalValue = None


  def tick(self, sensorValue, motorValue, goalValue=None):
    self.currentGoalValue = goalValue
    return self.update(sensorValue, motorValue, goalValue=goalValue)


  def update(self, sensorValue, motorValue, goalValue=None):
    """To override"""
    return None


DEFAULT_TM_PARAMS = {
  "basalInputDimensions": (999999,), # Dodge the input checking.
}

class PositionPredictionModel(Model):

  def __init__(self, motorValues=range(-4, 4+1),
               sparsity=0.02, encoderResolution=1.0, tmParams=None):
    super(PositionPredictionModel, self).__init__(motorValues=motorValues)

    tmParams = dict(DEFAULT_TM_PARAMS)
    tmParams.update(tmParams or {})
    self.tm = MonitoredApicalTiebreakPairMemory(mmName="TM", **tmParams)
    self.n = self.tm.numberOfColumns()
    self.w = int(self.n * sparsity) + 1
    self.encoderResolution = encoderResolution
    self.sensorEncoder = CoordinateEncoder(w=self.w, n=self.n)
    self.motorEncoder = CoordinateEncoder(w=self.w, n=self.n)
    self.prevMotorPattern = ()


  def update(self, sensorValue, motorValue, goalValue=None):
    scale = 100
    radius = int(self.encoderResolution * scale)
    sensorInput = (numpy.array([int(sensorValue * scale)]), radius)
    motorInput = (numpy.array([int(motorValue * scale)]), radius)
    sensorPattern = set(self.sensorEncoder.encode(sensorInput).nonzero()[0])
    motorPattern = set(self.motorEncoder.encode(motorInput).nonzero()[0])

    self.tm.compute(sensorPattern,
                    activeCellsExternalBasal=motorPattern,
                    reinforceCandidatesExternalBasal=self.prevMotorPattern,
                    growthCandidatesExternalBasal=self.prevMotorPattern,
                    learn=True)

    self.prevMotorPattern = motorPattern



class PositionBehaviorModel(Model):

  def __init__(self, motorValues=range(-4, 4+1),
               sparsity=0.02, encoderResolution=0.5, bmParams=None):
    super(PositionBehaviorModel, self).__init__(motorValues=motorValues)
    self.encoderResolution = encoderResolution
    bmParams = bmParams or {}

    numMotorColumns = len(self.motorValues)
    bmParams["numMotorColumns"] = numMotorColumns
    self.bm = BehaviorMemory(**bmParams)

    self.sensorN = self.bm.numSensorColumns
    self.sensorW = int(self.sensorN * sparsity) + 1

    self.sensorEncoder = CoordinateEncoder(w=self.sensorW, n=self.sensorN)


  def update(self, sensorValue, motorValue, goalValue=None):
    motorPattern = set([self.motorValues.index(motorValue)])

    scale = 100
    radius = int(self.encoderResolution * scale)
    sensorInput = (numpy.array([int(sensorValue * scale)]), radius)
    sensorPattern = set(self.sensorEncoder.encode(sensorInput).nonzero()[0])

    goalPattern = set()
    if goalValue is not None:
      goalInput = (numpy.array([int(goalValue * scale)]), radius)
      goalPattern = set(self.sensorEncoder.encode(goalInput).nonzero()[0])

    self.bm.compute(motorPattern, sensorPattern, goalPattern)

    if goalValue is not None:
      return self.decodeMotor()


  def decodeMotor(self):
    idx = self.bm.motor.argmax()
    return self.motorValues[idx]
