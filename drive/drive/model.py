import numpy

from sensorimotor.general_temporal_memory import GeneralTemporalMemory
from nupic.research.monitor_mixin.temporal_memory_monitor_mixin import (
  TemporalMemoryMonitorMixin)
class MonitoredGeneralTemporalMemory(TemporalMemoryMonitorMixin,
                                     GeneralTemporalMemory): pass
from sensorimotor.behavior_memory import BehaviorMemory

from nupic.encoders.coordinate import CoordinateEncoder



class Model(object):

  def __init__(self, motorValues=range(-4, 4+1)):
    self.motorValues = motorValues


  def update(self, sensorValue, motorValue, goal=None):
    pass



class PositionPredictionModel(Model):

  def __init__(self, motorValues=range(-4, 4+1),
               sparsity=0.02, encoderResolution=1.0, tmParams=None):
    super(PositionPredictionModel, self).__init__(motorValues=motorValues)
    tmParams = tmParams or {}
    self.tm = MonitoredGeneralTemporalMemory(mmName="TM", **tmParams)
    self.n = self.tm.numberOfColumns()
    self.w = int(self.n * sparsity) + 1
    self.encoderResolution = encoderResolution
    self.sensorEncoder = CoordinateEncoder(w=self.w, n=self.n)
    self.motorEncoder = CoordinateEncoder(w=self.w, n=self.n)


  def update(self, sensorValue, motorValue, goalValue=None):
    scale = 100
    radius = int(self.encoderResolution * scale)
    sensorInput = (numpy.array([int(sensorValue * scale)]), radius)
    motorInput = (numpy.array([int(motorValue * scale)]), radius)
    sensorPattern = set(self.sensorEncoder.encode(sensorInput).nonzero()[0])
    motorPattern = set(self.motorEncoder.encode(motorInput).nonzero()[0])

    self.tm.compute(sensorPattern,
                    activeExternalCells=motorPattern,
                    formInternalConnections=True,
                    learn=True)



class PositionBehaviorModel(Model):

  def __init__(self, motorValues=range(-4, 4+1),
               sparsity=0.02, encoderResolution=0.1, bmParams=None):
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
      print motorValue, self.decodeMotor()
      return self.decodeMotor()


  def decodeMotor(self):
    idx = self.bm.motor.argmax()
    return self.motorValues[idx]
