class Plots(object):

  def __init__(self, field, vehicle, scorer, model):
    self.field = field
    self.vehicle = vehicle
    self.scorer = scorer
    self.model = model

    import matplotlib.pyplot as plt
    self.plt = plt
    import matplotlib.cm as cm
    self.cm = cm

    from pylab import rcParams
    rcParams['figure.figsize'] = 8, 14

    self.plt.ion()
    self.plt.show()

    self.positions = []
    self.sensorValues = []
    self.sensorNoiseAmounts = []
    self.motorValues = []
    self.motorNoiseAmounts = []
    self.scores = []
    self.goalValues = []


  def update(self):
    self.positions.append(self.vehicle.position)
    self.sensorValues.append(self.vehicle.sensorValue)
    self.sensorNoiseAmounts.append(self.vehicle.sensor.noiseAmount)
    self.motorValues.append(self.vehicle.motorValue)
    self.motorNoiseAmounts.append(self.vehicle.motor.noiseAmount)
    self.scores.append(self.scorer.score)
    self.goalValues.append(self.model.currentGoalValue)


  def render(self):
    rows = 4
    cols = 2
    self.plt.clf()

    self.plt.subplot(rows, cols, 1)
    self.plt.ylim(min(self.sensorValues), max(self.sensorValues))
    self._plot(self.goalValues, "Goal")

    self.plt.subplot(rows, cols, 2)
    self._plot(self.positions, "Position")

    self.plt.subplot(rows, cols, 3)
    self._plot(self.sensorValues, "Sensor value")

    self.plt.subplot(rows, cols, 4)
    self._plot(self.sensorNoiseAmounts, "Sensor noise")

    self.plt.subplot(rows, cols, 5)
    self._plot(self.motorValues, "Motor value")

    self.plt.subplot(rows, cols, 6)
    self._plot(self.motorNoiseAmounts, "Motor noise")

    self.plt.draw()


  def _plot(self, data, title):
    self.plt.title(title)
    self.plt.xlim(0, len(data))
    self.plt.plot(range(len(data)), data)



class PositionPredictionPlots(Plots):

  def render(self):
    self.plt.figure(1)
    super(PositionPredictionPlots, self).render()

    # self.plt.figure(2)
    # self.model.tm.mmGetCellActivityPlot()

    self.plt.figure(2)
    self.plt.clf()
    rows = 4
    cols = 1

    data = self.model.tm.mmGetTraceActiveColumns().data
    overlaps = [len(a & b) for a, b in zip(data[:-1], data[1:])]
    self.plt.subplot(rows, cols, 1)
    self.plt.title("Active columns overlap with t-1")
    self.plt.plot(range(len(overlaps)), overlaps)

    data = self.model.tm.mmGetTraceUnpredictedActiveColumns().makeCountsTrace().data
    self.plt.subplot(rows, cols, 2)
    self.plt.title("Unpredicted active columns")
    self.plt.plot(range(len(data)), data)

    data = self.model.tm.mmGetTracePredictedActiveColumns().makeCountsTrace().data
    self.plt.subplot(rows, cols, 3)
    self.plt.title("Predicted active columns")
    self.plt.plot(range(len(data)), data)

    data = self.model.tm.mmGetTracePredictedInactiveColumns().makeCountsTrace().data
    self.plt.subplot(rows, cols, 4)
    self.plt.title("Predicted inactive columns")
    self.plt.plot(range(len(data)), data)

    self.plt.draw()



class PositionBehaviorPlots(Plots):

  def __init__(self, field, vehicle, scorer, model):
    super(PositionBehaviorPlots, self).__init__(field, vehicle, scorer, model)
    self.activeSensorColumns = []

    self.activeBehavior = self.model.bm.activeBehavior
    self.lastActiveBehavior = self.model.bm.activeBehavior
    self.motor = self.model.bm.motor
    self.lastMotor = self.model.bm.motor

    self.reconstructedBehaviorOverlap = []
    self.reconstructedMotorOverlap = []


  def update(self):
    super(PositionBehaviorPlots, self).update()
    self.activeSensorColumns.append(self.model.bm.activeSensorColumns)

    self.lastActiveBehavior = self.activeBehavior
    self.activeBehavior = self.model.bm.activeBehavior
    self.lastMotor = self.motor
    self.motor = self.model.bm.motor

    reconstructedBehavior = self.model.bm.reconstructedBehavior
    behaviorOverlap = (self.lastActiveBehavior * reconstructedBehavior).sum()
    behaviorOverlap /= reconstructedBehavior.sum()
    self.reconstructedBehaviorOverlap.append(behaviorOverlap)

    reconstructedMotor = self.model.bm.reconstructedMotor
    motorOverlap = (self.lastMotor * reconstructedMotor).sum()
    motorOverlap /= reconstructedMotor.sum()
    self.reconstructedMotorOverlap.append(motorOverlap)


  def render(self):
    self.plt.figure(1)
    super(PositionBehaviorPlots, self).render()

    self.plt.figure(2)
    self.plt.clf()
    rows = 9
    cols = 1

    data = self.activeSensorColumns
    overlaps = [len(a & b) for a, b in zip(data[:-1], data[1:])]
    self.plt.subplot(rows, cols, 1)
    self._plot(overlaps,
               "Active columns overlap with t-1")

    self.plt.subplot(rows, cols, 2)
    self._imshow(self._imageData(self.model.bm.goal),
                 "Goal")

    self.plt.subplot(rows, cols, 3)
    self._imshow(self.model.bm.activeBehavior.transpose(),
                 "Active behavior")

    self.plt.subplot(rows, cols, 4)
    self._imshow(self.model.bm.reconstructedBehavior.transpose(),
                 "Reconstructed behavior")

    self.plt.subplot(rows, cols, 5)
    self._imshow(self.model.bm.learningBehavior.transpose(),
                 "Learning behavior")

    self.plt.subplot(rows, cols, 6)
    self._imshow(self.model.bm.motor.reshape([1, self.model.bm.motor.size]),
                 "Motor")

    self.plt.subplot(rows, cols, 7)
    self._imshow(self._imageData(self.model.bm.reconstructedMotor),
                 "Reconstructed motor")

    self.plt.subplot(rows, cols, 8)
    self._plot(self.reconstructedBehaviorOverlap,
               "Reconstructed behavior overlap")

    self.plt.subplot(rows, cols, 9)
    self._plot(self.reconstructedMotorOverlap,
               "Reconstructed motor overlap")

    self.plt.draw()

    self.plt.figure(3)
    self.plt.clf()
    rows = 3
    cols = 1

    self.plt.subplot(rows, cols, 1)
    self._imshow(self.model.bm.goalToBehaviorFlat(),
                 "Goal to behavior connections")

    self.plt.subplot(rows, cols, 2)
    self._imshow(self.model.bm.behaviorToMotorFlat(),
                 "Behavior to motor connections")

    self.plt.subplot(rows, cols, 3)
    self._imshow(self.model.bm.motorToBehaviorFlat(),
                 "Motor to behavior connections")

    self.plt.draw()


  @staticmethod
  def _imageData(array):
    return array.reshape([1, array.size])


  def _imshow(self, data, title):
    self.plt.title(title)
    self.plt.imshow(data,
                    cmap=self.cm.Greys,
                    interpolation="nearest",
                    aspect='auto')

