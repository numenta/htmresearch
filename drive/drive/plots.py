class Plots(object):

  def __init__(self, field, vehicle, scorer, model):
    self.field = field
    self.vehicle = vehicle
    self.scorer = scorer
    self.model = model

    import matplotlib.pyplot as plt
    self.plt = plt
    # import matplotlib.cm as cm
    # self.cm = cm

    self.plt.ion()
    self.plt.show()

    self.positions = []
    self.sensorValues = []
    self.sensorNoiseAmounts = []
    self.motorValues = []
    self.motorNoiseAmounts = []
    self.scores = []


  def update(self):
    self.positions.append(self.vehicle.position)
    self.sensorValues.append(self.vehicle.sensorValue)
    self.sensorNoiseAmounts.append(self.vehicle.sensor.noiseAmount)
    self.motorValues.append(self.vehicle.motorValue)
    self.motorNoiseAmounts.append(self.vehicle.motor.noiseAmount)
    self.scores.append(self.scorer.score)


  def render(self):
    rows = 4
    cols = 2
    self.plt.clf()

    self.plt.subplot(rows, cols, 1)
    self.plt.ylabel("Sensor value")
    self.plt.plot(range(len(self.sensorValues)), self.sensorValues)

    self.plt.subplot(rows, cols, 2)
    self.plt.ylabel("Sensor noise")
    self.plt.plot(range(len(self.sensorNoiseAmounts)), self.sensorNoiseAmounts)

    self.plt.subplot(rows, cols, 3)
    self.plt.ylabel("Motor value")
    self.plt.plot(range(len(self.motorValues)), self.motorValues)

    self.plt.subplot(rows, cols, 4)
    self.plt.ylabel("Motor noise")
    self.plt.plot(range(len(self.motorNoiseAmounts)), self.motorNoiseAmounts)

    self.plt.subplot(rows, cols, 5)
    self.plt.ylabel("Position")
    self.plt.ylim([0, self.field.width])
    self.plt.plot(range(len(self.positions)), self.positions)

    self.plt.subplot(rows, cols, 6)
    self.plt.ylabel("Score")
    self.plt.plot(range(len(self.scores)), self.scores)

    self.plt.draw()



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
    self.plt.ylabel("Active columns overlap with t-1")
    self.plt.plot(range(len(overlaps)), overlaps)

    data = self.model.tm.mmGetTraceUnpredictedActiveColumns().makeCountsTrace().data
    self.plt.subplot(rows, cols, 2)
    self.plt.ylabel("Unpredicted active columns")
    self.plt.plot(range(len(data)), data)

    data = self.model.tm.mmGetTracePredictedActiveColumns().makeCountsTrace().data
    self.plt.subplot(rows, cols, 3)
    self.plt.ylabel("Predicted active columns")
    self.plt.plot(range(len(data)), data)

    data = self.model.tm.mmGetTracePredictedInactiveColumns().makeCountsTrace().data
    self.plt.subplot(rows, cols, 4)
    self.plt.ylabel("Predicted inactive columns")
    self.plt.plot(range(len(data)), data)

    self.plt.draw()



class PositionBehaviorPlots(Plots):

  def __init__(self, field, vehicle, scorer, model):
    super(PositionBehaviorPlots, self).__init__(field, vehicle, scorer, model)
    self.activeSensorColumns = []


  def update(self):
    super(PositionBehaviorPlots, self).update()
    self.activeSensorColumns.append(self.model.bm.activeSensorColumns)


  def render(self):
    self.plt.figure(1)
    super(PositionBehaviorPlots, self).render()

    self.plt.figure(2)
    self.plt.clf()
    rows = 1
    cols = 1

    data = self.activeSensorColumns
    overlaps = [len(a & b) for a, b in zip(data[:-1], data[1:])]
    self.plt.subplot(rows, cols, 1)
    self.plt.ylabel("Active columns overlap with t-1")
    self.plt.plot(range(len(overlaps)), overlaps)

    self.plt.draw()
