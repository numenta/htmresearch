class Game(object):

  def __init__(self, field, vehicle, scorer, model, goal=None,
               logs=None, plots=None, graphics=None,
               plotEvery=25):
    self.field = field
    self.vehicle = vehicle
    self.scorer = scorer
    self.model = model
    self.goal = goal

    self.logs = logs
    self.plots = plots
    self.graphics = graphics
    self.logs = logs
    self.graphics = graphics

    self.plotEvery = plotEvery


  def run(self):
    i = 0
    while True:
      try:
        if self.logs is not None:
          self.logs.log()

        if self.plots is not None:
          self.plots.update()

          if i % self.plotEvery == 0:
            self.plots.render()

        if self.graphics is not None:
          self.graphics.update()
          self.graphics.render()

        self.vehicle.tick()
        self.scorer.update()
        motorValue = self.model.update(self.vehicle.sensorValue,
                                       self.vehicle.motorValue,
                                       goalValue=self.goal)

        if motorValue is not None:
          self.vehicle.setMotorValue(motorValue)

        i += 1
      except KeyboardInterrupt:
        key = raw_input("Enter a command [(q)uit, (g)oal]: ")
        print key
        if key == "q":
          break
        elif key == "g":
          self.goal = raw_input("Enter new goal: ")
