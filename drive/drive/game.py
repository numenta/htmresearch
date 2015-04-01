class Game(object):

  def __init__(self, field, vehicle, scorer, model, goal=None,
               logs=None, plots=None, graphics=None,
               plotEvery=25,
               manualRun=False):
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
    self.manualRun = manualRun

    self.plotsEnabled = True


  def run(self):
    i = 0
    while True:
      try:
        if self.graphics is not None:
          self.graphics.update()
          self.graphics.render()

        if self.manualRun and self.vehicle.graphics.currentKey is None:
          continue

        self.vehicle.tick()
        self.scorer.update()
        motorValue = self.model.tick(self.vehicle.sensorValue,
                                     self.vehicle.motorValue,
                                     goalValue=self.goal)

        if motorValue is not None:
          self.vehicle.setMotorValue(motorValue)

        if self.logs is not None:
          self.logs.log()

        if self.plots is not None:
          self.plots.update()

          if i % self.plotEvery == 0 and self.plotsEnabled:
            self.plots.render()

        i += 1
      except KeyboardInterrupt:
        print "Paused."
        key = raw_input("Enter a command [(q)uit, (g)oal, (p)lots-toggle]: ")

        if key == "q":
          break
        elif key == "g":
          try:
            self.goal = int(raw_input("Enter new goal (non-number will be None): "))
          except ValueError:
            self.goal = None
          print "Set new goal:", self.goal
        elif key == "p":
          self.plotsEnabled = not self.plotsEnabled

        print "Resuming..."
