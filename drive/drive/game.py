import sys



class Game(object):

  def __init__(self, field, vehicle, scorer, model, goal=None,
               logs=None, plots=None, graphics=None,
               runSpeed=1.0,
               plotEvery=25, plotsEnabled=True,
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

    self.runSpeed = runSpeed
    self.plotEvery = plotEvery
    self.manualRun = manualRun

    self.plotsEnabled = plotsEnabled
    self.autoGoal = False


  def run(self):
    i = 0
    t = 0
    while True:
      if self.graphics is not None:
        self.graphics.update()
        self.graphics.render()

      if self.graphics is not None:
        if self.graphics.paused:
          self.pause()

        if self.graphics.currentLeftClick is not None:
          posX = float(self.graphics.currentLeftClick[0])
          self.setGoal(posX / self.graphics.size[0] * self.field.width)

        if self.graphics.currentRightClick is not None:
          self.setGoal(None)

      if self.manualRun and self.graphics.currentKey is None:
        continue

      t += self.runSpeed
      if t < 1 and not self.manualRun:
        continue
      t = 0

      self.vehicle.tick()
      self.scorer.update()

      if self.autoGoal:
        position, _ = self.field.road.get(self.vehicle.distance, self.field)
        self.setGoal(position)

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

      if i % 100 == 0:
        print "{0} iterations completed.".format(i)


  def pause(self):
    print "Paused."
    key = raw_input("Enter a command "
                    "[(q)uit, set-(g)oal, (p)lots-toggle, (r)un-speed, (a)uto-goal-toggle]: ")

    if key == "q":
      sys.exit()

    elif key == "g":
      try:
        goal = int(raw_input("Enter new goal (non-number will be None): "))
      except ValueError:
        goal = None
      self.setGoal(goal)

    elif key == "p":
      self.plotsEnabled = not self.plotsEnabled

    elif key == "r":
      try:
        speed = float(raw_input("Enter new run speed (1.0 is highest): "))
        self.runSpeed = speed
      except ValueError:
        pass

    elif key == "a":
      self.autoGoal = not self.autoGoal
      if not self.autoGoal:
        self.setGoal(None)

    print "Resuming..."


  def setGoal(self, goal):
    self.goal = goal
    print "Set new goal: ", self.goal
