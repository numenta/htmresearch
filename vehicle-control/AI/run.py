from unity_client.fetcher import Fetcher


def run(plotEvery=1):
  fetcher = Fetcher()
  plotter = Plotter()

  while True:
    outputData = fetcher.sync()

    if outputData is None:
      continue

    if fetcher.skippedTimesteps > 0:
      print ("Warning: skipped {0} timesteps, "
             "now at {1}").format(fetcher.skippedTimesteps, fetcher.timestep)

    if outputData["reset"]:
      print "Reset."

    plotter.update(outputData["ForwardsSweepSensor"])

    if fetcher.timestep % plotEvery == 0:
      plotter.render()



class Plotter(object):

  def __init__(self):
    self.sensor = []

    import matplotlib.pyplot as plt
    self.plt = plt
    import matplotlib.cm as cm
    self.cm = cm

    self.plt.ion()
    self.plt.show()


  def update(self, sensor):
    self.sensor.append(sensor)


  def render(self):
    self.plt.figure(1)

    self.plt.clf()

    self._imshow(self.sensor)

    self.plt.draw()


  def _imshow(self, data):
    self.plt.imshow(data,
                    cmap=self.cm.Greys,
                    interpolation="nearest",
                    aspect='auto',
                    vmin=0,
                    vmax=1)



run()
