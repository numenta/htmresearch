import numpy

from unity_client.fetcher import Fetcher
from sensorimotor.encoders.one_d_depth import OneDDepthEncoder



def run(plotEvery=1):
  encoder = OneDDepthEncoder(positions=[i*20 for i in range(36)],
                             radius=3,
                             wrapAround=True,
                             nPerPosition=57,
                             wPerPosition=3,
                             minVal=0,
                             maxVal=1)
  fetcher = Fetcher()
  plotter = Plotter(encoder)

  while True:
    outputData = fetcher.sync()

    if outputData is None:
      continue

    if fetcher.skippedTimesteps > 0:
      print ("Warning: skipped {0} timesteps, "
             "now at {1}").format(fetcher.skippedTimesteps, fetcher.timestep)

    if outputData["reset"]:
      print "Reset."

    sensor = outputData["ForwardsSweepSensor"]
    encoding = encoder.encode(numpy.array(sensor))

    plotter.update(sensor, encoding)

    if fetcher.timestep % plotEvery == 0:
      plotter.render()



class Plotter(object):

  def __init__(self, encoder):
    self.encoder = encoder

    self.sensor = []
    self.encoding = []

    import matplotlib.pyplot as plt
    self.plt = plt
    import matplotlib.cm as cm
    self.cm = cm

    self.plt.ion()
    self.plt.show()


  def update(self, sensor, encoding):
    self.sensor.append(sensor)
    self.encoding.append(encoding)


  def render(self):
    self.plt.figure(1)

    self.plt.clf()

    self.plt.subplot(3,1,1)
    self._imshow(self.sensor)

    self.plt.subplot(3,1,2)
    self._imshow(self.encoding)

    self.plt.subplot(3,1,3)
    shape = len(self.encoder.positions), self.encoder.scalarEncoder.getWidth()
    encoding = numpy.array(self.encoding[-1]).reshape(shape).transpose()
    self._imshow(encoding)

    self.plt.draw()


  def _imshow(self, data):
    self.plt.imshow(data,
                    cmap=self.cm.Greys,
                    interpolation="nearest",
                    aspect='auto',
                    vmin=0,
                    vmax=1)



run()
