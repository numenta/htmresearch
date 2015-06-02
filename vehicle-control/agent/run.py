import numpy

from unity_client.fetcher import Fetcher
from sensorimotor.encoders.one_d_depth import OneDDepthEncoder



def run(positions, plotEvery=1):
  encoder = OneDDepthEncoder(positions=positions,
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

    from pylab import rcParams
    rcParams.update({'figure.figsize': (6, 9)})
    # rcParams.update({'figure.autolayout': True})
    rcParams.update({'figure.facecolor': 'white'})

    self.plt.ion()
    self.plt.show()


  def update(self, sensor, encoding):
    self.sensor.append(sensor)
    self.encoding.append(encoding)


  def render(self):
    self.plt.figure(1)

    self.plt.clf()

    self.plt.subplot(4,1,1)
    self._imshow(self.sensor, "Sensor over time")

    self.plt.subplot(4,1,2)
    self._imshow(self.encoding, "Encoding over time")

    self.plt.subplot(4,1,3)
    shape = len(self.encoder.positions), self.encoder.scalarEncoder.getWidth()
    encoding = numpy.array(self.encoding[-1]).reshape(shape).transpose()
    self._imshow(encoding, "Encoding at time t")

    self.plt.subplot(4,1,4)
    data = self.encoding
    w = self.encoder.w
    overlaps = [sum(a & b) / float(w) for a, b in zip(data[:-1], data[1:])]
    self._plot(overlaps, "Encoding overlaps between consecutive times")

    self.plt.draw()


  def _plot(self, data, title):
    self.plt.title(title)
    self.plt.xlim(0, len(data))
    self.plt.plot(range(len(data)), data)


  def _imshow(self, data, title):
    self.plt.title(title)
    self.plt.imshow(data,
                    cmap=self.cm.Greys,
                    interpolation="nearest",
                    aspect='auto',
                    vmin=0,
                    vmax=1)



# complete uniform
# positions = [i*20 for i in range(36)]

# forward uniform
positions = [i*10 for i in range(18, 54)]

run(positions)
