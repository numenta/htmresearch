import numpy



class Road(object):

  def get(self, distance, field):
    raise NotImplementedError



class StraightRoad(Road):

  def get(self, distance, field):
    return (float(field.width) / 2, 25)



class ZigZagRoad(Road):

  def __init__(self, width=25, zigZagEvery=60):
    super(ZigZagRoad, self).__init__()

    self.width = width
    self.zigZagEvery = zigZagEvery


  def get(self, distance, field):
    zig = int(distance / self.zigZagEvery)
    zag = int((distance + self.zigZagEvery) / self.zigZagEvery)

    zigDistance = distance % self.zigZagEvery

    numpy.random.seed(zig)
    zigPosition = numpy.random.randint(0, field.width)
    numpy.random.seed(zag)
    zagPosition = numpy.random.randint(0, field.width)
    separation = zagPosition - zigPosition

    zigPercent = float(zigDistance) / self.zigZagEvery
    position = zigPosition + (separation * zigPercent)

    return (position, self.width)
