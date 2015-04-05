class Scorer(object):

  def __init__(self, field, vehicle):
    self.field = field
    self.vehicle = vehicle

    self.score = 0
    self.scoreDelta = 0


  def getChange(self):
    raise NotImplementedError


  def update(self):
    self.scoreDelta = self.getChange()
    self.score += self.scoreDelta



class StayOnRoadScorer(Scorer):

  def getChange(self):
    position, width = self.field.road.get(self.vehicle.distance, self.field)
    left = position - float(width / 2)
    right = left + width

    if self.vehicle.position >= left and self.vehicle.position <= right:
      return 1
    else:
      return -1
