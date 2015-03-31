class Logs(object):

  def __init__(self, field, vehicle, scorer, model):
    self.field = field
    self.vehicle = vehicle
    self.scorer = scorer
    self.model = model


  def log(self):
    pass
    # print self.scorer.score
