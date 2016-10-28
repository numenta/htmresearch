class Cluster(object):
  def __init__(self, id, center):
    self.id = id
    self.center = center
    self.points = []
    self.size = 0


  def add(self, point, actual_label):
    self.points.append({'point': point, 'label': actual_label})
    self.size += 1



class PerfectClustering(object):
  def __init__(self, max_num_clusters, distance_func):
    self.clusters = [Cluster(i, None) for i in range(max_num_clusters)]
    self.distance = distance_func


  def cluster(self, point, actual_label):
    closest = self.clusters[int(actual_label)]
    closest.add(point, actual_label)
    return closest
