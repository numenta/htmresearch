import heapq
import operator
import scipy



class Cluster(object):
  def __init__(self, cluster_id, center, distance_func):
    self.id = cluster_id
    self.center = center
    self.size = 0
    self.distance_func = distance_func
    self.points = []


  def add(self, e, label):
    self.size += 1
    self.center += (e - self.center) / self.size
    self.points.append({'point': e, 'label': label})


  def merge(self, c):
    self.center = (self.center * self.size + c.center * c.size) / (
      self.size + c.size)
    self.size += c.size
    self.points.extend(c.points)


  def resize(self, dim):
    extra = scipy.zeros(dim - len(self.center))
    self.center = scipy.append(self.center, extra)


  def __str__(self):
    return "Cluster( %s, %s, %.2f )" % (self.id, self.center, self.size)



class Dist(object):
  """
  this is just a tuple,
  but we need an object so we can define cmp for heapq
  """


  def __init__(self, c1, c2, d):
    self.c1 = c1
    self.c2 = c2
    self.d = d


  def __cmp__(self, o):
    return cmp(self.d, o.d)


  def __str__(self):
    return "Dist(%f)" % (self.d)



class OnlineClustering(object):
  def __init__(self,
               max_num_clusters,
               distance_func,
               cluster_size_cutoff=0.1):
    """
    N-1 is the largest number of clusters that can be found.
    Higher N makes clustering slower.
    """

    self._num_points_processed = 0
    self._total_num_clusters_created = 0
    self._max_num_clusters = max_num_clusters

    self._distance_func = distance_func
    self._cluster_size_cutoff = cluster_size_cutoff

    self.clusters = []
    # max number of dimensions we've seen so far
    self._dim = 0

    # cache inter-cluster distances
    self._dist = []


  def _resize(self, dim):
    for c in self.clusters:
      c._resize(dim)
    self._dim = dim


  def find_closest_cluster(self, point, clusters):
    c = [(i, self._distance_func(c.center, point))
         for i, c in enumerate(clusters)]
    closest = clusters[min(c, key=operator.itemgetter(1))[0]]
    return closest


  def cluster(self, new_point, label, trim_clusters=False):

    if len(new_point) > self._dim:
      self._resize(len(new_point))

    if len(self.clusters) > 0:
      # compare new point to each existing cluster
      closest = self.find_closest_cluster(new_point, self.clusters)
      closest.add(new_point, label)
      # invalidate dist-cache for this cluster
      self._update_dist(closest)
    else:
      closest = None

    if len(self.clusters) >= self._max_num_clusters and len(
      self.clusters) > 1:
      # merge closest two clusters
      inter_cluster_dist = heapq.heappop(self._dist)
      cluster_to_merge = inter_cluster_dist.c2
      inter_cluster_dist.c1.merge(cluster_to_merge)
      if cluster_to_merge in self.clusters:
        self.clusters.remove(cluster_to_merge)

      # update inter-cluster distances      
      self._remove_dist(cluster_to_merge)
      self._update_dist(inter_cluster_dist.c1)

    # make a new cluster for this point
    cluster_id = self._total_num_clusters_created + 1
    new_cluster = Cluster(cluster_id, new_point, self._distance_func)
    self._total_num_clusters_created += 1
    self.clusters.append(new_cluster)
    self._update_dist(new_cluster)

    self._num_points_processed += 1

    if trim_clusters:
      trimmed_clusters = self._trim_clusters()
      # closest cluster might not be in the list of trimmed clusters
      self.find_closest_cluster(new_point, trimmed_clusters)
      return trimmed_clusters, closest
    else:
      return closest


  def _remove_dist(self, d):
    """Invalidate inter-cluster distance cache for c"""
    inter_cluster_dist_to_remove = []
    for inter_cluster_dist in self._dist:
      if inter_cluster_dist.c1 == d or inter_cluster_dist.c2 == d:
        inter_cluster_dist_to_remove.append(inter_cluster_dist)
    for x in inter_cluster_dist_to_remove:
      self._dist.remove(x)
    heapq.heapify(self._dist)


  def _update_dist(self, c):
    """Cluster c has changed, re-compute all inter-cluster distances"""
    self._remove_dist(c)

    for x in self.clusters:
      if x == c: continue
      d = self._distance_func(x.center, c.center)
      inter_cluster_dist = Dist(x, c, d)
      heapq.heappush(self._dist, inter_cluster_dist)


  def _trim_clusters(self):
    """Return only clusters over threshold"""
    mean_cluster_size = scipy.mean([x.size for x in self.clusters])
    t = mean_cluster_size * self._cluster_size_cutoff
    return filter(lambda x: x.size >= t, self.clusters)
