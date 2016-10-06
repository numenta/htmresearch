import heapq
import operator
import scipy



class Cluster(object):
  def __init__(self, a, distance_func, kernel):
    self.center = a
    self.size = 0
    self.distance_func = distance_func
    self.kernel = kernel

  def add(self, e):
    if self.kernel:
      self.size += self.kernel(self.center, e)
    else:
      self.size += 1
    self.center += (e - self.center) / self.size


  def merge(self, c):
    self.center = (self.center * self.size + c.center * c.size) / (
      self.size + c.size)
    self.size += c.size


  def resize(self, dim):
    extra = scipy.zeros(dim - len(self.center))
    self.center = scipy.append(self.center, extra)


  def __str__(self):
    return "Cluster( %s, %f )" % (self.center, self.size)



class Dist(object):
  """
  this is just a tuple,
  but we need an object so we can define cmp for heapq
  """


  def __init__(self, x, y, d):
    self.x = x
    self.y = y
    self.d = d


  def __cmp__(self, o):
    return cmp(self.d, o.d)


  def __str__(self):
    return "Dist(%f)" % (self.d)



class OnlineCluster(object):
  def __init__(self, N, distance_func, kernel=None):
    """
    N-1 is the largest number of clusters that can be found.
    Higher N makes clustering slower.
    """

    self.n = 0
    self.N = N

    self.distance_func = distance_func
    self.kernel = kernel

    self.clusters = []
    # max number of dimensions we've seen so far
    self.dim = 0

    # cache inter-cluster distances
    self.dist = []


  def resize(self, dim):
    for c in self.clusters:
      c.resize(dim)
    self.dim = dim


  def cluster(self, e):

    if len(e) > self.dim:
      self.resize(len(e))

    if len(self.clusters) > 0:
      # compare new points to each existing cluster
      c = [(i, self.distance_func(x.center, e))
           for i, x in enumerate(self.clusters)]
      closest = self.clusters[min(c, key=operator.itemgetter(1))[0]]
      closest.add(e)
      # invalidate dist-cache for this cluster
      self.updatedist(closest)

    if len(self.clusters) >= self.N and len(self.clusters) > 1:
      # merge closest two clusters
      m = heapq.heappop(self.dist)
      m.x.merge(m.y)

      self.clusters.remove(m.y)
      self.removedist(m.y)

      self.updatedist(m.x)

    # make a new cluster for this point
    newc = Cluster(e, self.distance_func, self.kernel)
    self.clusters.append(newc)
    self.updatedist(newc)

    self.n += 1


  def removedist(self, c):
    """invalidate intercluster distance cache for c"""
    r = []
    for x in self.dist:
      if x.x == c or x.y == c:
        r.append(x)
    for x in r: self.dist.remove(x)
    heapq.heapify(self.dist)


  def updatedist(self, c):
    """Cluster c has changed, re-compute all intercluster distances"""
    self.removedist(c)

    for x in self.clusters:
      if x == c: continue
      d = self.distance_func(x.center, c.center)
      t = Dist(x, c, d)
      heapq.heappush(self.dist, t)


  def trimclusters(self):
    """Return only clusters over threshold"""
    t = scipy.mean([x.size for x in self.clusters]) * 0.1
    return filter(lambda x: x.size >= t, self.clusters)




