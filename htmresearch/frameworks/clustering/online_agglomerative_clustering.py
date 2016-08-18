#!/usr/bin/env python 

import heapq
import operator
import scipy
import numpy as np



def percent_overlap(x, y):
  """
  Distance metric for SDRs defined as: |x1 and x2|^2 / (|x1| . |x2|)
  :param x: binary vector
  :param y: binary vector
  :return: percent overlap (result between 0 and 1)
  """
  return np.linalg.norm(np.logical_and(x, y)) ** 2 / (np.linalg.norm(x)
                                                      * np.linalg.norm(y))



class Cluster(object):
  def __init__(self, a, distance_func):
    self.center = a
    self.size = 0
    self.distance_func = distance_func


  def add(self, e):
    self.size += self.distance_func(self.center, e)
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
  def __init__(self, N, distance_func):
    """
    N-1 is the largest number of clusters that can be found.
    Higher N makes clustering slower.
    """

    self.n = 0
    self.N = N

    self.distance_func = distance_func

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
    newc = Cluster(e, self.distance_func)
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



def generate_points():
  import random

  points = []
  # create three random 2D gaussian clusters
  for i in range(3):
    x = i
    y = i
    c = [scipy.array(
      (x + random.normalvariate(0, 0.1), y + random.normalvariate(0, 0.1))) for
         j in range(100)]
    points += c

  random.shuffle(points)
  return points



if __name__ == "__main__":

  import time
  from matplotlib import pyplot as plt

  plt.ion()  # interactive mode on

  # the value of N is generally quite forgiving, i.e.
  # giving 6 will still only find the 3 clusters.
  # around 10 it will start finding more
  N = 6

  points = generate_points()
  n = len(points)

  start = time.time()
  c = OnlineCluster(N, percent_overlap)
  last_cx = []
  last_cy = []
  while len(points) > 0:
    point = points.pop()
    plt.plot(point[0], point[1], 'bo')

    c.cluster(point)
    clusters = c.trimclusters()
    print "I clustered %d points in %.2f seconds and found %d clusters." % (
      n, time.time() - start, len(clusters))

    cx = [x.center[0] for x in clusters]
    cy = [y.center[1] for y in clusters]

    plt.plot(last_cx, last_cy, "bo")
    plt.plot(cx, cy, "ro")
    plt.pause(0.001)

    last_cx = cx
    last_cy = cy

  plt.plot(cx, cy, "go")
