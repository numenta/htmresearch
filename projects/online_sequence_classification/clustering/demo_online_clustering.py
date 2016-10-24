import scipy

from htmresearch.frameworks.clustering.online_agglomerative_clustering import (
  OnlineCluster)

from htmresearch.frameworks.clustering.distances import kernel_dist
from htmresearch.frameworks.clustering.kernels import normalized_gaussian_kernel

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



def demo2D():

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
  kernel = normalized_gaussian_kernel
  kernel_distance = kernel_dist(kernel)
  c = OnlineCluster(N, kernel_distance, kernel)
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

if __name__ == "__main__":
  demo2D()