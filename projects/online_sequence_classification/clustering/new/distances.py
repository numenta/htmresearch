import numpy as np



def euclidian(x1, x2):
  return np.linalg.norm(np.array(x1) - np.array(x2))



def cluster_distance_factory(distance):
  def cluster_distance(c1, c2):
    """
    Symmetric distance between two clusters
  
    :param c1: (np.array) cluster 1
    :param c2: (np.array) cluster 2
    :return: distance between 2 clusters
    """
    cluster_dist = cluster_distance_directed_factory(distance)
    d12 = cluster_dist(c1, c2)
    d21 = cluster_dist(c2, c1)
    return np.mean([d12, d21])


  return cluster_distance



def cluster_distance_directed_factory(distance):
  def cluster_distance_directed(c1, c2):
    """
    Directed distance from cluster 1 to cluster 2
  
    :param c1: (np.array) cluster 1
    :param c2: (np.array) cluster 2
    :return: distance between 2 clusters
    """
    if len(c1) == 0 or len(c2) == 0:
      return 0
    else:
      return distance(np.sum(c1, axis=0) / float(len(c1)),
                      np.sum(c2, axis=0) / float(len(c2)))


  return cluster_distance_directed
