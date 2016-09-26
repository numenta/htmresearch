import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt
from sklearn import manifold

from htmresearch.frameworks.clustering.distances import (
  percentOverlap, clusterDist)



def convertNonZeroToSDR(patternNZs, numCells):
  sdrs = []
  for patternNZ in patternNZs:
    sdr = np.zeros(numCells)
    sdr[patternNZ] = 1
    sdrs.append(sdr)

  return sdrs



def computeDistanceMat(sdrs):
  """
  Compute distance matrix between SDRs
  :param sdrs: (array of arrays) array of SDRs
  :return: distance matrix
  """
  numSDRs = len(sdrs)
  # calculate pairwise distance
  distanceMat = np.zeros((numSDRs, numSDRs), dtype=np.float64)
  for i in range(numSDRs):
    for j in range(numSDRs):
      distanceMat[i, j] = 1 - percentOverlap(sdrs[i], sdrs[j])
  return distanceMat



def computeClusterDistanceMat(sdrClusters):
  """
  Compute distance matrix between clusters of SDRs
  :param sdrClusters: list of sdr clusters,
                      each cluster is a list of SDRs
                      each SDR is a list of active indices
  :return: distance matrix
  """
  numClusters = len(sdrClusters)
  distanceMat = np.zeros((numClusters, numClusters), dtype=np.float64)
  for i in range(numClusters):
    for j in range(i, numClusters):
      distanceMat[i, j] = clusterDist(sdrClusters[i], sdrClusters[j])
      distanceMat[j, i] = distanceMat[i, j]

  return distanceMat



def viz2DProjection(vizTitle, numClusters, clusterAssignments, npos):
  """
  Visualize SDR clusters with MDS
  :param npos: 2D projection of SDRs
  """
  colors = ['g', 'b', 'r', 'p']
  plt.figure()
  colorList = colors[:numClusters]
  colorNames = []
  for i in range(len(clusterAssignments)):
    clusterId = int(clusterAssignments[i])
    if clusterId not in colorNames:
      colorNames.append(clusterId)
    sdrProjection = npos[i]
    label = 'Category %s' % clusterId
    plt.scatter(sdrProjection[0], sdrProjection[1], label=label, alpha=0.5,
                color=colorList[clusterId], marker='o', edgecolor='black')

  # Add nicely formatted legend
  handles, labels = plt.gca().get_legend_handles_labels()
  by_label = OrderedDict(zip(labels, handles))
  plt.legend(by_label.values(), by_label.keys(), scatterpoints=1, loc=2)

  plt.title(vizTitle)
  plt.show()



def assignClusters(sdrs, numClusters, numSDRsPerCluster):
  clusterAssignments = np.zeros(len(sdrs))

  clusterIDs = range(numClusters)
  for clusterID in clusterIDs:
    selectPts = np.arange(numSDRsPerCluster) + clusterID * numSDRsPerCluster
    clusterAssignments[selectPts] = clusterID

  return clusterAssignments



def project2D(sdrs):
  distanceMat = computeDistanceMat(sdrs)

  seed = np.random.RandomState(seed=3)

  mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                     dissimilarity="precomputed", n_jobs=1)
  pos = mds.fit(distanceMat).embedding_

  nmds = manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
                      dissimilarity="precomputed", random_state=seed, n_jobs=1,
                      n_init=1)

  npos = nmds.fit_transform(distanceMat, init=pos)

  return npos, distanceMat



def projectClusters2D(sdrClusters):
  distanceMat = computeClusterDistanceMat(sdrClusters)

  seed = np.random.RandomState(seed=3)

  mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                     dissimilarity="precomputed", n_jobs=1)

  pos = mds.fit(distanceMat).embedding_

  nmds = manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
                      dissimilarity="precomputed", random_state=seed, n_jobs=1,
                      n_init=1)

  npos = nmds.fit_transform(distanceMat, init=pos)

  return npos, distanceMat



def plotDistanceMat(distanceMat):
  plt.figure()
  plt.imshow(distanceMat)
  plt.show()
