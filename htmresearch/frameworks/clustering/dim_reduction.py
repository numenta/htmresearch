import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt
from matplotlib import colors
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



def computeClusterDistanceMat(sdrClusters, numCells):
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
      distanceMat[i, j] = clusterDist(sdrClusters[i], sdrClusters[j], numCells)
      distanceMat[j, i] = distanceMat[i, j]

  return distanceMat



def viz2DProjection(vizTitle, outputFile, numClusters, clusterAssignments,
                    npos):
  """
  Visualize SDR clusters with MDS
  """

  colorList = colors.cnames.keys()
  plt.figure()
  colorList = colorList
  colorNames = []
  for i in range(len(clusterAssignments)):
    clusterId = int(clusterAssignments[i])
    if clusterId not in colorNames:
      colorNames.append(clusterId)
    sdrProjection = npos[i]
    label = 'Category %s' % clusterId
    if len(colorList) > clusterId:
      color = colorList[clusterId]
    else:
      color = 'black'
    plt.scatter(sdrProjection[0], sdrProjection[1], label=label, alpha=0.5,
                color=color, marker='o', edgecolor='black')

  # Add nicely formatted legend
  handles, labels = plt.gca().get_legend_handles_labels()
  by_label = OrderedDict(zip(labels, handles))
  plt.legend(by_label.values(), by_label.keys(), scatterpoints=1, loc=2)

  plt.title(vizTitle)
  plt.draw()
  plt.savefig(outputFile)



def assignClusters(sdrs, numClusters, numSDRsPerCluster):
  clusterAssignments = np.zeros(len(sdrs))

  clusterIDs = range(numClusters)
  for clusterID in clusterIDs:
    selectPts = np.arange(numSDRsPerCluster) + clusterID * numSDRsPerCluster
    clusterAssignments[selectPts] = clusterID

  return clusterAssignments



def project2D(sdrs, method='mds'):
  distance_mat = computeDistanceMat(sdrs)

  seed = np.random.RandomState(seed=3)

  if method == 'mds':
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9,
                       random_state=seed,
                       dissimilarity="precomputed", n_jobs=1)

    pos = mds.fit(distance_mat).embedding_

    nmds = manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
                        dissimilarity="precomputed", random_state=seed,
                        n_jobs=1, n_init=1)

    pos = nmds.fit_transform(distance_mat, init=pos)
  elif method == 'tSNE':
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    pos = tsne.fit_transform(distance_mat)
  else:
    raise NotImplementedError

  return pos, distance_mat



def projectClusters2D(sdrClusters, numCells):
  distanceMat = computeClusterDistanceMat(sdrClusters, numCells)

  seed = np.random.RandomState(seed=3)

  mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9,
                     random_state=seed,
                     dissimilarity="precomputed", n_jobs=1)

  pos = mds.fit(distanceMat).embedding_

  nmds = manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
                      dissimilarity="precomputed", random_state=seed, n_jobs=1,
                      n_init=1)

  npos = nmds.fit_transform(distanceMat, init=pos)

  return npos, distanceMat



def plotDistanceMat(distanceMat, title, outputFile, showPlot=False):
  plt.figure()
  plt.imshow(distanceMat, interpolation="nearest")
  plt.colorbar()
  plt.title(title)
  plt.savefig(outputFile)
  plt.draw()
  if showPlot:
    plt.show()
