import numpy as np

from matplotlib import pyplot as plt
from sklearn import manifold
import matplotlib.cm as cm

from htmresearch.frameworks.clustering.distances import percentOverlap



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



def viz2DProjection(vizTitle, numClusters, clusterAssignments, npos):
  """
  Visualize SDR clusters with MDS
  :param npos: 2D projection of SDRs
  """
  colors = ['r', 'b', 'g', 'p']
  plt.figure()
  colorList = colors[:numClusters]
  colorNames = []
  for i in range(len(clusterAssignments)):
    clusterId = int(clusterAssignments[i])
    if clusterId not in colorNames:
      colorNames.append(clusterId)
    sdrProjection = npos[i]
    plt.scatter(sdrProjection[0], sdrProjection[1], label=clusterId,
                color=colorList[clusterId], marker='o', edgecolor='black')
    
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



def plotDistanceMat(distanceMat):
  plt.figure()
  plt.imshow(distanceMat)
  plt.show()
