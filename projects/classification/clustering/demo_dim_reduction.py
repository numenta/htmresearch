from htmresearch.frameworks.clustering.dim_reduction import (project2D,
                                                             assignClusters,
                                                             viz2DProjection,
                                                             plotDistanceMat)
from htmresearch.frameworks.clustering.utils import generateSDRs

def main():
  numClasses = 7
  numSDRsPerClass = 20
  noiseLevel = 0.1
  vizTitle = 'MDS, noise level: {}'.format(noiseLevel)
  
  # SDR parameters
  n = 1024
  w = 20
  
  sdrs = generateSDRs(numClasses, numSDRsPerClass, n, w, noiseLevel)
  
  npos, distanceMat = project2D(sdrs)
  
  clusterAssignments = assignClusters(sdrs, numClasses, numSDRsPerClass)
  
  viz2DProjection(vizTitle, numClasses, clusterAssignments, npos)
  
  plotDistanceMat(distanceMat)


if __name__ == '__main__':
  main()