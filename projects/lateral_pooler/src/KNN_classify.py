from nupic.algorithms.knn_classifier import KNNClassifier
 
 
# Create classifier to hold patterns
# Can specify k here. By default k=1
self.classifier = KNNClassifier(distanceMethod="rawOverlap")
 

# Learning. N is dimensionality of the underlying space (e.g. the
# number of columns in the SpatialPooler)
self.classifier.learn(sdr, category, isSparse=N)
 
 
# Inference (need to create a dense vector)
sdr = list(sdr)
sdr.sort()
dense = numpy.zeros(N)
dense[sdr] = 1.0
(winner, inferenceResult, dist, categoryDist) =  self.classifier.infer(dense)

if winner == correctCategory:
  numCorrectClassifications += 1.0