
import datetime
import os
import sys
import argparse
import cPickle as pickle
import random

from nupic.bindings.math import SparseMatrix
from cortipy.cortical_client import CorticalClient, RETINA_SIZES
from htmresearch.support.text_preprocess import TextPreprocess

import pylab
import plotly.plotly as py
import plotly.graph_objs as go

def printFrequencyStatistics(counts, frequencies, numWords, size):
  """
  Print interesting statistics regarding the counts and frequency matrices
  """
  avgBits = float(counts.sum())/numWords
  print "Retina width=128, height=128"
  print "Total number of words processed=",numWords
  print "Average number of bits per word=",avgBits,
  print "avg sparsity=",avgBits/size
  print "counts matrix sum=",counts.sum(),
  print "max=",counts.max(), "min=",counts.min(),
  print "mean=",counts.sum()/float(size)
  print "frequency matrix sum=",frequencies.sum(),
  print "max=",frequencies.max(), "min=",frequencies.min(),
  print "mean=",frequencies.sum()/float(size)
  print "Number of bits with zero entries",frequencies.nZeroCols()


def countRandomBitFrequencies(numTerms = 100000, percentSparsity = 0.03):
  """Create a uniformly random counts matrix through sampling."""
  # Accumulate counts by inplace-adding sparse matrices
  counts = SparseMatrix()
  size = 128*128
  counts.resize(1, size)

  # Pre-allocate buffer sparse matrix
  sparseBitmap = SparseMatrix()
  sparseBitmap.resize(1, size)

  random.seed(42)

  # Accumulate counts for each bit for each word
  numWords=0
  for term in xrange(numTerms):

    bitmap = random.sample(xrange(size), int(size*percentSparsity))
    bitmap.sort()

    sparseBitmap.setRowFromSparse(0, bitmap, [1]*len(bitmap))
    counts += sparseBitmap
    numWords += 1

  # Compute normalized version of counts as a separate matrix
  frequencies = SparseMatrix()
  frequencies.resize(1, size)
  frequencies.copy(counts)
  frequencies.divide(float(numWords))

  # Wrap up by printing some statistics and then saving the normalized version
  printFrequencyStatistics(counts, frequencies, numWords, size)

  frequencyFilename = "bit_frequencies_random.pkl"
  print "Saving frequency matrix in",frequencyFilename
  with open(frequencyFilename, "wb") as frequencyPickleFile:
    pickle.dump(frequencies, frequencyPickleFile)

  return counts


def countBitFrequenciesForTerms(client, lines):
  # Accumulate counts by inplace-adding sparse matrices
  counts = SparseMatrix()
  width = RETINA_SIZES[client.retina]["width"]
  height = RETINA_SIZES[client.retina]["height"]
  counts.resize(1, width*height)

  # Pre-allocate buffer sparse matrix
  sparseBitmap = SparseMatrix()
  sparseBitmap.resize(1, width*height)

  # Accumulate counts for each bit for each word
  numWords=0
  for line in lines:
    tokens = TextPreprocess().tokenize(line)
    for term in tokens:

      try:
        bitmap = client.getBitmap(term)["fingerprint"]["positions"]
      except Exception as err:
        print "Skipping '{}', reason: {}".format(term, str(err))
        continue

      if not bitmap:
        print "Skipping '{}', reason: empty".format(term)
        continue

      sparseBitmap.setRowFromSparse(0, bitmap, [1]*len(bitmap))
      counts += sparseBitmap
      numWords += 1

  # Compute normalized version of counts as a separate matrix
  frequencies = SparseMatrix()
  frequencies.resize(1, width*height)
  frequencies.copy(counts)
  frequencies.divide(float(numWords))

  # Wrap up by printing some statistics and then saving the normalized version
  printFrequencyStatistics(counts, frequencies, numWords, width*height)

  frequencyFilename = "bit_frequencies_"+client.retina+".pkl"
  print "Saving frequency matrix in",frequencyFilename
  with open(frequencyFilename, "wb") as frequencyPickleFile:
    pickle.dump(frequencies, frequencyPickleFile)

  return counts


def plotlyFrequencyHistogram(counts):
  """
  x-axis is a count of how many times a bit was active
  y-axis is number of bits that have that frequency
  """
  data = [
    go.Histogram(
      x=tuple(count for _, _, count in counts.getNonZerosSorted())
    )
  ]
  py.plot(data, filename=os.environ.get("HEATMAP_NAME",
                                        str(datetime.datetime.now())))


def plotHistogram(counts):
  histogram = [(x[1], x[2]) for x in counts.getNonZerosSorted()]

  histogram.sort()
  x = tuple(position for position, _ in histogram)
  y = tuple(count for _, count in histogram)
  pylab.bar(x, y)
  pylab.show()


def plotlyHeatmap(counts):
  counts.reshape(128, 128)
  data = [go.Heatmap(z=counts.toDense())]
  py.plot(data, filename=os.environ.get("HEATMAP_NAME",
                                        str(datetime.datetime.now())))

def main(terms):
  parser = argparse.ArgumentParser()
  parser.add_argument("--retinaId",
                      default="en_associative",
                      type=str)
  parser.add_argument("--corticalApiKey",
                      default=os.environ.get("CORTICAL_API_KEY"),
                      type=str)
  parser.add_argument("--plot",
                      default="histogram",
                      choices=["histogram", "heatmap", "frequencyHistogram",
                               "None", "none"])
  parser.add_argument("--cacheDir", default=None, type=str)

  opts = parser.parse_args()

  if opts.retinaId == "random":
    counts = countRandomBitFrequencies()
  else:
    client = CorticalClient(opts.corticalApiKey,
                            retina=opts.retinaId,
                            verbosity=0,
                            cacheDir=opts.cacheDir,
                            fillSDR=None)

    counts = countBitFrequenciesForTerms(client, terms)

  if opts.plot == "histogram":
    plotHistogram(counts)
  elif opts.plot == "heatmap":
    plotlyHeatmap(counts)
  elif opts.plot == "frequencyHistogram":
    plotlyFrequencyHistogram(counts)


if __name__ == "__main__":
    main(sys.stdin)
