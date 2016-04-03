#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import os
import argparse
import random
import cPickle as pickle
import time

from prettytable import PrettyTable
from cortipy.cortical_client import CorticalClient


def countWordOverlaps(client, words,
                      wordBitmapFilename = None):

  # Step 1:
  # Get bitmaps of all words with > 40 bits
  #
  if wordBitmapFilename is None:
    bitmaps = {}
    numBits = 0
    maxBits = 0
    minBits = 100000
    numWords = 0
    for word in words:
      b = set(client.getBitmap(word)["fingerprint"]["positions"])
      bits = len(b)

      # We only want to consider words with enough bits
      if bits > 40:
        bitmaps[word] = b
        numBits += bits
        maxBits = max(maxBits, bits)
        minBits = min(minBits, bits)
        numWords += 1

    print "average numBits=",float(numBits)/numWords
    print "max=",maxBits,"min=",minBits
    print numWords,"have > 40 bits out of",len(words),"total words"

    # Write out all the word bitmaps
    with open("word_bitmaps_40_bits_minimum.pkl","wb") as f:
      pickle.dump(bitmaps, f)

  else:
    with open(wordBitmapFilename,"rb") as f:
      bitmaps = pickle.load(f)


  # Step 2:
  # Find pair-wise overlap statistics
  #
  printTemplate = PrettyTable(["Overlap", "Word1", "Word2"],
                              sortby="Overlap", reversesort=True)
  printTemplate.align = "l"
  printTemplate.header_style = "upper"

  # 68,304 words in the goodWords list with > 20 bits
  # leading to 2,332,684,056 total comparisons
  # 55,753 have > 40 bits
  maxOverlap = 0
  bestWords = ""
  goodWords = bitmaps.keys()
  goodWords.sort()
  overlapSum = long(0)
  numOverlaps = 0
  goodOverlapPairs = []
  startTime = time.time()
  # Compute all pair-wise overlaps. This is very slow. In retrospect I
  # should have just used scipy.sparse matrices.
  for i, w1 in enumerate(goodWords):
    for w2 in goodWords[i+1:]:
      overlap = len(bitmaps[w1] & bitmaps[w2])
      if overlap > maxOverlap:
        maxOverlap = overlap
        bestWords = w1 + ":" + w2

      overlapSum += overlap
      numOverlaps += 1
      if overlap >= 50:
        goodOverlapPairs.append((w1,w2,overlap))
        printTemplate.add_row([overlap, w1, w2])

  print "Done. Elapsed time for pairwise comparisons =",time.time()-startTime

  # Save the list for later analysis
  with open("goodOverlapPairs.pkl","wb") as f:
    pickle.dump(goodOverlapPairs, f)

  print printTemplate
  print
  print "Max overlap=",maxOverlap, bestWords
  print "Num overlaps=",numOverlaps
  print "Average overlap=",overlapSum / float(numOverlaps)
  print "Number of words with >= 50 bits overlap=",len(goodOverlapPairs)



def countWordOverlapFrequencies(filename="goodOverlapPairs.pkl"):
  """
  Count how many high overlaps each word has, and print it out
  """
  with open(filename,"rb") as f:
    goodOverlapPairs = pickle.load(f)

  with open("word_bitmaps_40_bits_minimum.pkl","rb") as f:
    bitmaps = pickle.load(f)

  # Count how often each word has a highly overlapping match with other words
  wordFrequencies = {}
  for w1, w2, overlap in goodOverlapPairs:
    wordFrequencies[w1] = wordFrequencies.get(w1, 0) + 1


  printTemplate = PrettyTable(["Num High Overlaps", "Word", "On Bits"],
                              sortby="Num High Overlaps", reversesort=True)

  for word in wordFrequencies.iterkeys():
    printTemplate.add_row([wordFrequencies[word], word, len(bitmaps[word])])

  print printTemplate

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--retinaId",
                      default="en_synonymous",
                      type=str)
  parser.add_argument("--corticalApiKey",
                      default=os.environ.get("CORTICAL_API_KEY"),
                      type=str)
  parser.add_argument("--cacheDir",
                      default="../../htmresearch/encoders/CioCache", type=str)

  opts = parser.parse_args()

  client = CorticalClient(opts.corticalApiKey,
                          retina=opts.retinaId,
                          verbosity=0,
                          cacheDir=opts.cacheDir,
                          fillSDR=None)


  # Read in words from dictionary
  with open("enable1.txt", "r") as f:
    lines = f.readlines()

  print "Processing",len(lines),"lines..."
  words = []
  random.seed(42)
  # Subsample small percentage of words
  for line in lines:
      p = random.uniform(0,1)
      if p <= 1.05:
        # print line
        words.append(line.strip())
  words.sort()

  print "Done, got",len(words),"words"

  countWordOverlaps(client, words)

  countWordOverlapFrequencies()