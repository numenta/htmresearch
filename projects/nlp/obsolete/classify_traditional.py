# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import math
import numpy
from collections import Counter, defaultdict

from htmresearch.frameworks.nlp.classification_model import ClassificationModel



class ClassificationModelTraditional(ClassificationModel):
  """
  Class to run the survey response classification task with TFIDF
  """

  def __init__(self, verbosity=1, numLabels=3):
    super(ClassificationModelTraditional, self).__init__(verbosity)

    # class -> {term -> count}
    self.tf = defaultdict(Counter)
    # class -> number of terms total
    self.counts = Counter()
    # term -> document count
    self.df = Counter()
    # number of documents total
    self.num_docs = 0
    # term -> log(num_docs / df)
    self.idf = defaultdict(float)
    # class -> {term -> tf * idf}
    self.tfidf = defaultdict(lambda : defaultdict(float))
    # class -> norm of tfidf
    self.norms = defaultdict(float)

    self.numLabels = numLabels


  def encodePattern(self, sample):
    """
    Encode an SDR of the input string into an empty array

    @param pattern     (list)           Tokenized sample, where each item is a
                                        string
    @return           (dictionary)      Dictionary, containing text, sparsity,
                                        and bitmap
    Example return dict:
    {
      "text": ["Example", "text"],
      "sparsity": 0.03,
      "bitmap": numpy.array()
    }
    """
    # Never use the bitmap so don't need it
    return {"text": sample, "sparsity": 0, "bitmap": numpy.zeros(0)}


  def resetModel(self):
    """Reset the model by clearing the classifier."""
    self.tf.clear()
    self.counts.clear()
    self.df.clear()
    self.num_docs = 0
    self.idf.clear()
    self.tfidf.clear()
    self.norms.clear()


  def trainModel(self, samples, labels):
    """
    Train the classifier on the input sample and label.

    @param samples     (dictionary)      Dictionary, containing text, sparsity,
    and bitmap
    @param labels      (int)             Reference index for the classification
                                         of this sample.
    """
    # Handle samples
    for sample, sample_labels in zip(samples, labels):
      self.num_docs += 1

      text = sample["text"]
      for token in set(text):
        # Only want a token to count once in a document
        self.df[token] += 1

      for label in sample_labels:
        for token in text:
          self.tf[label][token] += 1

        self.counts[label] += len(text)

    # Dependent on number of documents so need to reset every time
    # Best to use batch training
    # Convert to idf
    self.idf.clear()
    for token, count in self.df.iteritems():
      self.idf[token] = math.log(self.num_docs / float(count))

    # Convert to tfidf
    self.tfidf.clear()
    for name, tf in self.tf.iteritems():
      total = self.counts[name]
      for token, count in tf.iteritems():
        tf = count / float(total)
        self.tfidf[name][token] = tf * self.idf[token]

    # Update norm
    for name, tfidf in self.tfidf.iteritems():
      norm = 0.0
      for token, count in tfidf.iteritems():
        norm += count ** 2
      self.norms[name] = math.sqrt(norm)


  def testModel(self, sample):
    """
    Test the classifier on the input sample.  Returns the classification that
    is closest using cosine similarity

    @param sample     (dictionary)      Dictionary, containing text, sparsity,
                                        and bitmap
    @return           (list)            The label with the highest cosine
                                        similarity
    """

    text = sample["text"]

    tf = {}
    for token in text:
      tf[token] = tf.get(token, 0.0) + 1

    # Don't update self.idf because this is not training
    tfidf = {}
    for token, count in tf.iteritems():
      if token in self.idf:
        tfidf[token] = count / len(text) * self.idf[token]

    # Update norm
    norm = 0.0
    for token, count in tfidf.iteritems():
      norm += count ** 2
    norm = math.sqrt(norm)

    if norm == 0:
      return []

    # Cosine similarity
    matches = []
    for label, tf in self.tf.iteritems():
      distance = 0.0
      for token, c in tfidf.iteritems():
        if token in tf:
          distance += tfidf[token] * self.tfidf[label][token]

      # Normalize
      distance /= (norm * self.norms[label])

      matches.append((label, distance))

    matches = sorted(matches, key=lambda x:x[1], reverse=True)

    # Only return the labels, not the distances
    return zip(*matches[:self.numLabels])[0]
