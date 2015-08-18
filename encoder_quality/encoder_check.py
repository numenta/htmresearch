# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
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

import numpy as np



class _InputTripleCreator(object):
  def getNewTriple(self):
    """Returns a triple of inputs."""
    raise "Unimplemented."



class InputTripleCreator(_InputTripleCreator):
  def __init__(self, sampleGenerator):
    self.sampleGenerator = sampleGenerator


  def getNewTriple(self):
    first = self.sampleGenerator()
    second = self.sampleGenerator()
    triple = self.sampleGenerator()

    return (first, second, triple)



class InputTripleCreatorFromList(_InputTripleCreator):
  def __init__(self, sampleList):
    self.sampleList = sampleList


  def getNewTriple(self):
    first = np.random.choice(self.sampleList)
    second = np.random.choice(self.sampleList)
    third = np.random.choice(self.sampleList)

    return (first, second, third)



class InputTriangle(object):
  def __init__(self, input_triple, distance_function):
    self.input_triple = input_triple
    self.distance_function = distance_function
    self.edges = {}
    self.sdrs = {}


  def _computeEdges(self):
    pairs = [(0, 1), (1, 2), (2, 0)]

    for i, j in pairs:
      self.edges[(i, j)] = self.distance_function(self.input_triple[i],
                                                  self.input_triple[j])


  def _computeSDRs(self, encoder):
    pairs = [(0, 1), (1, 2), (2, 0)]

    for i, j in pairs:
      self.sdrs[(i, j)] = sdrSimilarity(encoder.encode(self.input_triple[i]),
                                    encoder.encode(self.input_triple[j]))


  def checkTriangle(self, encoder):
    self._computeEdges()
    self._computeSDRs(encoder)

    edges = self.edges.keys()

    for edge1 in edges:
      for edge2 in edges:
        input_comp = self.edges[edge1] > self.edges[edge2]
        sdr_comp = self.sdrs[edge1] < self.sdrs[edge2]

        if sdr_comp and not input_comp:
          print "Warning: ",
          print "The distance between",
          print self.input_triple[edge1[0]],
          print "and",
          print self.input_triple[edge1[1]],
          print "is",
          print self.edges[edge1],
          print "with SDR overlap",
          print self.sdrs[edge1],
          print ", and the distance between",
          print self.input_triple[edge2[0]],
          print "and",
          print self.input_triple[edge2[1]],
          print "is",
          print self.edges[edge2],
          print "with SDR overlap",
          print self.sdrs[edge2],
          print "."
          print

          return True

    return False



def sdrSimilarity(first_sdr, second_sdr):
  """Percent similarity between the two SDRs."""
  overlap = np.sum(np.logical_and(first_sdr, second_sdr))
  return overlap


def encoderCheck(encoder, distance_function, input_pairs_source, trials=1000,
                 verbosity=0):
  """Find potentially problematic encoding instances in the input space.

  This checker uses the following logic: if the distance between A and B is
  less than the distance between C and D, then the overlap between the SDRs of
  A and B should not be less than the distance between those of C and D.

  @param encoder An Encoder object that has an encode method for converting
            inputs into SDRs.

  @param distance_function A function that takes two inputs and returns a
          scalar between 0 and 1 capturing semantic similarity.

  @param input_pairs_source A _InputPairCreator object.

  @param trials The number of pairs compared to estimate encoder error.

  @param verbosity How much chatter during running.
  """
  next_input = input_pairs_source.getNewTriple()
  counter = 0
  total_problems = 0.0

  while next_input and counter < trials:

    triangle = InputTriangle(next_input, distance_function)

    if verbosity:
      print "Samples: ", next_input

    if triangle.checkTriangle(encoder):
      total_problems += 1

    next_input = input_pairs_source.getNewTriple()
    counter += 1

    if verbosity:
      print

  print "Ran "+str(counter)+" samples."

  avg_error = total_problems/counter

  return avg_error
