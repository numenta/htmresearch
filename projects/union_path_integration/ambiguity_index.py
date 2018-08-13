# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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

"""
A metric for the ambiguity of each sensory input.
"""

import itertools
import math
import numbers

import numpy as np


def choose(n, k):
  """
  Computes "n choose k". This can handle higher values than
  scipy.special.binom().
  """
  if isinstance(k, numbers.Number):
    return _choose(n, k)
  else:
    return np.array([_choose(n, k2) for k2 in k])


def _choose(n, k):
  k = np.minimum(k, n - k)
  return (np.product(np.arange(n-k+1, n+1, dtype="float128")) /
          np.product(np.arange(1, k+1, dtype="float128")))


class BinomialDistribution(object):
  """
  Given a coin with P(Heads=p), flip it n times.
  What's the probability of getting k heads?
  """

  def __init__(self, n, p, cache=False):
    self.n = n
    self.p = p
    self.possibleValues = xrange(0, n+1)
    self.cache = cache
    if self.cache:
      self._cache()

  def _cache(self):
    self._cachedPmf = self._pmf(np.arange(self.n + 1))
    self._cachedCdf = np.zeros(self.n + 1)
    for k in xrange(self.n + 1):
      self._cachedCdf[k:] += self._cachedPmf[k]

  def _pmf(self, k):
    return (choose(self.n, k)*
            np.power(self.p, k)*
            np.power(1. - self.p, self.n - k))

  def pmf(self, k):
    if self.cache:
      withinBounds = (k >= 0) & (k <= self.n)
      k2 = np.where(withinBounds, k, 0)
      return np.where(withinBounds, self._cachedPmf(k2))
    else:
      return self._pmf(k)

  def _cdf(self, k):
    return np.sum(self._pmf(np.arange(k+1)))

  def cdf(self, k):
    if self.cache:
      notNegative = k >= 0
      notGtN = k <= self.n
      withinBounds = notNegative & notGtN
      k2 = np.where(withinBounds, k, 0)
      return np.where(notNegative, np.where(notGtN, self._cachedCdf[k2],
                                            1.),
                      0.)
    else:
      if isinstance(k, numbers.Number):
        return self._cdf(k)
      else:
        return np.array([self._cdf(k2) for k2 in k])


class SampleMinimumDistribution(object):
  """
  Sample a random variable.
  What is the probability that the lowest sample is k?
  """
  def __init__(self, numSamples, distribution):
    self.numSamples = numSamples
    self.distribution = distribution
    self.possibleValues = distribution.possibleValues

  def pmf(self, k):
    return self.cdf(k) - self.cdf(k-1)

  def cdf(self, k):
    return 1 - np.power(1 - self.distribution.cdf(k), self.numSamples)


def getExpectedValue(distribution):
  """
  Calculates E[X] where X is a distribution.
  """
  k = np.array(distribution.possibleValues)
  return np.sum(k * distribution.pmf(k))


def findBinomialNsWithExpectedSampleMinimum(desiredValuesSorted, p, numSamples, nMax):
  """
  For each desired value, find an approximate n for which the sample minimum
  has a expected value equal to this value.

  For each value, find an adjacent pair of n values whose expected sample minima
  are below and above the desired value, respectively, and return a
  linearly-interpolated n between these two values.

  @param p (float)
  The p if the binomial distribution.

  @param numSamples (int)
  The number of samples in the sample minimum distribution.

  @return
  A list of results. Each result contains
    (interpolated_n, lower_value, upper_value).
  where each lower_value and upper_value are the expected sample minimum for
  floor(interpolated_n) and ceil(interpolated_n)
  """

  # mapping from n -> expected value
  actualValues = [
    getExpectedValue(
      SampleMinimumDistribution(numSamples,
                                BinomialDistribution(n, p, cache=True)))
    for n in xrange(nMax + 1)]

  results = []

  n = 0

  for desiredValue in desiredValuesSorted:
    while n + 1 <= nMax and actualValues[n + 1] < desiredValue:
      n += 1

    if n + 1 > nMax:
      break

    interpolated = n + ((desiredValue - actualValues[n]) /
                        (actualValues[n+1] - actualValues[n]))
    result = (interpolated, actualValues[n], actualValues[n + 1])
    results.append(result)

  return results


def findBinomialNsWithLowerBoundSampleMinimum(confidence, desiredValuesSorted,
                                              p, numSamples, nMax):
  """
  For each desired value, find an approximate n for which the sample minimum
  has a probabilistic lower bound equal to this value.

  For each value, find an adjacent pair of n values whose lower bound sample
  minima are below and above the desired value, respectively, and return a
  linearly-interpolated n between these two values.

  @param confidence (float)
  For the probabilistic lower bound, this specifies the probability. If this is
  0.8, that means that there's an 80% chance that the sample minimum is >= the
  desired value, and 20% chance that it's < the desired value.

  @param p (float)
  The p if the binomial distribution.

  @param numSamples (int)
  The number of samples in the sample minimum distribution.

  @return
  A list of results. Each result contains
    (interpolated_n, lower_value, upper_value).
  where each lower_value and upper_value are the probabilistic lower bound
  sample minimum for floor(interpolated_n) and ceil(interpolated_n)
  respectively.
   ...]
  """

  def P(n, numOccurrences):
    """
    Given n, return probability than the sample minimum is >= numOccurrences
    """
    return 1 - SampleMinimumDistribution(numSamples, BinomialDistribution(n, p)).cdf(
      numOccurrences - 1)

  results = []

  n = 0

  for desiredValue in desiredValuesSorted:
    while n + 1 <= nMax and P(n + 1, desiredValue) < confidence:
      n += 1

    if n + 1 > nMax:
      break

    left = P(n, desiredValue)
    right = P(n + 1, desiredValue)

    interpolated = n + ((confidence - left) /
                        (right - left))
    result = (interpolated, left, right)
    results.append(result)

  return results


def generateExpectedList(numUniqueFeatures, numLocationsPerObject, maxNumObjects):
  """
  Metric: How unique is each object's most unique feature? Calculate the
  expected number of occurrences of an object's most unique feature.
  """
  # We're choosing a location, checking its feature, and checking how many
  # *other* occurrences there are of this feature. So we check n - 1 locations.
  maxNumOtherLocations = maxNumObjects*10 - 1

  results = zip(itertools.count(1),
                findBinomialNsWithExpectedSampleMinimum(
                  itertools.count(1), 1./numUniqueFeatures, numLocationsPerObject,
                  maxNumOtherLocations))

  finalResults = [(numOtherLocations, interpolatedN / numLocationsPerObject)
                  for numOtherLocations, (interpolatedN, _, _) in results]

  return [(0, 0.)] + finalResults


def generateLowerBoundList(confidence, numUniqueFeatures, numLocationsPerObject,
                           maxNumObjects):
  """
  Metric: How unique is each object's most unique feature? Calculate the
  probabilistic lower bound for the number of occurrences of an object's most
  unique feature. For example, if confidence is 0.8, the tick "3" will be placed
  at the point where 80% of objects are completely composed of features with 3
  or more total occurrences, and 20% of objects have at least one feature that
  has 2 or fewer total occurrences.
  """
  # We're choosing a location, checking its feature, and checking how many
  # *other* occurrences there are of this feature. So we check n - 1 locations.
  maxNumOtherLocations = maxNumObjects*10 - 1

  results = zip(itertools.count(1),
                findBinomialNsWithLowerBoundSampleMinimum(
                  confidence,
                  itertools.count(1), 1./numUniqueFeatures, numLocationsPerObject,
                  maxNumOtherLocations))

  finalResults = [(numOtherLocations, interpolatedN / numLocationsPerObject)
                  for numOtherLocations, (interpolatedN, _, _) in results]

  return finalResults


def getTotalExpectedOccurrencesTicks_2_5(ticks):
  """
  Extract a set of tick locations and labels. The input ticks are assumed to
  mean "How many *other* occurrences are there of the sensed feature?" but we
  want to show how many *total* occurrences there are. So we add 1.

  We label tick 2, and then 5, 10, 15, 20, ...

  @param ticks
  A list of ticks, typically calculated by one of the above generate*List functions.
  """
  locs = [loc
          for label, loc in ticks]
  labels = [(str(label + 1) if
             (label + 1 == 2
             or (label+1) % 5 == 0)
             else "")
            for label, loc in ticks]
  return locs, labels



# Result of generateExpectedList(100, 10, 175)
numOtherOccurrencesOfMostUniqueFeature_expected_100features_10locationsPerObject = [(0, 0.), (1, 35.045167030044915), (2, 51.216815324993732), (3, 65.992011877006846), (4, 80.035086889843299), (5, 93.606591830738111), (6, 106.84133218518113), (7, 119.81991111184459), (8, 132.59509350322159), (9, 145.20366899818202), (10, 157.67250545353863), (11, 170.02193806774264)]

# Result of generateLowerBoundList(100, 10, 800)
numOtherOccurrencesOfMostUniqueFeature_lowerBound80_100features_10locationsPerObject = [(1, 37.945711986980281298), (2, 56.949050730340841475), (3, 73.613153481567207309), (4, 89.151770000095628528), (5, 103.99430588840699859), (6, 118.35178815706850657), (7, 132.34606437289453924), (8, 146.05527446633717879), (9, 159.53297434899399285), (10, 172.81778839309870084), (11, 185.93870238076113069), (12, 198.91800794416756371), (13, 211.77345422740526286), (14, 224.5192892475029286), (15, 237.16727926124247836), (16, 249.72717145985129587), (17, 262.20721393780454658), (18, 274.61448911011742019), (19, 286.95503748208463807), (20, 299.23409113244846178), (21, 311.45628337016557993), (22, 323.6256546907131344), (23, 335.74582776301934292), (24, 347.81997350922715531), (25, 359.85100720265628849), (26, 371.84147764836780187), (27, 383.79371226466881223), (28, 395.70988089399994073), (29, 407.59188220054527277), (30, 419.44150867845537373), (31, 431.26033550021001997), (32, 443.04987285220273052), (33, 454.81147221081922999), (34, 466.54645370480893457), (35, 478.25594059309893721), (36, 489.94104056621675652), (37, 501.60274579810184209), (38, 513.24207330833960572), (39, 524.85983580646023339), (40, 536.45688598866419194), (41, 548.03399620404267384), (42, 559.59188451926507002), (43, 571.13127973254151287), (44, 582.65278616200897643), (45, 594.15701982627148642), (46, 605.64456035814804691), (47, 617.11593961883466847), (48, 628.57170193793512336), (49, 640.0123014000995455), (50, 651.43823892632214934), (51, 662.84992186514829277), (52, 674.24777851463038131), (53, 685.63220582033170314), (54, 697.00357217790089892), (55, 708.36228952392155422), (56, 719.70863491784629157), (57, 731.04300093527000082), (58, 742.36565691894599961), (59, 753.67691947970915894), (60, 764.97708472053819151), (61, 776.26642987792063488), (62, 787.54521479547484686), (63, 798.81368324995423569)]


# Result of generateList(200, 10, 175)
ticks_expectedNumOtherOccurrencesOfMostUniqueFeature_200_features_10_locationsPerObject = [(0, 0.), (1, 70.230862035193098), (2, 102.6060368217816), (3, 132.18080065565022), (4, 160.28751037290095)]



if __name__ == "__main__":
  # print generateExpectedList(100, 10, 100)
  print generateLowerBoundList(0.8, 100, 10, 800)
