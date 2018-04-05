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

"""Ideal model for recognizing objects/environments with noise."""

import numpy as np


def run(numObjects, numFeatures, noise):
  print "Simulating {} objects with {} possible features and {} noise".format(numObjects, numFeatures, noise)
  correct = 0.0
  total = 0.0

  for trial in xrange(40):
    objs = [np.random.randint(numFeatures, size=16) for _ in xrange(numObjects)]
    for i, obj in enumerate(objs):
      withNoise = np.copy(obj)
      withNoise[:noise] = np.random.randint(numFeatures, size=noise)

      scores = [np.sum(withNoise == other) for other in objs]
      best = np.argmax(scores)

      total += 1.0
      if best == i:
        correct += 1.0

  print "Accuracy: ", correct / total




if __name__ == "__main__":
  for noise in (0, 2, 4, 6, 8, 10, 12, 14):
    run(numObjects=1000, numFeatures=50, noise=noise)
