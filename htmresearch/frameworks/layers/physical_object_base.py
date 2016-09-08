# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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

from abc import ABCMeta, abstractmethod

try:
  from mpl_toolkits.mplot3d import Axes3D
except ImportError:
  print "Update matplotlib or don't use plotting functions."
import matplotlib.pyplot as plt



class PhysicalObject(object):
  """
  Base class to create physical objects, for L4-L2 inference experiments.
  It is assumed that objects have continuous locations (that will be encoded
  by a coordinate encoder), and that features are constant over ranges of
  locations. Features are defined as feature indices (which will be mapped
  to random SDR's by the object machine.

  All objects should implement the three abstract methods defined below.
  """

  __metaclass__ = ABCMeta

  # typical feature indices
  EMPTY_FEATURE = -1
  FLAT = 0
  EDGE = 1
  SPHERICAL_SURFACE = 2
  CYLINDER_SURFACE = 3
  CYLINDER_EDGE = 4
  POINTY = 5

  # default resolution to use for matching locations (to avoid having a zero
  # null probability of sampling an edge)
  DEFAULT_EPSILON = 1

  @abstractmethod
  def getFeatureID(self, location):
    """
    Returns the feature index associated with the provided location.

    If the location is not valid (i.e. not on the object's surface), -1 is
    returned, which will yield an empty sensory input.
    """

  @abstractmethod
  def contains(self, location):
    """
    Checks that the object contains the provided location, i.e. that it is on
    the object's surface (at epsilon's precision).
    """


  @abstractmethod
  def sampleLocation(self):
    """
    Sample a location from the object. The locations should be sampled
    uniformly whenever is possible.
    """

  @abstractmethod
  def sampleLocationFromFeature(self, feature):
    """
    Samples a location from the provided specific feature.
    """


  def almostEqual(self, number, other):
    """
    Checks that the two provided number are equal with a precision of epsilon.

    Epsilon should be specified at construction, otherwise a default value
    will be used.
    """
    return abs(number - other) <= self.epsilon


  def getFeatures(self):
    """
    Returns the list of object feature spans, from which the user can sample.
    """
    return self.features


  def plot(self, numPoints=100):
    """
    Plots the object in a 3D scatter.

    This method should be overriden when possible. This default behavior simply
    samples numPoints points from the object and plots them in a 3d scatter.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for feature in self.features:

      for _ in xrange(numPoints):
        x, y, z = tuple(self.sampleLocationFromFeature(feature))
        ax.scatter(x, y, z, marker=".")

      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Z')

    plt.title("{}".format(self))
    return fig, ax
