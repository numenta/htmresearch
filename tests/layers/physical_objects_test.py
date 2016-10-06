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

import unittest

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from htmresearch.frameworks.layers.physical_objects import (
  Sphere, Cylinder, Box, Cube
)



class PhysicalObjectsTest(unittest.TestCase):
  """Unit tests for physical objects."""


  def testInitParams(self):
    """Simple construction test."""
    sphere = Sphere(radius=5, dimension=6)
    cylinder = Cylinder(height=50, radius=100, epsilon=5)
    box = Box(dimensions=[1, 2, 3, 4], dimension=4)
    cube = Cube(width=10, dimension=2)

    self.assertEqual(sphere.radius, 5)
    self.assertEqual(sphere.dimension, 6)
    self.assertEqual(sphere.epsilon, sphere.DEFAULT_EPSILON)

    self.assertEqual(cylinder.radius, 100)
    self.assertEqual(cylinder.height, 50)
    self.assertEqual(cylinder.dimension, 3)
    self.assertEqual(cylinder.epsilon, 5)

    self.assertEqual(box.dimensions, [1, 2, 3, 4])
    self.assertEqual(box.dimension, 4)
    self.assertEqual(box.epsilon, box.DEFAULT_EPSILON)

    self.assertEqual(cube.dimensions, [10, 10])
    self.assertEqual(cube.width, 10)
    self.assertEqual(cube.dimension, 2)
    self.assertEqual(sphere.epsilon, cube.DEFAULT_EPSILON)


  def testSampleContains(self):
    """Samples points from the objects and test contains."""
    sphere = Sphere(radius=20, dimension=6)
    cylinder = Cylinder(height=50, radius=100, epsilon=2)
    box = Box(dimensions=[10, 20, 30, 40], dimension=4)
    cube = Cube(width=20, dimension=2)

    for i in xrange(50):
      self.assertTrue(sphere.contains(sphere.sampleLocation()))
      self.assertTrue(cylinder.contains(cylinder.sampleLocation()))
      self.assertTrue(box.contains(box.sampleLocation()))
      self.assertTrue(cube.contains(cube.sampleLocation()))

    # inside
    self.assertFalse(sphere.contains([1] * sphere.dimension))
    self.assertFalse(cube.contains([1] * cube.dimension))
    self.assertFalse(cylinder.contains([1] * cylinder.dimension))
    self.assertFalse(box.contains([1] * box.dimension))

    # outside
    self.assertFalse(sphere.contains([100] * sphere.dimension))
    self.assertFalse(cube.contains([100] * cube.dimension))
    self.assertFalse(cylinder.contains([100] * cylinder.dimension))
    self.assertFalse(box.contains([100] * box.dimension))


  def testPlotSampleLocations(self):
    """Samples points from objects and plots them in a 3D scatter."""
    objects = []
    objects.append(Sphere(radius=20, dimension=3))
    objects.append(Cylinder(height=50, radius=100, epsilon=2))
    objects.append(Box(dimensions=[10, 20, 30], dimension=3))
    objects.append(Cube(width=20, dimension=3))
    numPoints = 500

    for i in xrange(4):
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      for _ in xrange(numPoints):
        x, y, z = tuple(objects[i].sampleLocation())
        ax.scatter(x, y, z)

      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Z')
      plt.title("Sampled points from {}".format(objects[i]))
      plt.savefig("object{}.png".format(str(i)))
      plt.close()


  def testPlotSampleFeatures(self):
    """Samples points from objects and plots them in a 3D scatter."""
    objects = []
    objects.append(Sphere(radius=20, dimension=3))
    objects.append(Cylinder(height=50, radius=100, epsilon=2))
    objects.append(Box(dimensions=[10, 20, 30], dimension=3))
    objects.append(Cube(width=20, dimension=3))
    numPoints = 500

    for i in xrange(4):

      for feature in objects[i].features:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for _ in xrange(numPoints):
          x, y, z = tuple(objects[i].sampleLocationFromFeature(feature))
          ax.scatter(x, y, z)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title("Sampled points on {} from {}".format(feature, objects[i]))
        plt.savefig("object_{}_{}.png".format(str(i), feature))
        plt.close()



if __name__ == "__main__":
  unittest.main()
