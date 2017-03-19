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

"""
Actual implementation of physical objects.

Note that because locations are integers, rather large object sizes should be
used.
"""

import random
from math import pi, cos, sin, sqrt
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt

from htmresearch.frameworks.layers.physical_object_base import PhysicalObject

import plyfile as ply


class Sphere(PhysicalObject):
  """
  A classic sphere.

  It's particularity is that it has only one feature.

  Example:
    sphere = Sphere(radius=20, dimension=3, epsilon=1)

  It's only feature is its surface.

  """

  _FEATURES = ["surface"]

  def __init__(self, radius, dimension=3, epsilon=None):
    """
    The only key parameter to provide is the sphere's radius.

    Supports arbitrary dimensions.

    Parameters:
    ----------------------------
    @param    radius (int)
              Sphere radius.

    @param    dimension (int)
              Space dimension. Typically 3.

    @param    epsilon (float)
              Object resolution. Defaults to self.DEFAULT_EPSILON

    """
    self.radius = radius
    self.dimension = dimension

    if epsilon is None:
      self.epsilon = self.DEFAULT_EPSILON
    else:
      self.epsilon = epsilon


  def getFeatureID(self, location):
    """
    Returns the feature index associated with the provided location.

    In the case of a sphere, it is always the same if the location is valid.
    """
    if not self.contains(location):
      return self.EMPTY_FEATURE

    return self.SPHERICAL_SURFACE


  def contains(self, location):
    """
    Checks that the provided point is on the sphere.
    """
    return self.almostEqual(
      sum([coord ** 2 for coord in location]), self.radius ** 2
    )


  def sampleLocation(self):
    """
    Samples from the only available feature.
    """
    return self.sampleLocationFromFeature(self._FEATURES[0])


  def sampleLocationFromFeature(self, feature):
    """
    Samples a location from the provided specific feature.

    In the case of a sphere, there is only one feature.
    """
    if feature == "surface":
      coordinates = [random.gauss(0, 1.) for _ in xrange(self.dimension)]
      norm = sqrt(sum([coord ** 2 for coord in coordinates]))
      return [self.radius * coord / norm for coord in coordinates]
    elif feature == "random":
      return self.sampleLocation()
    else:
      raise NameError("No such feature in {}: {}".format(self, feature))


  def plot(self, numPoints=100):
    """
    Specific plotting method for cylinders.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # generate sphere
    phi, theta = np.meshgrid(
      np.linspace(0, pi, numPoints),
      np.linspace(0, 2 * pi, numPoints)
    )
    x = self.radius * np.sin(phi) * np.cos(theta)
    y = self.radius * np.sin(phi) * np.sin(theta)
    z = self.radius * np.cos(phi)

    # plot
    ax.plot_surface(x, y, z, alpha=0.2, rstride=20, cstride=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.title("{}".format(self))
    return fig, ax


  def __repr__(self):
    """
    Custom representation.
    """
    template = self.__class__.__name__ + "(R={})"
    return template.format(self.radius)



class Cylinder(PhysicalObject):
  """
  A classic cylinder.

  Example:
    cyl = Cylinder(height=20, radius=5, epsilon=1)

  It has five different features to sample locations from: topDisc, bottomDisc,
  topEdge, bottomEdge, and side.
  """

  _FEATURES = ["topDisc", "bottomDisc", "topEdge", "bottomEdge", "side"]


  def __init__(self, height, radius, epsilon=None):
    """
    The two key parameters are height and radius.

    Does not support arbitrary dimensions.

    Parameters:
    ----------------------------
    @param    height (int)
              Cylinder height.

    @param    radius (int)
              Cylinder radius.

    @param    dimension (int)
              Space dimension. Typically 3.

    @param    epsilon (float)
              Object resolution. Defaults to self.DEFAULT_EPSILON

    """
    self.radius = radius
    self.height = height
    self.dimension = 3  # no choice for cylinder dimension

    if epsilon is None:
      self.epsilon = self.DEFAULT_EPSILON
    else:
      self.epsilon = epsilon


  def getFeatureID(self, location):
    """
    Returns the feature index associated with the provided location.

    Three possibilities:
      - top discs
      - edges
      - lateral surface

    """
    if not self.contains(location):
      return self.EMPTY_FEATURE

    if self.almostEqual(abs(location[2]), self.height / 2.):
      if self.almostEqual(location[0] ** 2 + location[1] ** 2,
                          self.radius ** 2):
        return self.CYLINDER_EDGE
      else:
        return self.FLAT
    else:
      return self.CYLINDER_SURFACE


  def contains(self, location):
    """
    Checks that the provided point is on the cylinder.
    """
    if self.almostEqual(location[0] ** 2 + location[1] ** 2, self.radius ** 2):
      return abs(location[2]) < self.height / 2.
    if self.almostEqual(location[2], self.height / 2.):
      return location[0] ** 2 + location[1] ** 2 < self.radius ** 2
    return False


  def sampleLocation(self):
    """
    Simple method to sample uniformly from a cylinder.
    """
    areaRatio = self.radius / (self.radius + self.height)
    if random.random() < areaRatio:
      return self._sampleLocationOnDisc()
    else:
      return self._sampleLocationOnSide()


  def sampleLocationFromFeature(self, feature):
    """
    Samples a location from the provided specific features.
    """
    if feature == "topDisc":
      return self._sampleLocationOnDisc(top=True)
    elif feature == "topEdge":
      return self._sampleLocationOnEdge(top=True)
    elif feature == "bottomDisc":
      return self._sampleLocationOnDisc(top=False)
    elif feature == "bottomEdge":
      return self._sampleLocationOnEdge(top=False)
    elif feature == "side":
      return self._sampleLocationOnSide()
    elif feature == "random":
      return self.sampleLocation()
    else:
      raise NameError("No such feature in {}: {}".format(self, feature))


  def _sampleLocationOnDisc(self, top=None):
    """
    Helper method to sample from the top and bottom discs of a cylinder.

    If top is set to True, samples only from top disc. If top is set to False,
    samples only from bottom disc. If not set (defaults to None), samples from
    both discs.
    """
    if top is None:
      z = random.choice([-1, 1]) * self.height / 2.
    else:
      z = self.height / 2. if top else - self.height / 2.
    sampledAngle = 2 * random.random() * pi
    sampledRadius = self.radius * sqrt(random.random())
    x, y = sampledRadius * cos(sampledAngle), sampledRadius * sin(sampledAngle)
    return [x, y, z]


  def _sampleLocationOnEdge(self, top=None):
    """
    Helper method to sample from the top and bottom edges of a cylinder.

    If top is set to True, samples only from top edge. If top is set to False,
    samples only from bottom edge. If not set (defaults to None), samples from
    both edges.
    """
    if top is None:
      z = random.choice([-1, 1]) * self.height / 2.
    else:
      z = self.height / 2. if top else - self.height / 2.
    sampledAngle = 2 * random.random() * pi
    x, y = self.radius * cos(sampledAngle), self.radius * sin(sampledAngle)
    return [x, y, z]


  def _sampleLocationOnSide(self):
    """
    Helper method to sample from the lateral surface of a cylinder.
    """
    z = random.uniform(-1, 1) * self.height / 2.
    sampledAngle = 2 * random.random() * pi
    x, y = self.radius * cos(sampledAngle), self.radius * sin(sampledAngle)
    return [x, y, z]


  def plot(self, numPoints=100):
    """
    Specific plotting method for cylinders.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # generate cylinder
    x = np.linspace(- self.radius, self.radius, numPoints)
    z = np.linspace(- self.height / 2., self.height / 2., numPoints)
    Xc, Zc = np.meshgrid(x, z)
    Yc = np.sqrt(self.radius ** 2 - Xc ** 2)

    # plot
    ax.plot_surface(Xc, Yc, Zc, alpha=0.2, rstride=20, cstride=10)
    ax.plot_surface(Xc, -Yc, Zc, alpha=0.2, rstride=20, cstride=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.title("{}".format(self))
    return fig, ax


  def __repr__(self):
    """
    Custom representation.
    """
    template = self.__class__.__name__ + "(H={}, R={})"
    return template.format(self.height, self.radius)



class Box(PhysicalObject):
  """
  A box is a classic cuboid.

  Example:
    box = Box(dimensions=[10, 10, 5], dimension=3, epsilon=1)

  It has three features to sample locations from: face, edge, and vertex.
  """

  _FEATURES = ["face", "edge", "vertex"]


  def __init__(self, dimensions, dimension=3, epsilon=None):
    """
    The only key parameter is the list (or tuple) of dimensions, which can be
    of any size as long as its length is equal to the "dimension" parameter.

    Parameters:
    ----------------------------
    @param    dimensions (list(int))
              List of the box's dimensions.

    @param    dimension (int)
              Space dimension. Typically 3.

    @param    epsilon (float)
              Object resolution. Defaults to self.DEFAULT_EPSILON

    """
    self.dimensions = dimensions
    self.dimension = dimension

    if epsilon is None:
      self.epsilon = self.DEFAULT_EPSILON
    else:
      self.epsilon = epsilon


  def getFeatureID(self, location):
    """
    There are three possible features for a box:
      - flat for a face
      - pointy for a vertex
      - edge

    """
    if not self.contains(location):
      return self.EMPTY_FEATURE  # random feature

    numFaces = sum(
      [self.almostEqual(abs(coord), self.dimensions[i] / 2.) \
       for i, coord in enumerate(location)]
    )

    if numFaces == 1:
      return self.FLAT
    if numFaces == 2:
      return self.EDGE
    if numFaces == 3:
      return self.POINTY


  def contains(self, location):
    """
    A location is on the box if one of the dimension is "satured").
    """
    for i, coord in enumerate(location):
      if self.almostEqual(abs(coord), self.dimensions[i] / 2.):
        return True
    return False


  def sampleLocation(self):
    """
    Random sampling fron any location corresponds to sampling form the faces.
    """
    return self._sampleFromFaces()


  def sampleLocationFromFeature(self, feature):
    """
    Samples a location from one specific feature.

    This is only supported with three dimensions.
    """
    if feature == "face":
      return self._sampleFromFaces()
    elif feature == "edge":
      return self._sampleFromEdges()
    elif feature == "vertex":
      return self._sampleFromVertices()
    elif feature == "random":
      return self.sampleLocation()
    else:
      raise NameError("No such feature in {}: {}".format(self, feature))


  def _sampleFromFaces(self):
    """
    We start by sampling a dimension to "max out", then sample the sign and
    the other dimensions' values.
    """
    coordinates = [random.uniform(-1, 1) * dim / 2. for dim in self.dimensions]
    dim = random.choice(range(self.dimension))
    coordinates[dim] = self.dimensions[dim] / 2. * random.choice([-1, 1])
    return coordinates


  def _sampleFromEdges(self):
    """
    We start by sampling dimensions to "max out", then sample the sign and
    the other dimensions' values.
    """
    dimensionsToChooseFrom = range(self.dimension)
    random.shuffle(dimensionsToChooseFrom)
    dim1, dim2 = dimensionsToChooseFrom[0], dimensionsToChooseFrom[1]
    coordinates = [random.uniform(-1, 1) * dim / 2. for dim in self.dimensions]
    coordinates[dim1] = self.dimensions[dim1] / 2. * random.choice([-1, 1])
    coordinates[dim2] = self.dimensions[dim2] / 2. * random.choice([-1, 1])
    return coordinates


  def _sampleFromVertices(self):
    """
    We start by sampling dimensions to "max out", then sample the sign and
    the other dimensions' values.
    """
    coordinates = [
      self.dimensions[0] / 2. * random.choice([-1, 1]),
      self.dimensions[1] / 2. * random.choice([-1, 1]),
      self.dimensions[2] / 2. * random.choice([-1, 1]),
    ]
    return coordinates

  def plot(self, numPoints=100):
    """
    Specific plotting method for boxes.

    Only supports 3-dimensional objects.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # generate cylinder
    x = np.linspace(- self.dimensions[0]/2., self.dimensions[0]/2., numPoints)
    y = np.linspace(- self.dimensions[1]/2., self.dimensions[1]/2., numPoints)
    z = np.linspace(- self.dimensions[2]/2., self.dimensions[2]/2., numPoints)

    # plot
    Xc, Yc = np.meshgrid(x, y)
    ax.plot_surface(Xc, Yc, -self.dimensions[2]/2,
                    alpha=0.2, rstride=20, cstride=10)
    ax.plot_surface(Xc, Yc, self.dimensions[2]/2,
                    alpha=0.2, rstride=20, cstride=10)
    Yc, Zc = np.meshgrid(y, z)
    ax.plot_surface(-self.dimensions[0]/2, Yc, Zc,
                    alpha=0.2, rstride=20, cstride=10)
    ax.plot_surface(self.dimensions[0]/2, Yc, Zc,
                    alpha=0.2, rstride=20, cstride=10)
    Xc, Zc = np.meshgrid(x, z)
    ax.plot_surface(Xc, -self.dimensions[1]/2, Zc,
                    alpha=0.2, rstride=20, cstride=10)
    ax.plot_surface(Xc, self.dimensions[1]/2, Zc,
                    alpha=0.2, rstride=20, cstride=10)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.title("{}".format(self))
    return fig, ax


  def __repr__(self):
    """
    Custom representation.
    """
    template = self.__class__.__name__ + "(dim=({}))"
    return template.format(", ".join([str(dim) for dim in self.dimensions]))



class Cube(Box):
  """
  A cube is a particular box where all dimensions have equal length.


  Example:
    cube = Cube(width=100, dimension=3, epsilon=2)
  """


  def __init__(self, width, dimension=3, epsilon=None):
    """
    We simply pass the width as every dimension.

    Parameters:
    ----------------------------
    @param    width (int)
              Cube width.

    @param    dimension (int)
              Space dimension. Typically 3.

    @param    epsilon (float)
              Object resolution. Defaults to self.DEFAULT_EPSILON

    """
    self.width = width
    dimensions = [width] * dimension
    super(Cube, self).__init__(dimensions=dimensions,
                               dimension=dimension,
                               epsilon=epsilon)


  def __repr__(self):
    """
    Custom representation.
    """
    template = self.__class__.__name__ + "(width={})"
    return template.format(self.width)


class PlyModel(PhysicalObject):
  """
  A 3D .ply format model loader as a physical object.

  '.ply' format is simple polygon format. 
  It can represent vertices, edges and faces (color, material) and 
  custom attributes.
  
  TODO:
  Compute normals of adjacent face of random sample and decide 
  if its a FLAT or CURVED surface

  Example:
    appleModel = PlyModel(file='apple.ply', normalTolerance=0.3, epsilon=.1)

  """

  _FEATURES = ["face", "vertex", "edge", "surface"]

  def __init__(self, file=None, normalTolerance = 0., epsilon=None):
    """
    The only key parameter to provide is location of file.

    Supports arbitrary dimensions.

    Parameters:
    ----------------------------
    @param    file (string)
              string representing location of file (.ply to be specific) .

    @param    epsilon (float)
              Object resolution. Defaults to self.DEFAULT_EPSILON
    
    @param    normalTolerance (float)
              Adjacent Faces Normal Tolerance. Defaults to zero - edges appear more.

    """
    try:
      self.file = file
      self.model = ply.PlyData.read(self.file)
      self.vertices = self.model['vertex']
      self.faces = self.model['face']
    except IOError as e:
      print("Something went wrong!")
      print("Please check if file exists at {}".format(file))
      raise IOError
    self.graphicsWindow = None
    self.mesh = None
    self.rng = random.Random()
    self.epsilon = self.DEFAULT_EPSILON if epsilon is None else epsilon
    self.sampledPoints = {i:[] for i in self._FEATURES}
    self.nTol = normalTolerance

  def getFeatureID(self, location):
    """
    Returns the feature index associated with the provided location.

    In the case of a sphere, it is always the same if the location is valid.
    """
    truthyFeature = self.contains(location)
    if not truthyFeature:
      return self.EMPTY_FEATURE
    elif truthyFeature=='face':
      return self.FLAT
    elif truthyFeature=='vertex':
      return self.POINTY
    elif truthyFeature=="edge":
      return self.EDGE
    elif truthyFeature=="surface":
      return self.SURFACE
    else:
      return self.EMPTY_FEATURE

  def _containsOnFace(self, location, face):
    vertices = self.vertices[face]
    vertices = np.array((vertices['x'], vertices['y'], vertices['z'])).T
    v0 = vertices[2] - vertices[0]
    v1 = vertices[1] - vertices[0]
    v2 = location - vertices[0]
    v3 = vertices[2] - location
    N1 = np.cross(v0,v1)
    N1 = N1/(np.dot(N1,N1))**.5

    N2 = np.cross(v2,v3)
    N2 = N2/(np.dot(N2,N2))**.5
    if self.almostEqual(abs(np.dot(N1,N2)), 1.0):
      return True
    return False
  
  def _containsOnEdge(self, location, edge):
    edge = np.array((edge['x'], edge['y'], edge['z'])).T
    v = edge[1] - location
    av = v/(np.dot(v,v))**.5
    d = edge[1] - edge[0]
    ad = d/(np.dot(d,d))**.5
    vd = np.dot(av,ad)
    self.epsilon = 0.0001
    if self.almostEqual(vd, 1.0):
          self.epsilon = self.DEFAULT_EPSILON
          return True

  def contains(self, location):
    """
    Checks that the provided point is on the sphere.
    TODO: Temporary Hack... need math for this
    """
    for vertex in self.vertices:
      V = np.array((vertex['x'], vertex['y'], vertex['z'])).T
      if np.allclose(location, V, rtol=1.e-3):
        return "vertex"
    for face in self.faces:
      edges = np.choose(np.array(list(combinations(range(3),2))), self.vertices[face])
      for edge in edges:
        if self._containsOnEdge(location, edge):
          return "edge"
      if self._containsOnFace(location,face):
        return "face"
    return False

  def sampleLocation(self):
    """
    Samples from the only available feature.
    """
    return self.sampleLocationFromFeature(random.choice(self._FEATURES))


  def sampleLocationFromFeature(self, feature):
    """
    Samples a location from the provided specific feature.

    TODO surface feature is not handled correctly.
    forwarded to sample from face instead.
    """
    if feature == "surface":
        #TODO
        pass
    elif feature=="face":
      indx = self.rng.choice(range(self.faces.count))
      rndFace = self.faces[indx]
      return self._sampleLocationOnFace(rndFace)

    elif feature == "edge":
      indx = self.rng.choice(range(self.faces.count))
      rndFace = self.faces[indx]
      rndVertices = self.rng.sample(self.vertices[rndFace],2)
      return self._sampleLocationOnEdge(rndVertices)

    elif feature == "vertex":
      rndVertexIndx = self.rng.choice(range(self.vertices.count))
      return np.array(self.vertices[rndVertexIndx].tolist())

    elif feature == "surface":
      return self.sampleLocationFromFeature("face")      # Temporary workaround for surfaces
    elif feature == "random":
      return self.sampleLocation()
    else:
      raise NameError("No such feature in {}: {}".format(self, feature))

  def _sampleLocationOnEdge(self, vertices):
    rnd = self.rng.random()
    vertices = np.array([i.tolist() for i in vertices])
    return np.array([rnd, 1-rnd]).dot(vertices)

  def _sampleLocationOnFace(self,face):
    vertices = self.vertices[face]
    vertices = np.array((vertices['x'], vertices['y'], vertices['z'])).T
    r1 = self.rng.random()
    r2 = self.rng.random()
    return (1 - sqrt(r1))*vertices[0] + sqrt(r1)*(1 - r2)*vertices[1] + sqrt(r1)*r2*vertices[2]
  
  def plot(self, numPoints=100):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    (x,y,z) = (self.vertices[t] for t in ('x', 'y', 'z'))
    tri_idx = self.faces['vertex_indices']
    idx_dtype = tri_idx[0].dtype
    triangles = np.fromiter(tri_idx, [('data', idx_dtype, (3,))],
                                      count=len(tri_idx))['data']
    ax.plot_trisurf(x,y,z, triangles=triangles)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.title("{}".format(self))
    return fig, ax

  def visualize(self, numPoints=100):
    """
    Visualization utility for models.
    Helps to debug the math and logic. 
    Helps to monitor complex objects with difficult to define boundaries.

    Only supports 3-dimensional objects.
    TODO: center the objects using scale, rotate and translate operations on mesh objects.
    """
    try:
      import pyqtgraph as pg
      import pyqtgraph.multiprocess as mp
      import pyqtgraph.opengl as gl
    except ImportError as e:
      print("PyQtGraph needs to be installed.")
      return (None, None, None, None, None)

    class PlyVisWindow:
      """
      The pyqtgraph visualization utility window class

      Creates a remote process with viewbox frame for visualizations
      Provided access to mesh and scatter for realtime update to view. 
      """
      def __init__(self): 
        self.proc = mp.QtProcess()
        self.rpg = self.proc._import('pyqtgraph')
        self.rgl = self.proc._import('pyqtgraph.opengl')
        self.rview = self.rgl.GLViewWidget()
        self.rview.setBackgroundColor('k')
        self.rview.setCameraPosition(distance=10)
        self.grid = self.rgl.GLGridItem()
        self.rview.addItem(self.grid)
        self.rpg.setConfigOption('background', 'w')
        self.rpg.setConfigOption('foreground', 'k')

      def snapshot(self, name=""):
        """
        utility to grabframe of the visualization window.

        @param name (string) helps to avoid overwriting grabbed images programmatically.
        """
        self.rview.grabFrameBuffer().save("{}.png".format(name))

    
    # We might need this for future purposes Dont Delete
    # class MeshUpdate:
    #     def __init__(self, proc):
    #         self.data_x = proc.transfer([])
    #         self.data_y = proc.transfer([])
    #         self._t = None

    #     @property
    #     def t(self):
    #         return self._t
        
    #     def update(self,x):
    #         self.data_y.extend([x], _callSync='async')
    #         self.data_x.extend([self.t], _callSync='async',)
    #         self.curve.setData(y=self.data_y, _callSync='async')
    
    pg.mkQApp()
    self.graphicsWindow = PlyVisWindow()
    self.graphicsWindow.rview.setWindowTitle(self.file)
    vertices = self.vertices.data
    vertices = np.array(vertices.tolist())
    faces = np.array([self.faces[i]['vertex_indices'] for i in  range(self.faces.count)])
    self.mesh = self.graphicsWindow.rgl.GLMeshItem(vertexes=vertices, faces=faces, 
                    shader='normalColor', drawEdges=True, 
                    drawFaces=True, computeNormals=False, 
                    smooth=False)
    self.graphicsWindow.rview.addItem(self.mesh)
    self.graphicsWindow.rview.show()
    pos = np.empty((numPoints,3))
    size = np.ones((numPoints,))
    color = np.ones((numPoints,4))
    self.scatter = self.graphicsWindow.rgl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=True)
    self.graphicsWindow.rview.addItem(self.scatter)
    return self.scatter, self.mesh, pos, size, color


  def __repr__(self):
    """
    Custom representation.
    """
    template = self.__class__.__name__ + "(Model obj: \n Vertices: \n {}\n Faces:\n{}\n)"
    return template.format(self.vertices.data, self.faces.data)
  
  def __str__(self):
    """
    Custom string
    """
    template = self.__class__.__name__+ " {} "+ " Vertices: {} Faces: {}"
    return template.format(self.file.split('/')[-1].split(".")[-2],self.vertices.count, self.faces.count)
