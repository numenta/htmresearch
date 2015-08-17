#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
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

"""
This file implements the Sequence Classifier region. See the comments in the 
class definition of SequenceClassifierRegion for a description.
"""

from nupic.regions.PyRegion import PyRegion

from algorithms.sequence_classifier_factory import SequenceClassifierFactory



class SequenceClassifierRegion(PyRegion):
  """
  A Sequence classifier accepts a binary input from the level below (the
  "activationPattern") and information from the sensor and encoders (the
  "classification") describing the input to the system at that time step.

  When learning, for every bit in activation pattern, it records a history of 
  the classification each time that bit was active. The history is weighted so
  that more recent activity has a bigger impact than older activity. The alpha
  parameter controls this weighting.

  For inference, it takes an ensemble approach. For every active bit in the
  activationPattern, it looks up the most likely classification(s) from the
  history stored for that bit and then votes across these to get the resulting
  classification(s).

  """

  @classmethod
  def getSpec(cls):
    ns = dict(
      description=SequenceClassifierRegion.__doc__,
      singleNodeOnly=True,

      inputs=dict(
        categoryIn=dict(
          description='Vector of categories of the input sample',
          dataType='Real32',
          count=0,
          required=True,
          regionLevel=True,
          isDefaultInput=False,
          requireSplitterMap=False),

        bottomUpIn=dict(
          description='Belief values over children\'s groups',
          dataType='Real32',
          count=0,
          required=True,
          regionLevel=False,
          isDefaultInput=True,
          requireSplitterMap=False),
      ),

      outputs=dict(
        categoriesOut=dict(
          description='Classification results - i.e. the most likely '
                      'categorie(s)',
          dataType='Real32',
          count=0,
          required=True,
          regionLevel=True,
          isDefaultOutput=True,
          requireSplitterMap=False),
      ),

      parameters=dict(
        learningMode=dict(
          description='Boolean (0/1) indicating whether or not a region '
                      'is in learning mode.',
          dataType='UInt32',
          count=1,
          constraints='bool',
          defaultValue=1,
          accessMode='ReadWrite'),

        inferenceMode=dict(
          description='Boolean (0/1) indicating whether or not a region '
                      'is in inference mode.',
          dataType='UInt32',
          count=1,
          constraints='bool',
          defaultValue=0,
          accessMode='ReadWrite'),

        alpha=dict(
          description='The alpha used to compute running averages of the '
                      'bucket duty cycles for each activation pattern bit. A '
                      'lower '
                      'alpha results in longer term memory',
          dataType="Real32",
          count=1,
          constraints='',
          defaultValue=0.001,
          accessMode='Create'),

        implementation=dict(
          description='The classifier implementation to use.',
          accessMode='ReadWrite',
          dataType='Byte',
          count=0,
          constraints='enum: py, cpp'),

        clVerbosity=dict(
          description='An integer that controls the verbosity level, '
                      '0 means no verbose output, increasing integers '
                      'provide more verbosity.',
          dataType='UInt32',
          count=1,
          constraints='',
          defaultValue=0,
          accessMode='ReadWrite'),

      ),
      commands=dict()
    )

    return ns


  def __init__(self,
               alpha=0.001,
               clVerbosity=0,
               implementation=None,
               ):

    self.alpha = alpha
    self.verbosity = clVerbosity

    # Initialize internal structures
    self._classifier = SequenceClassifierFactory.create(
      alpha=self.alpha,
      verbosity=self.verbosity,
      implementation=implementation,
    )
    self.learningMode = True
    self.inferenceMode = False

    self._initEphemerals()

    self.recordNum = 0


  def _initEphemerals(self):
    pass


  def initialize(self, dims, splitterMaps):
    pass

  def clear(self):
    self._classifier.clear()


  def getParameter(self, name, index=-1):
    """
    Get the value of the parameter.

    @param name -- the name of the parameter to retrieve, as defined
            by the Node Spec.
    """
    # If any spec parameter name is the same as an attribute, this call
    # will get it automatically, e.g. self.learningMode
    return PyRegion.getParameter(self, name, index)


  def setParameter(self, name, index, value):
    """
    Set the value of the parameter.

    @param name -- the name of the parameter to update, as defined
            by the Node Spec.
    @param value -- the value to which the parameter is to be set.
    """
    if name == "learningMode":
      self.learningMode = bool(int(value))
    elif name == "inferenceMode":
      self.inferenceMode = bool(int(value))
    else:
      return PyRegion.setParameter(self, name, index, value)


  def reset(self):
    pass


  def compute(self, inputs, outputs):
    """
    Process one input sample.
    This method is called by the runtime engine.

    """

    # Allow training on multiple categories:
    #  An input can potentially belong to multiple categories. 
    #  If a category value is < 0, it means that the input does not belong to
    #  that category.
    categories = []
    for category in inputs["categoryIn"]:
      # if a category value <0, then it means 
      # the input record does not belong to that category.
      if category >= 0:
        categories.append(category)

    # Get TM states.
    activeCells = inputs["bottomUpIn"]
    patternNZ = activeCells.nonzero()[0]

    # Call classifier. Don't train. Just inference. Train after.
    clResults = self._classifier.compute(recordNum=self.recordNum, 
                                         patternNZ=patternNZ, 
                                         classification=None,
                                         learn=False, 
                                         infer=self.inferenceMode)

    for category in categories:
      classificationIn = {
        "bucketIdx": int(category),
        "actValue": int(category)
        }

      # Train classifier, no inference
      self._classifier.compute(recordNum=self.recordNum, 
                               patternNZ=patternNZ,
                               classification=classificationIn, 
                               learn=self.learningMode, 
                               infer=False)

    inferredValue = clResults["actualValues"][
      clResults["probabilities"].argmax()]

    outputs["categoriesOut"][0] = inferredValue

    self.recordNum += 1


  def customCompute(self, recordNum, patternNZ, classification):
    """
    Process one input sample.

    Parameters:
    --------------------------------------------------------------------
    patternNZ:      list of the active indices from the output below
    classification: dict of the classification information:
                      bucketIdx: index of the encoder bucket
                      actValue:  actual value going into the encoder

    retval:     dict containing inference results. The entry 'probabilities'
                is an array containing the relative likelihood for
                each bucketIdx starting from bucketIdx 0.

                There is also an entry containing the average actual value to
                use for each bucket. The key is 'actualValues'.

                for example:
                  {'probabilities': [0.1, 0.3, 0.2, 0.7],
                   'actualValues': [1.5, 3,5, 5,5, 7.6],
                  }
    """

    return self._classifier.compute(recordNum=recordNum,
                                    patternNZ=patternNZ,
                                    classification=classification,
                                    learn=self.learningMode,
                                    infer=self.inferenceMode)


  def getOutputValues(self, name):
    """Return the dictionary of output values. Note that these are normal Python
    lists, rather than numpy arrays. This is to support lists with mixed scalars
    and strings, as in the case of records with categorical variables
    """
    return self._outputValues[name]


  def getOutputElementCount(self, name):
    """Returns the width of dataOut."""

    if name == "categoriesOut":
      return 1
    else:
      raise Exception("Unknown output {}.".format(name))


  def getInputElementCount(self, name):
    """Returns the width of dataIn."""

    if name == "categoriesOut":
      return 1
    else:
      raise Exception("Unknown output {}.".format(name))
