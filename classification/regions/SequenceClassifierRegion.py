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
This file implements the Sequence Classifier region. See the comments in the class
definition of SequenceClassifierRegion for a description.
"""

from nupic.regions.PyRegion import PyRegion
from classification.algorithms.sequence_classifier_factory import SequenceClassifierFactory



class SequenceClassifierRegion(PyRegion):
  """
  A Sequence classifier accepts a binary input from the level below (the
  "activationPattern") and information from the sensor and encoders (the
  "classification") describing the input to the system at that time step.

  When learning, for every bit in activation pattern, it records a history of the
  classification each time that bit was active. The history is weighted so that
  more recent activity has a bigger impact than older activity. The alpha
  parameter controls this weighting.

  For inference, it takes an ensemble approach. For every active bit in the
  activationPattern, it looks up the most likely classification(s) from the
  history stored for that bit and then votes across these to get the resulting
  classification(s).
  """



  @classmethod
  def getSpec(cls):
     
    
    spec = {
        'description': SequenceClassifierRegion.__doc__,
        'singleNodeOnly': True,

        'inputs': {
          'categoryIn': {
            'description': 'Category of the input sample',
            'dataType': 'Real32',
            'count': 1, # TODO: number of categories can be more than 1 in the future
            'required': True,
            'regionLevel': True,
            'isDefaultInput': False,
            'requireSplitterMap':False
            },

          'bottomUpIn': {
            'description': 'Belief values over children\'s groups',
            'dataType': 'Real32',
            'count': 0,
            'required': True,
            'regionLevel': False,
            'isDefaultInput': True,
            'requireSplitterMap': False
          },
        },

        'outputs': {
          # TODO: categoriesOut and categoryProbabilitiesOut needs to be combined into a dict: {categoryValue: probability}
          'categoryActualValuesOut': {
            'description': 'A vector representing the actual value of each category',
          'dataType':'Real32',
          'count':0,
          'regionLevel':True,
          'isDefaultOutput':True
          },

          'categoryProbabilitiesOut': {
          'description':'A vector representing, for each category '
                      'index, the probability that the input to the node belongs '
                      'to that category.',
          'dataType':'Real32',
          'count':0,
          'regionLevel':True,
          'isDefaultOutput':True,
          },
          
          'categoryOut': {
          'description':'The most likely category that the input belongs to.',
          'dataType':'UInt32',
          'count':0,
          'regionLevel':True,
          'isDefaultOutput':True,
          },
          
        },

        'parameters': {
          
          'maxCategoryCount': {
            'description': 'The maximal number of categories the '
                        'classifier will distinguish between.',
            'dataType': 'UInt32',
            'count': 1,
            'constraints': '',
            'defaultValue': None,
            'accessMode':'Create'
          },
          'learningMode': {
            'description': 'Boolean (0/1) indicating whether or not a region '
                        'is in learning mode.',
            'dataType': 'UInt32',
            'count': 1,
            'constraints': 'bool',
            'defaultValue': 1,
            'accessMode': 'ReadWrite'
          },

          'inferenceMode': {
            'description': 'Boolean (0/1) indicating whether or not a region '
                        'is in inference mode.',
            'dataType': 'UInt32',
            'count': 1,
            'constraints': 'bool',
            'defaultValue': 0,
            'accessMode': 'ReadWrite'
          },

          #TODO: this should go away for sequence classification. we always map TM states to current timestep
          'steps': {
            'description': 'Comma separated list of the desired steps of '
                        'prediction that the classifier should learn',
            'dataType': "Byte",
            'count': 0,
            'constraints': '',
            'defaultValue': '0',
            'accessMode': 'Create'
          },

          'alpha': {
            'description': 'The alpha used to compute running averages of the '
               'bucket duty cycles for each activation pattern bit. A lower '
               'alpha results in longer term memory',
            'dataType': "Real32",
            'count': 1,
            'constraints': '',
            'defaultValue': 0.001,
            'accessMode': 'Create'
          },

          'implementation': {
            'description': 'The classifier implementation to use.',
            'accessMode': 'ReadWrite',
            'dataType': 'Byte',
            'count': 0,
            'constraints': 'enum: py' # Removed 'cpp' implementation, since it doesn't exist -- for now.
          },
    
           'clVerbosity': {
            'description': 'An integer that controls the verbosity level, '
                        '0 means no verbose output, increasing integers '
                        'provide more verbosity.',
            'dataType': 'UInt32',
            'count': 1,
            'constraints': '',
            'defaultValue': 0 ,
            'accessMode': 'ReadWrite'
           },
            

      },
      'commands': {}
    }

    return spec


  def __init__(self,
               steps='0',
               alpha=0.001,
               clVerbosity=0,
               implementation='py',
               maxCategoryCount=None
               ):

    # Convert the steps designation to a list
    self.steps = steps
    self.stepsList = eval("[%s]" % (steps))
    self.alpha = alpha
    self.verbosity = clVerbosity

    # Initialize internal structures
    self._claClassifier = SequenceClassifierFactory.create(
        steps=self.stepsList,
        alpha=self.alpha,
        verbosity=self.verbosity,
        implementation=implementation,
        )
    self.learningMode = True
    self.inferenceMode = False

    self._initEphemerals()
    
    self.recordNum = 0
    if maxCategoryCount:
      self.maxCategoryCount = maxCategoryCount
    else:
      raise Exception("'maxCategoryCount' value needs to be specified in in the input params of the classifier.")

  def _initEphemerals(self):
    pass


  def initialize(self, dims, splitterMaps):
    pass


  def clear(self):
    self._claClassifier.clear()


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

    patternNZ = inputs['bottomUpIn'].nonzero()[0]
    classificationIn =  {"bucketIdx": int(inputs['categoryIn'][0]),
                          "actValue": int(inputs['categoryIn'][0])
                         }
    
    
    clResults = self._claClassifier.compute(recordNum=self.recordNum,
                                patternNZ=patternNZ,
                                classification=classificationIn,
                                learn = self.learningMode,
                                infer = self.inferenceMode)
    
    
    # populate results     
    clResultsSize = len(clResults["actualValues"])
    for i in xrange(clResultsSize):
      outputs['categoryActualValuesOut'][i] = clResults["actualValues"][i]
      outputs['categoryProbabilitiesOut'][i] = clResults[int(self.steps)][i]
      
    outputs['categoryOut'][0] = clResults["actualValues"][clResults[int(self.steps)].argmax()]

    self.recordNum += 1

  def customCompute(self, recordNum, patternNZ, classification):
    """
    Process one input sample.
    This method is called by outer loop code outside the nupic-engine. We
    use this instead of the nupic engine compute() because our inputs and
    outputs aren't fixed size vectors of reals.

    Parameters:
    --------------------------------------------------------------------
    patternNZ:      list of the active indices from the output below
    classification: dict of the classification information:
                      bucketIdx: index of the encoder bucket
                      actValue:  actual value going into the encoder

    retval:     dict containing inference results, one entry for each step in
                self.steps. The key is the number of steps, the value is an
                array containing the relative likelihood for each bucketIdx
                starting from bucketIdx 0.

                for example:
                  {1 : [0.1, 0.3, 0.2, 0.7]
                   4 : [0.2, 0.4, 0.3, 0.5]}
    """

    return self._claClassifier.compute( recordNum=recordNum,
                                        patternNZ=patternNZ,
                                        classification=classification,
                                        learn = self.learningMode,
                                        infer = self.inferenceMode)


  def getOutputElementCount(self, name):
    """This method will be called only when the node is used in nuPIC 2"""
    if name == 'categoryActualValuesOut':
      return self.maxCategoryCount
    elif name == 'categoryProbabilitiesOut':
      return self.maxCategoryCount
    elif name == 'categoryOut':
      return 1
    else:
      raise Exception('Unknown output: ' + name)
  
if __name__=='__main__':
  from nupic.engine import Network
  n = Network()
  classifier = n.addRegion(
    'classifier',
    'py.SequenceClassifierRegion',
    '{}'
  )
