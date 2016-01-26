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

import simplejson

from htmresearch.frameworks.nlp.classification_model import ClassificationModel
from htmresearch.frameworks.nlp.classify_document_fingerprint import (
  ClassificationModelDocumentFingerprint)
from htmresearch.frameworks.nlp.classify_fingerprint import (
  ClassificationModelFingerprint)
from htmresearch.frameworks.nlp.classify_htm import (
  ClassificationModelHTM)
from htmresearch.frameworks.nlp.classify_keywords import (
  ClassificationModelKeywords)
from htmresearch.frameworks.nlp.classify_network_api import (
  ClassificationNetworkAPI
)


"""
This module contains functions helpful for creating ClassificationModel
instances.
"""



class ClassificationModelTypes(object):
  """ Enumeration of supported classification model types, mapping userland
  identifier to constructor.  See createModel() for actual factory method
  implementation.
  """
  # Multiple names per model type enable us to run several instances in parallel
  CioWordFingerprint = ClassificationModelFingerprint
  CioDocumentFingerprint = ClassificationModelFingerprint
  ciodoc = ClassificationModelFingerprint
  cioword = ClassificationModelFingerprint

  Keywords = ClassificationModelKeywords
  keywords = ClassificationModelKeywords

  HTMNetwork = ClassificationModelHTM
  htm = ClassificationModelHTM
  htm1 = ClassificationModelHTM
  htm2 = ClassificationModelHTM
  htm3 = ClassificationModelHTM

  DocumentFingerPrint = ClassificationModelDocumentFingerprint
  docfp = ClassificationModelDocumentFingerprint


  @classmethod
  def getTypes(cls):
    """ Get sequence of acceptable model types.  Iterates through class
    attributes and separates the user-defined enumerations from the default
    attributes implicit to Python classes. i.e. this function returns the names
    of the attributes explicitly defined above.
    """

    acceptableClassImplementations = (
      ClassificationModel,
      ClassificationNetworkAPI
    )

    for attrName in dir(cls):
      attrValue = getattr(cls, attrName)
      if (isinstance(attrValue, type) and
          issubclass(attrValue, acceptableClassImplementations)):
        yield attrName # attrName is an acceptable model name



def getNetworkConfig(networkConfigPath):
  """
  Given path to JSON model config file, return a dict.
  """
  try:
    with open(networkConfigPath, "rb") as fin:
      return simplejson.load(fin)
  except IOError as e:
    print "Could not find network configuration JSON at '{}'.".format(
      networkConfigPath)
    raise e



def createModel(modelName, **kwargs):
  """
  Return a classification model of the appropriate type. The model could be any
  supported subclass of ClassficationModel based on modelName.

  @param modelName (str)  A string representing a supported model type:
                          HTMNetwork, htm    : ClassificationModelHTM
                          Keywords, keywords  : ClassificationModelKeywords
                          docfp     : ClassificationModelDocumentFingerprint
                          CioWordFingerprint, CioDocumentFingerprint, cioword:
                              ClassificationModelFingerprint

  @param kwargs    (dict) Constructor argument for the class that will be
                          instantiated. Keyword parameters specific to each
                          model type should be passed in here.
  """

  if modelName not in ClassificationModelTypes.getTypes():
    raise RuntimeError("Unknown model type: " + modelName)

  return getattr(ClassificationModelTypes, modelName)(**kwargs)

