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

from enum import Enum
import simplejson

from htmresearch.frameworks.nlp.classify_document_fingerprint import (
  ClassificationModelDocumentFingerprint)
from htmresearch.frameworks.nlp.classify_fingerprint import (
  ClassificationModelFingerprint)
from htmresearch.frameworks.nlp.classify_htm import (
  ClassificationModelHTM)
from htmresearch.frameworks.nlp.classify_keywords import (
  ClassificationModelKeywords)
from htmresearch.frameworks.nlp.classify_windows import (
  ClassificationModelWindows)



"""
This module contains functions helpful for creating ClassificationModel
instances.
"""



class ClassificationModelTypes(Enum):
  """ Enumeration of supported classification model types, mapping userland
  identifier to constructor.  See createModel() for actual factory method
  implementation.
  """
  CioWordFingerprint = ClassificationModelFingerprint
  CioDocumentFingerprint = ClassificationModelFingerprint
  cioword = ClassificationModelFingerprint
  CioWindows = ClassificationModelWindows
  Keywords = ClassificationModelKeywords
  keywords = ClassificationModelKeywords
  HTMNetwork = ClassificationModelHTM
  htm = ClassificationModelHTM
  docfp = ClassificationModelDocumentFingerprint



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



def createModel(self, modelName, **kwargs):
  """
  Return a classification model of the appropriate type. The model could be any
  supported subclass of ClassficationModel based on modelName.

  @param modelName (str)  A string representing a supported model type:
                          HTMNetwork, htm,    : ClassificationModelHTM
                          Keywords, keywords  : ClassificationModelKeywords
                          docfp     : ClassificationModelDocumentFingerprint
                          CioWordFingerprint, CioDocumentFingerprint, cioword:
                              ClassificationModelFingerprint

  @param kwargs    (dict) Constructor argument for the class that will be
                          instantiated. Keyword parameters specific to each
                          model type should be passed in here.
  """
  if modelName in ClassificationModelTypes:
    modelConstructor = modelName
  elif modelName in ClassificationModelTypes.__members__:
    modelConstructor = getattr(ClassificationModelTypes, modelName)
  else:
    raise RuntimeError("Unknown model type: " + modelName)

  return modelConstructor.value(**kwargs)

