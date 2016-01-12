# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
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

from htmresearch.frameworks.nlp.classify_htm import ClassificationModelHTM
from htmresearch.frameworks.nlp.classify_keywords import (
  ClassificationModelKeywords)
from htmresearch.frameworks.nlp.classify_document_fingerprint import (
  ClassificationModelDocumentFingerprint)
from htmresearch.frameworks.nlp.classify_fingerprint import (
  ClassificationModelFingerprint)

"""
This module contains functions helpful for creating ClassificationModel
instances.
"""


def getNetworkConfig(networkConfigPath):
  """
  Given path to JSON model config file, return a dict.
  """
  try:
    with open(networkConfigPath, "rb") as fin:
      return simplejson.load(fin)
  except IOError as e:
    print "Could not find network configuration JSON at \'{}\'.".format(
      networkConfigPath)
    raise e


def createModel(modelName, **kwargs):
  """
  Return a classification model of the appropriate type. The model could be any
  supported subclass of ClassficationModel based on modelName.

  @param modelName (str)  A string representing a supported model type:
                            htm      : ClassificationModelHTM
                            keywords : ClassificationModelKeywords
                            docfp    : ClassificationModelDocumentFingerprint
                            cioword  : ClassificationModelFingerprint

  @param kwargs    (dict) Constructor argument for the class that will be
                          instantiated. Keyword parameters specific to each
                          model type should be passed in here.
  """

  print kwargs

  if modelName == "htm":
    # Instantiate the HTM model
    model = ClassificationModelHTM(**kwargs)

  elif modelName == "keywords":
    # Instantiate the keywords model
    model = ClassificationModelKeywords(**kwargs)

  elif modelName == "docfp":
    # Instantiate the document fingerprint model
    model = ClassificationModelDocumentFingerprint(**kwargs)

  elif modelName == "cioword":
    # Instantiate the Cio word fingerprint model
    model = ClassificationModelFingerprint(**kwargs)

  else:
    raise RuntimeError("Unknown model type: " + args.modelName)

  return model

