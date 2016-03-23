# ----------------------------------------------------------------------
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
"""
Implements Imbu's web API.
"""
import simplejson as json
import logging
import os
import pkg_resources
import web

from htmresearch.frameworks.nlp.imbu import ImbuModels
from htmresearch.frameworks.nlp.model_factory import ClassificationModelTypes



g_log = logging.getLogger(__name__)



# No training in Imbu web app, user must specify loadPath
if "IMBU_LOAD_PATH_PREFIX" in os.environ:
  _IMBU_LOAD_PATH_PREFIX = os.environ["IMBU_LOAD_PATH_PREFIX"]
else:
  raise KeyError("Required IMBU_LOAD_PATH_PREFIX missing from environment")

g_imbus = {}  # Global ImbuModels cache
g_models = {}  # Global NLP model cache
for datasetName in os.listdir(_IMBU_LOAD_PATH_PREFIX):
  datasetPath = os.path.join(_IMBU_LOAD_PATH_PREFIX, datasetName)
  if os.path.isdir(datasetPath):
    # Create an imbu instance for each dataset
    imbu = ImbuModels(
      cacheRoot=os.environ.get("MODEL_CACHE_DIR", os.getcwd()),
      modelSimilarityMetric=None,
      # dataPath=os.environ.get("IMBU_DATA", pkg_resources.resource_filename(__name__, "data.csv")),
      # dataPath=os.path.join(datasetName, "data.csv"),
      dataPath=os.path.join(datasetPath, "data.csv"),
      retina=os.environ["IMBU_RETINA_ID"],
      apiKey=os.environ["CORTICAL_API_KEY"]
    )
    g_imbus.update(((datasetName, imbu),))
    # Init the dict for this dataset's models
    g_models[datasetName] = {}


def addStandardHeaders(contentType="application/json; charset=UTF-8"):
  """
  Add Standard HTTP Headers ("Content-Type", "Server") to the response.
  Here is an example of the headers added by this method using the default
  values::
      Content-Type: application/json; charset=UTF-8
      Server: Imbu x.y.z
  :param content_type: The value for the "Content-Type" header.
                       (default "application/json; charset=UTF-8")
  """
  web.header("Server", "Imbu 1.0.0", True)
  web.header("Content-Type", contentType, True)



def addCORSHeaders():
  """
  Add CORS (http://www.w3.org/TR/cors/) headers
  """
  web.header("Access-Control-Allow-Origin", "*", True)
  web.header("Access-Control-Allow-Headers",
             "accept, access-control-allow-origin, content-type", True)
  web.header("Access-Control-Allow-Credentials", "true", True)
  web.header("Access-Control-Allow-Methods", "POST", True)



class FluentWrapper(object):

  def query(self, dataset, model, text):
    """
    Queries the model (which is specific to this dataset) and returns an ordered
        list of matching samples.
    :param str dataset: Dataset name, specifying the ImbuModels instance to use.
        Possible values correspond to data dirs in _IMBU_LOAD_PATH_PREFIX.
    :param str model: Name of the model to use. Possible values are mapped to
        classes in the NLP model factory.
    :param str text: The text to match.
    :returns: a sequence of matching samples.
    ::
    [
        {"0": {"text": "sampleText", "scores": [0.75, ...]},
        ...
    ]
    """
    global g_imbus
    global g_models

    if model not in g_models[dataset]:
      loadPath = os.path.join(_IMBU_LOAD_PATH_PREFIX, dataset, model)
      g_models[dataset][model] = g_imbus[dataset].createModel(
        model, str(loadPath), None)

    if text:
      _, sortedIds, sortedDistances = g_imbus[dataset].query(
        g_models[dataset][model], text)
      return g_imbus[dataset].formatResults(model, text, sortedDistances, sortedIds)

    else:
      return []



class DefaultHandler(object):
  def GET(self, *args):  # pylint: disable=R0201,C0103
    addStandardHeaders("text/html; charset=UTF-8")
    return "<html><body><h1>Welcome to Nupic Fluent</h1></body></html>"



class FluentAPIHandler(object):
  """Handles API requests"""

  def OPTIONS(self, modelName=ImbuModels.defaultModelType): # pylint: disable=R0201,C0103
    addStandardHeaders()
    addCORSHeaders()


  def GET(self, *args):
    """ GET global ready status.  Returns "true" when all models have been
    created and are ready for queries.
    """
    addStandardHeaders()
    addCORSHeaders()

    return json.dumps(True)


  def POST(self,
           modelName=ImbuModels.defaultModelType,
           dataset=ImbuModels.defaultDataset): # pylint: disable=R0201,C0103
    addStandardHeaders()
    addCORSHeaders()

    response = {}

    data = web.data()

    if data:
      if isinstance(data, basestring):
        response = g_fluent.query(dataset, modelName, data)
      else:
        raise web.badrequest("Invalid Data. Query data must be a string")

    if len(response) == 0:
      # No data, just return all samples
      # See "ImbuModels.formatResults" for expected format
      for item in g_imbus[dataset].dataDict.items():
        response[item[0]] = {"text": item[1][0], "scores": [0]}

    return json.dumps(response)



urls = (
  "", "DefaultHandler",
  "/", "DefaultHandler",
  "/fluent", "FluentAPIHandler",
  "/fluent/(.*)", "FluentAPIHandler",
  "/fluent/(.*)/(.*)", "FluentAPIHandler"
)
app = web.application(urls, globals())

# Create Imbu model runner
g_fluent = FluentWrapper()

# Required by uWSGI per WSGI spec
application = app.wsgifunc()
