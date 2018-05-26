# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017-2018, Numenta, Inc.  Unless you have an agreement
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
Connect the Path Integration Union Narrowing experiment to the
PathIntegrationUnionNarrowing.js visualization.
"""

from __future__ import print_function
from collections import defaultdict
import json
import os
from pkg_resources import resource_string
import StringIO

import numpy as np

from htmresearch.frameworks.location.path_integration_union_narrowing import (
  PIUNExperimentMonitor)



class PIUNLogger(PIUNExperimentMonitor):
  """
  Logs the state of the world and the state of each layer to a file.
  """

  def __init__(self, out, exp, includeSynapses=True):
    self.exp = exp
    self.out = out
    self.includeSynapses = includeSynapses

    self.locationRepresentations = exp.locationRepresentations
    self.inputRepresentations = exp.inputRepresentations

    self.locationModules = exp.column.L6aModules
    self.inputLayer = exp.column.L4

    self.subscriberToken = exp.addMonitor(self)

    print(json.dumps({"numMinicolumns": exp.column.L4.numberOfColumns(),
                      "cellsPerColumn": exp.column.L4.getCellsPerColumn()}),
          file=self.out)

    print(json.dumps([{"cellDimensions": module.cellDimensions.tolist(),
                       "moduleMapDimensions": module.moduleMapDimensions.tolist(),
                       "orientation": module.orientation}
                      for module in self.locationModules]),
          file=self.out)

    print("learnedObjects", file=self.out)
    print(json.dumps(exp.learnedObjects), file=self.out)


  def __enter__(self, *args):
    pass


  def __exit__(self, *args):
    self.unsubscribe()


  def unsubscribe(self):

    self.exp.removeMonitor(self.subscriberToken)
    self.subscriberToken = None


  def beforeSense(self, featureSDR):
    print("featureInput", file=self.out)
    print(json.dumps(featureSDR.tolist()), file=self.out)
    print(json.dumps(
      [k
       for k, sdr in self.exp.features.iteritems()
       if np.intersect1d(featureSDR, sdr).size == sdr.size]), file=self.out)


  def afterLocationInitialize(self):
    print("initialSensation", file=self.out)


  def afterReset(self):
    print("reset", file=self.out)


  def beforeSensoryRepetition(self):
    print("sensoryRepetition", file=self.out)


  def beforeInferObject(self, obj):
    print("currentObject", file=self.out)
    print(json.dumps(obj), file=self.out)


  def afterLocationChanged(self, locationOnObject):
    print("locationOnObject", file=self.out)
    print(json.dumps(locationOnObject), file=self.out)


  def afterLocationShift(self, displacement, **kwargs):
    print("shift", file=self.out)
    print(json.dumps({"top": displacement[0], "left": displacement[1]}),
                     file=self.out)
    phaseDisplacementByModule = [module.phaseDisplacement.tolist()
                                 for module in self.locationModules]
    print(json.dumps(phaseDisplacementByModule), file=self.out)

    cellsByModule = [module.getActiveCells().tolist()
                     for module in self.locationModules]
    print(json.dumps(cellsByModule), file=self.out)

    cellPointsByModule = []
    for module in self.locationModules:
      cellPoints = module.activePhases * module.cellDimensions
      cellPointsByModule.append(cellPoints.tolist())
    print(json.dumps(cellPointsByModule), file=self.out)

    activeLocationCells = self.exp.column.getLocationRepresentation()

    decodings = []
    for (objectName, iFeature), sdrs in self.locationRepresentations.iteritems():
      amountContained = np.amax([(np.intersect1d(sdr, activeLocationCells).size /
                         float(sdr.size)) for sdr in sdrs])
      decodings.append([objectName, iFeature, amountContained])

    print(json.dumps(decodings), file=self.out)


  def afterLocationAnchor(self, anchorInput, **kwargs):
    print("locationLayer", file=self.out)

    cellsByModule = []
    for module in self.locationModules:
      activeCells = module.getActiveCells()

      if self.includeSynapses:
        segmentsForActiveCellsDict = defaultdict(list)

        activeSegments = module.connections.filterSegmentsByCell(
          module.activeSegments, activeCells)
        cellForActiveSegments = (
          module.connections.mapSegmentsToCells(activeSegments))

        for i, segment in enumerate(activeSegments):
          connectedSynapses = np.where(
            module.connections.matrix.getRow(segment)
            >= module.connectedPermanence)[0]

          activeSynapses = np.intersect1d(connectedSynapses, anchorInput)
          segmentsForActiveCellsDict[cellForActiveSegments[i]].append(
            activeSynapses.tolist())

        segmentsForActiveCells = [segmentsForActiveCellsDict[cell]
                                  for cell in activeCells]

        cellsByModule.append([activeCells.tolist(),
                              {"inputLayer": segmentsForActiveCells}])
      else:
        cellsByModule.append([activeCells.tolist()])

    print(json.dumps(cellsByModule), file=self.out)

    cellPointsByModule = []
    for module in self.locationModules:
      cellPoints = module.activePhases * module.cellDimensions
      cellPointsByModule.append(cellPoints.tolist())
    print(json.dumps(cellPointsByModule), file=self.out)


    activeLocationCells = self.exp.column.getLocationRepresentation()

    decodings = []
    for (objectName, iFeature), sdrs in self.locationRepresentations.iteritems():
      amountContained = np.amax([(np.intersect1d(sdr, activeLocationCells).size /
                         float(sdr.size)) for sdr in sdrs])
      decodings.append(
        [objectName, iFeature, amountContained])
    print(json.dumps(decodings), file=self.out)


  def getInputSegments(self, cells, basalInput, apicalInput):
    basalSegmentsForCellDict = defaultdict(list)

    basalSegments = self.inputLayer.basalConnections.filterSegmentsByCell(
      self.inputLayer.activeBasalSegments, cells)
    cellForBasalSegment = self.inputLayer.basalConnections.mapSegmentsToCells(
      basalSegments)

    for i, segment in enumerate(basalSegments):
      connectedSynapses = np.where(
        self.inputLayer.basalConnections.matrix.getRow(
          segment) >= self.inputLayer.connectedPermanence)[0]

      activeSynapses = np.intersect1d(connectedSynapses, basalInput)
      basalSegmentsForCellDict[cellForBasalSegment[i]].append(
        activeSynapses.tolist())

    apicalSegmentsForCellDict = defaultdict(list)

    apicalSegments = self.inputLayer.apicalConnections.filterSegmentsByCell(
      self.inputLayer.activeApicalSegments, cells)
    cellForApicalSegment = (
      self.inputLayer.apicalConnections.mapSegmentsToCells(apicalSegments))

    for i, segment in enumerate(apicalSegments):
      connectedSynapses = np.where(
        self.inputLayer.apicalConnections.matrix.getRow(segment)
        >= self.inputLayer.connectedPermanence)[0]

      activeSynapses = np.intersect1d(connectedSynapses, apicalInput)
      apicalSegmentsForCellDict[cellForApicalSegment[i]].append(
        activeSynapses.tolist())

    return {
      "locationLayer": [basalSegmentsForCellDict[cell] for cell in cells]
    }


  def getInputDecodings(self, activeCells):
    decodings = []
    for (objectName, iFeature, featureName), sdr in self.inputRepresentations.iteritems():
      amountContained = (np.intersect1d(sdr, activeCells).size /
                         float(sdr.size))
      decodings.append([objectName, iFeature, amountContained])

    return decodings


  def afterInputCompute(self, activeColumns, basalInput, **kwargs):
    activeCells = self.inputLayer.getActiveCells().tolist()
    predictedCells = self.inputLayer.getPredictedCells().tolist()

    print("predictedFeatureLocationPair", file=self.out)
    if self.includeSynapses:
      segmentsForPredictedCells = self.getInputSegments(predictedCells, basalInput,
                                                        [])
      print(json.dumps(
        [predictedCells, segmentsForPredictedCells]), file=self.out)
    else:
      print(json.dumps(
        [predictedCells]), file=self.out)
    print(json.dumps(self.getInputDecodings(activeCells)), file=self.out)

    print("featureLocationPair", file=self.out)
    if self.includeSynapses:
      segmentsForActiveCells = self.getInputSegments(activeCells, basalInput,
                                                     [])

      print(json.dumps(
        [activeCells, segmentsForActiveCells]), file=self.out)

    else:
      print(json.dumps(
        [activeCells]), file=self.out)
    print(json.dumps(self.getInputDecodings(activeCells)), file=self.out)



class PIUNVisualizer(PIUNLogger):
  """
  Creates a self-contained interactive HTML file.
  """
  def __init__(self, out, *args, **kwargs):
    self.htmlOut = out
    self.logOut = StringIO.StringIO()
    super(PIUNVisualizer, self).__init__(self.logOut, *args, **kwargs)


  def __exit__(self, *args):
    super(PIUNVisualizer, self).__exit__(*args)

    self.close()


  def close(self):
    cssText = """
        .noselect {
            -webkit-touch-callout: none;
            -webkit-user-select: none;
            -moz-user-select: none;
            -ms-user-select: none;
            user-select: none;
        }
    """

    jsText = get_htmresearchviz0_js()

    logText = self.logOut.getvalue()

    page_content = u"""
    <!doctype html>
    <html>
    <head>
    <style>
    {}
    </style>
    <script>
    {}
    </script>
    </head>
    <body>
    <div id="putItHere"></div>
    <script>
      htmresearchviz0.pathIntegrationUnionNarrowing.printRecording(document.getElementById('putItHere'), '{}');
    </script>
    </body>
    </html>
    """.format(cssText,
               jsText,
               logText.replace("\r", "\\r").replace("\n", "\\n"))

    print(page_content, file=self.htmlOut)


def get_htmresearchviz0_js():
    path = os.path.join('package_data', 'htmresearchviz0-bundle.js')
    try:
      htmresearchviz0_js = resource_string('htmresearchviz0', path).decode('utf-8')
    except ImportError:
      print("===========")
      print("Error: You need to install the htmresearchviz0 package. "
            "Follow the instructions in htmresearch/projects/location_layer/visualizations/README.md")
      print("===========")
      raise
    return htmresearchviz0_js
