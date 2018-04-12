# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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
Connect the Grid2DLocationExperiment to the locationModuleInference.js
visualization.
"""

from __future__ import print_function
from collections import defaultdict
import json

import numpy as np

from grid_2d_location_experiment import Grid2DLocationExperimentMonitor



class Grid2DLocationExperimentVisualizer(Grid2DLocationExperimentMonitor):
  """
  Logs the state of the world and the state of each layer to a file.
  """

  def __init__(self, exp, out, includeSynapses=True):
    self.exp = exp
    self.out = out
    self.includeSynapses = includeSynapses

    self.locationRepresentations = exp.locationRepresentations
    self.inputRepresentations = exp.inputRepresentations
    self.objectRepresentations = exp.objectRepresentations

    self.locationModules = exp.locationModules
    self.inputLayer = exp.inputLayer
    self.objectLayer = exp.objectLayer

    self.subscriberToken = exp.addMonitor(self)

    # World dimensions
    print(json.dumps({"width": exp.worldDimensions[1],
                      "height": exp.worldDimensions[0]}),
          file=self.out)

    print(json.dumps({"A": "red",
                      "B": "blue",
                      "C": "gray"}),
          file=self.out)
    print(json.dumps(exp.objects), file=self.out)

    print(json.dumps([{"cellDimensions": module.cellDimensions.tolist(),
                       "moduleMapDimensions": module.moduleMapDimensions.tolist(),
                       "orientation": module.orientation}
                      for module in exp.locationModules]),
          file=self.out)

    print("objectPlacements", file=self.out)
    print(json.dumps(exp.objectPlacements), file=self.out)


  def __enter__(self, *args):
    pass


  def __exit__(self, *args):
    self.unsubscribe()


  def unsubscribe(self):

    self.exp.removeMonitor(self.subscriberToken)
    self.subscriberToken = None


  def beforeSense(self, featureSDR):
    print("sense", file=self.out)
    print(json.dumps(featureSDR.tolist()), file=self.out)
    print(json.dumps(
      [k
       for k, sdr in self.exp.features.iteritems()
       if np.intersect1d(featureSDR, sdr).size == sdr.size]), file=self.out)


  def beforeMove(self, deltaLocation):
    print("move", file=self.out)
    print(json.dumps(list(deltaLocation)), file=self.out)


  def afterReset(self):
    print("reset", file=self.out)


  def markSensoryRepetition(self):
    print("sensoryRepetition", file=self.out)


  def afterWorldLocationChanged(self, locationInWorld):
    print("locationInWorld", file=self.out)
    print(json.dumps(locationInWorld), file=self.out)


  def afterLocationShift(self, displacement):
    print("shift", file=self.out)

    cellsByModule = [module.getActiveCells().tolist()
                     for module in self.locationModules]
    print(json.dumps(cellsByModule), file=self.out)

    phasesByModule = []
    for module in self.locationModules:
      phasesByModule.append(module.activePhases.tolist())
    print(json.dumps(phasesByModule), file=self.out)

    activeLocationCells = self.exp.getActiveLocationCells()

    decodings = []
    for (objectName, location), sdr in self.locationRepresentations.iteritems():
      amountContained = (np.intersect1d(sdr, activeLocationCells).size /
                         float(sdr.size))
      decodings.append(
        [objectName, location[0], location[1], amountContained])

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

    phasesByModule = []
    for module in self.locationModules:
      phasesByModule.append(module.activePhases.tolist())
    print(json.dumps(phasesByModule), file=self.out)


    activeLocationCells = self.exp.getActiveLocationCells()

    decodings = []
    for (objectName, location), sdr in self.locationRepresentations.iteritems():
      amountContained = (np.intersect1d(sdr, activeLocationCells).size /
                         float(sdr.size))
      decodings.append(
        [objectName, location[0], location[1], amountContained])
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
      "locationLayer": [basalSegmentsForCellDict[cell] for cell in cells],
      "objectLayer": [apicalSegmentsForCellDict[cell] for cell in cells],
    }


  def getInputDecodings(self, activeCells):
    decodings = []
    for (objectName, location, feature), sdr in self.inputRepresentations.iteritems():
      amountContained = (np.intersect1d(sdr, activeCells).size /
                         float(sdr.size))
      decodings.append([objectName, location[0], location[1], amountContained])

    return decodings


  def afterInputCompute(self, activeColumns, basalInput, apicalInput, **kwargs):
    activeCells = self.inputLayer.getActiveCells().tolist()
    predictedCells = self.inputLayer.getPredictedCells().tolist()

    print("inputLayer", file=self.out)

    if self.includeSynapses:
      segmentsForActiveCells = self.getInputSegments(activeCells, basalInput,
                                                     apicalInput)
      segmentsForPredictedCells = self.getInputSegments(predictedCells, basalInput,
                                                        apicalInput)

      print(json.dumps(
        [activeCells, predictedCells, segmentsForActiveCells,
         segmentsForPredictedCells]), file=self.out)

    else:
      print(json.dumps(
        [activeCells, predictedCells]), file=self.out)

    print(json.dumps({
      "activeCellDecodings": self.getInputDecodings(activeCells),
      "predictedCellDecodings": self.getInputDecodings(predictedCells)
    }), file=self.out)


  def afterObjectCompute(self, feedforwardInput, **kwargs):
    activeCells = self.objectLayer.getActiveCells()

    print("objectLayer", file=self.out)
    if self.includeSynapses:
      segmentsForActiveCells = [[] for _ in activeCells]

      for i, cell in enumerate(activeCells):
        connectedSynapses = np.where(
          self.objectLayer.proximalPermanences.getRow(cell)
          >= self.objectLayer.connectedPermanenceProximal)[0]

        activeSynapses = np.intersect1d(connectedSynapses, feedforwardInput)
        segmentsForActiveCells[i].append(activeSynapses.tolist())

      print(json.dumps([activeCells.tolist(),
                        {"inputLayer": segmentsForActiveCells}]),
            file=self.out)
    else:
      print(json.dumps([activeCells.tolist()]), file=self.out)

    decodings = [k
                 for k, sdr in self.objectRepresentations.iteritems()
                 if np.intersect1d(activeCells, sdr).size == sdr.size]
    print(json.dumps(decodings), file=self.out)
