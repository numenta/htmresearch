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
Connect the MultiColumn2DExperiment to the locationModuleInference.js
visualization.
"""

from __future__ import print_function
from collections import defaultdict
import json

import numpy as np

from grid_multi_column_experiment import MultiColumn2DExperimentMonitor


class MultiColumn2DExperimentVisualizer(MultiColumn2DExperimentMonitor):
  """
  Logs the state of the world and the state of each layer to a file.
  """

  def __init__(self, exp, out, includeSynapses=True):
    self.exp = exp
    self.out = ControlledFlushStream(out)
    self.includeSynapses = includeSynapses

    self.subscriberToken = exp.addMonitor(self)

    print(exp.numCorticalColumns, file=self.out)

    # World dimensions
    print(json.dumps({"width": exp.worldDimensions[1],
                      "height": exp.worldDimensions[0]}),
          file=self.out)

    print(json.dumps({"A": "red",
                      "B": "blue",
                      "C": "gray"}),
          file=self.out)
    print(json.dumps(exp.objects), file=self.out)

    print(json.dumps(exp.locationConfigs), file=self.out)

    print("objectPlacements", file=self.out)
    print(json.dumps(exp.objectPlacements), file=self.out)


  def __enter__(self, *args):
    pass


  def __exit__(self, *args):
    self.out.flush()
    self.unsubscribe()


  def unsubscribe(self):
    self.exp.removeMonitor(self.subscriberToken)
    self.subscriberToken = None


  def beforeCompute(self, egocentricLocationByColumn, featureSDRByColumn,
                    isRepeat):
    print("compute", file=self.out)
    print("repeat" if isRepeat else "sense", file=self.out)
    print(json.dumps(egocentricLocationByColumn.tolist()), file=self.out)
    print(json.dumps([featureSDR.tolist()
                      for featureSDR in featureSDRByColumn]),
          file=self.out)
    print(json.dumps([[k
                       for (iCol2, k), sdr in self.exp.features.iteritems()
                       if iCol == iCol2
                       if np.intersect1d(featureSDR, sdr,
                                         assume_unique=True).size == sdr.size]
                      for iCol, featureSDR in enumerate(featureSDRByColumn)]),
          file=self.out)

    cellsByModule = [module.getActiveCells().tolist()
                     for module in self.exp.bodyToSpecificObjectModules]

    # Keep these so that we can use them when checking which internal synapses
    # are active.
    self.prevActiveObjectCellsByColumn = [
      c.objectLayer.getActiveCells()
      for c in self.exp.corticalColumns]


  def afterReset(self):
    print("reset", file=self.out)


  def afterBodyWorldLocationChanged(self, bodyWorldLocation):
    print("bodyLocationInWorld", file=self.out)
    print(json.dumps(bodyWorldLocation), file=self.out)


  def afterSensorWorldLocationChanged(self, worldLocationByColumn):
    print("locationInWorld", file=self.out)
    print(json.dumps(worldLocationByColumn.tolist()), file=self.out)


  def afterSensorToBodyCompute(self, paramsByColumn):
    cellsByModuleByColumn = [
      [module.getActiveCells().tolist()
       for module in c.sensorToBodyModules]
      for c in self.exp.corticalColumns]

    print("sensorToBody", file=self.out)
    print(json.dumps(cellsByModuleByColumn), file=self.out)


  def afterSensorMetricCompute(self, paramsByModuleByColumn):
    cellsByModuleByColumn = []

    for iCol, c in enumerate(self.exp.corticalColumns):
      cellsByModule = []
      for iModule, module in enumerate(c.sensorToSpecificObjectModules):
        params = paramsByModuleByColumn[iCol][iModule]
        activeCells = module.getActiveCells()

        if self.includeSynapses:
          cellsByModule.append(
            [activeCells.tolist(),
             {
               "{} sensorToBody".format(iCol):
               _getActiveSynapsesOnActiveSegments(
                 module.metricConnections.connectionsBySource[
                   "sensorToBody"],
                 activeCells,
                 module.activeMetricSegments,
                 params["sensorToBody"],
                 module.connectedPermanence,
                 offset=iModule * module.cellCount),

               "bodyToSpecificObject":
               _getActiveSynapsesOnActiveSegments(
                 module.metricConnections.connectionsBySource[
                   "bodyToSpecificObject"],
                 activeCells,
                 module.activeMetricSegments,
                 params["bodyToSpecificObject"],
                 module.connectedPermanence,
                 offset=iModule * module.cellCount),
             }])
        else:
          cellsByModule.append([activeCells.tolist()])

      cellsByModuleByColumn.append(cellsByModule)

    decodingsByColumn = []
    for c in self.exp.corticalColumns:
      decodings = []
      activeLocationCells = c.getSensorToSpecificObjectSDR()

      for ((objectName, location),
           sdr) in c.sensorLocationRepresentations.iteritems():
        amountContained = (np.intersect1d(sdr, activeLocationCells,
                                          assume_unique=True).size /
                           float(sdr.size))
        decodings.append(
          [objectName, location[0], location[1], amountContained])

      decodingsByColumn.append(decodings)

    print("predictedSensorToSpecificObject", file=self.out)
    print(json.dumps(cellsByModuleByColumn), file=self.out)
    print(json.dumps(decodingsByColumn), file=self.out)


  def afterSensorLocationAnchor(self, paramsByColumn):
    cellsByModuleByColumn = []
    for iCol, c in enumerate(self.exp.corticalColumns):
      cellsByModule = []
      for module in c.sensorToSpecificObjectModules:
        activeCells = module.getActiveCells()

        if self.includeSynapses:
          synapsesForActiveCells = _getActiveSynapsesOnActiveSegments(
            module.anchorConnections,
            activeCells,
            module.activeSegments,
            paramsByColumn[iCol]["anchorInput"],
            module.connectedPermanence)

          cellsByModule.append(
            [activeCells.tolist(),
             {"{} inputLayer".format(iCol): synapsesForActiveCells}])
        else:
          cellsByModule.append([activeCells.tolist()])
      cellsByModuleByColumn.append(cellsByModule)

    print("anchoredSensorToSpecificObject", file=self.out)
    print(json.dumps(cellsByModuleByColumn), file=self.out)


    decodingsByColumn = []
    for c in self.exp.corticalColumns:
      decodings = []
      activeLocationCells = c.getSensorToSpecificObjectSDR()

      for ((objectName, location),
           sdr) in c.sensorLocationRepresentations.iteritems():
        amountContained = (np.intersect1d(sdr, activeLocationCells,
                                          assume_unique=True).size /
                           float(sdr.size))
        decodings.append(
          [objectName, location[0], location[1], amountContained])

      decodingsByColumn.append(decodings)

    print(json.dumps(decodingsByColumn), file=self.out)


  def afterBodyLocationAnchor(self, paramsByModule):
    cellsByModule = []
    for iModule, module in enumerate(self.exp.bodyToSpecificObjectModules):
      params = paramsByModule[iModule]
      activeCells = module.getActiveCells()

      if self.includeSynapses:
        synapsesForActiveCellsBySourceLayer = {}

        for (iPresynapticCol,
             metricConnections) in enumerate(module.connectionsByColumn):
          synapsesForActiveCellsBySourceLayer[
            "{} sensorToBody".format(iPresynapticCol)] = (
              _getActiveSynapsesOnActiveSegments(
                metricConnections.connectionsBySource["sensorToBody"],
                activeCells,
                module.activeSegmentsByColumn[iPresynapticCol],
                params["sensorToBodyByColumn"][iPresynapticCol],
                module.connectedPermanence,
                offset=iModule * module.cellCount))

          synapsesForActiveCellsBySourceLayer[
            "{} sensorToSpecificObject".format(iPresynapticCol)] = (
              _getActiveSynapsesOnActiveSegments(
                metricConnections.connectionsBySource["sensorToSpecificObject"],
                activeCells,
                module.activeSegmentsByColumn[iPresynapticCol],
                params["sensorToSpecificObjectByColumn"][iPresynapticCol],
                module.connectedPermanence,
                offset=iModule * module.cellCount))

        cellsByModule.append(
          [activeCells.tolist(), synapsesForActiveCellsBySourceLayer])

      else:
        cellsByModule.append([activeCells.tolist()])

    print("anchoredBodyToSpecificObject", file=self.out)
    print(json.dumps(cellsByModule), file=self.out)


  def _printFeatureLocationLayers(self, paramsByColumn, predicted):
    cellsByColumn = []
    decodingsByColumn = []
    for iCol, c in enumerate(self.exp.corticalColumns):
      params = paramsByColumn[iCol]

      if predicted:
        cells = c.inputLayer.getPredictedCells()
      else:
        cells = c.inputLayer.getActiveCells()

      if self.includeSynapses:
        segmentsForCells = {
          "{} sensorToSpecificObject".format(iCol):
          _getActiveSynapsesOnActiveSegments(
            c.inputLayer.basalConnections,
            cells,
            c.inputLayer.activeBasalSegments,
            params["basalInput"],
            c.inputLayer.connectedPermanence),

          "{} objectLayer".format(iCol):
          _getActiveSynapsesOnActiveSegments(
            c.inputLayer.apicalConnections,
            cells,
            c.inputLayer.activeApicalSegments,
            params["apicalInput"],
            c.inputLayer.connectedPermanence),
        }

        cellsByColumn.append([
          cells.tolist(), segmentsForCells])

      else:
        cellsByColumn.append([cells.tolist()])

      decodingsByColumn.append(_getInputDecodings(c, cells))

    if predicted:
      print("predictedFeatureLocationPair", file=self.out)
    else:
      print("featureLocationPair", file=self.out)
    print(json.dumps(cellsByColumn), file=self.out)
    print(json.dumps(decodingsByColumn), file=self.out)


  def afterInputCompute(self, paramsByColumn):
    self._printFeatureLocationLayers(paramsByColumn, predicted=True)
    self._printFeatureLocationLayers(paramsByColumn, predicted=False)


  def afterObjectCompute(self, paramsByColumn):
    cellsByColumn = []
    decodingsByColumn = []
    for iCol, c in enumerate(self.exp.corticalColumns):
      params = paramsByColumn[iCol]
      activeCells = c.objectLayer.getActiveCells()

      if self.includeSynapses:
        prevActiveCells = self.prevActiveObjectCellsByColumn[iCol]

        inputSynapsesForActiveCells = []
        objectSynapsesForActiveCellsBySourceColumn = [
          [] for _ in xrange(self.exp.numCorticalColumns)]

        for i, cell in enumerate(activeCells):
          inputSynapsesForActiveCells.append(_getActiveSynapses(
            c.objectLayer.proximalPermanences,
            cell, params["feedforwardInput"],
            c.objectLayer.connectedPermanenceProximal))

          for iPresynapticCol in xrange(self.exp.numCorticalColumns):
            if iPresynapticCol == iCol:
              synapsesOnSegment = _getActiveSynapses(
                c.objectLayer.internalDistalPermanences,
                cell, prevActiveCells,
                c.objectLayer.connectedPermanenceDistal)
            else:
              if iPresynapticCol < iCol:
                matrixIndex = iPresynapticCol
              else:
                matrixIndex = iPresynapticCol - 1

              synapsesOnSegment = _getActiveSynapses(
                c.objectLayer.distalPermanences[matrixIndex],
                cell, params["lateralInputs"][matrixIndex],
                c.objectLayer.connectedPermanenceDistal)

            objectSynapsesForActiveCellsBySourceColumn[iPresynapticCol].append(
              synapsesOnSegment)

        synapsesForActiveCellsBySourceLayer = dict(
          ("{} objectLayer".format(i), synapses)
          for i, synapses in enumerate(
              objectSynapsesForActiveCellsBySourceColumn))
        synapsesForActiveCellsBySourceLayer[
          "{} inputLayer".format(iCol)] = inputSynapsesForActiveCells
        cellsByColumn.append(
          [activeCells.tolist(), synapsesForActiveCellsBySourceLayer])

      else:
        cellsByColumn.append([activeCells.tolist()])

      decodingsByColumn.append(
        [k
         for k, sdr in c.objectRepresentations.iteritems()
         if np.intersect1d(activeCells, sdr,
                           assume_unique=True).size == sdr.size])

    print("objectLayer", file=self.out)
    print(json.dumps(cellsByColumn), file=self.out)
    print(json.dumps(decodingsByColumn), file=self.out)


  def flush(self):
    self.out.flush()


  def clearUnflushedData(self):
    self.out.clear()



def _getActiveSynapsesOnActiveSegments(connections, cells, activeSegments,
                                       activeInput, connectedPermanence, offset=0):
  synapsesForCellDict = defaultdict(list)

  segments = connections.filterSegmentsByCell(activeSegments, cells)
  cellForSegment = connections.mapSegmentsToCells(segments)

  for i, segment in enumerate(segments):
    connectedSynapses = np.where(
      connections.matrix.getRow(segment) >= connectedPermanence)[0]

    activeSynapses = np.intersect1d(connectedSynapses, activeInput,
                                    assume_unique=True)
    if offset != 0:
      activeSynapses += offset

    synapsesForCellDict[cellForSegment[i]] += activeSynapses.tolist()

  return [sorted(set(synapsesForCellDict[cell]))
          for cell in cells]


def _getActiveSynapses(matrix, cell, activeInput, connectedPermanence):
  connectedSynapses = np.where(matrix.getRow(cell) >= connectedPermanence)[0]

  activeSynapses = np.intersect1d(connectedSynapses, activeInput,
                                  assume_unique=True)
  return activeSynapses.tolist()


def _getInputDecodings(c, activeCells):
  decodings = []
  for (objectName, location, feature), sdr in c.inputRepresentations.iteritems():
    amountContained = (np.intersect1d(sdr, activeCells,
                                      assume_unique=True).size /
                       float(sdr.size))
    decodings.append([objectName, location[0], location[1], amountContained])

  return decodings



class ControlledFlushStream(object):
  def __init__(self, out):
    self.out = out
    self.waiting = []

  def flush(self):
    for s in self.waiting:
      self.out.write(s)

    self.waiting = []

  def clear(self):
    self.waiting = []

  def write(self, s):
    self.waiting.append(s)

  def close(self):
    self.flush()
