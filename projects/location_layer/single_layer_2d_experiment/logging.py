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

import json

import numpy as np

from htmresearch.algorithms.apical_tiebreak_temporal_memory import (
  ApicalTiebreakTemporalMemoryMonitor)
from htmresearch.algorithms.column_pooler import ColumnPoolerMonitor
from htmresearch.algorithms.single_layer_location_memory import (
  SingleLayerLocationMemoryMonitor)

from runner import SingleLayer2DExperimentMonitor


class LocationLayerMonitor(SingleLayerLocationMemoryMonitor):
  """
  Outputs a recording of the LocationLayer to a CSV file. At each timestep, it
  logs the active cells, and it logs each active synapse on each active cell.
  """

  def __init__(self, locationLayer, csvOut, locationRepresentations):
    self.locationLayer = locationLayer
    self.csvOut = csvOut
    self.locationRepresentations = locationRepresentations
    self.subscriberToken = locationLayer.addMonitor(self)


  def __enter__(self, *args):
    pass


  def __exit__(self, *args):
    self.unsubscribe()


  def unsubscribe(self):
    self.locationLayer.removeMonitor(self.subscriberToken)
    self.subscriberToken = None


  def afterCompute(self, prevActiveCells, deltaLocation, newLocation,
                   featureLocationInput, featureLocationGrowthCandidates,
                   learn):
    activeCells = self.locationLayer.getActiveCells()

    cells = dict((cell, [])
                 for cell in activeCells.tolist())

    for cell in activeCells.tolist():
      presynapticCell = cell
      if presynapticCell in newLocation:
        segmentData = [
          ["newLocation", [cell]]
        ]
        cells[cell].append(segmentData)


    deltaSegments = self.locationLayer.deltaConnections.filterSegmentsByCell(
      self.locationLayer.activeDeltaSegments, activeCells)
    cellForDeltaSegment = self.locationLayer.deltaConnections.mapSegmentsToCells(
      deltaSegments)

    for i, segment in enumerate(deltaSegments):
      connectedDeltaSynapses = np.where(
        self.locationLayer.deltaConnections.matrix.getRow(
          segment) >= self.locationLayer.connectedPermanence)[0]
      connectedInternalSynapses = np.where(
        self.locationLayer.internalConnections.matrix.getRow(
          segment) >= self.locationLayer.connectedPermanence)[0]

      activeDeltaSynapses = np.intersect1d(connectedDeltaSynapses,
                                           deltaLocation)
      activeInternalSynapses = np.intersect1d(connectedInternalSynapses,
                                              prevActiveCells)
      segmentData = [
        ["deltaLocation", activeDeltaSynapses.tolist()],
        ["location", activeInternalSynapses.tolist()],
      ]
      cells[cellForDeltaSegment[i]].append(segmentData)


    featureLocationSegments = (
      self.locationLayer.featureLocationConnections.filterSegmentsByCell(
        self.locationLayer.activeFeatureLocationSegments, activeCells))
    cellForFeatureLocationSegment = (
      self.locationLayer.featureLocationConnections.mapSegmentsToCells(
        featureLocationSegments))

    for i, segment in enumerate(featureLocationSegments):
      connectedFeatureLocationSynapses = np.where(
        self.locationLayer.featureLocationConnections.matrix.getRow(segment)
        >= self.locationLayer.connectedPermanence)[0]

      activeFeatureLocationSynapses = np.intersect1d(
        connectedFeatureLocationSynapses, featureLocationInput)
      segmentData = [
        ["input", activeFeatureLocationSynapses.tolist()]
      ]
      cells[cellForFeatureLocationSegment[i]].append(segmentData)

    self.csvOut.writerow(("layer", "location"))
    self.csvOut.writerow([json.dumps(cells.items())])

    decodings = [k
                 for k, sdr in self.locationRepresentations.iteritems()
                 if np.intersect1d(activeCells, sdr).size == sdr.size]
    self.csvOut.writerow([json.dumps(decodings)])


  def afterReset(self):
    pass


class InputLayerMonitor(ApicalTiebreakTemporalMemoryMonitor):
  """
  Outputs a recording of the InputLayer to a CSV file. At each timestep, it
  logs the active cells, and it logs each active synapse on each active cell.
  """

  def __init__(self, inputLayer, csvOut, inputRepresentations):
    self.inputLayer = inputLayer
    self.csvOut = csvOut
    self.inputRepresentations = inputRepresentations
    self.subscriberToken = inputLayer.addMonitor(self)


  def __enter__(self, *args):
    pass


  def __exit__(self, *args):
    self.unsubscribe()


  def unsubscribe(self):
    self.inputLayer.removeMonitor(self.subscriberToken)
    self.subscriberToken = None


  def afterCompute(self, activeColumns, basalInput, apicalInput,
                   basalGrowthCandidates, apicalGrowthCandidates, learn):
    activeCells = self.inputLayer.getActiveCells()

    cells = dict((cell, [])
                 for cell in activeCells.tolist())


    for cell in activeCells.tolist():
      activeColumn = cell / self.inputLayer.cellsPerColumn
      assert activeColumn in activeColumns
      segmentData = [
        ["feature", [activeColumn]]
      ]
      cells[cell].append(segmentData)


    basalSegments = self.inputLayer.basalConnections.filterSegmentsByCell(
      self.inputLayer.activeBasalSegments, activeCells)
    cellForBasalSegment = self.inputLayer.basalConnections.mapSegmentsToCells(
      basalSegments)

    for i, segment in enumerate(basalSegments):
      connectedSynapses = np.where(
        self.inputLayer.basalConnections.matrix.getRow(
          segment) >= self.inputLayer.connectedPermanence)[0]

      activeSynapses = np.intersect1d(connectedSynapses, basalInput)
      segmentData = [
        ["location", activeSynapses.tolist()],
      ]
      cells[cellForBasalSegment[i]].append(segmentData)


    apicalSegments = self.inputLayer.apicalConnections.filterSegmentsByCell(
      self.inputLayer.activeApicalSegments, activeCells)
    cellForApicalSegment = (
      self.inputLayer.apicalConnections.mapSegmentsToCells(apicalSegments))

    for i, segment in enumerate(apicalSegments):
      connectedSynapses = np.where(
        self.inputLayer.apicalConnections.matrix.getRow(segment)
        >= self.inputLayer.connectedPermanence)[0]

      activeSynapses = np.intersect1d(connectedSynapses, apicalInput)
      segmentData = [
        ["object", activeSynapses.tolist()]
      ]
      cells[cellForApicalSegment[i]].append(segmentData)

    self.csvOut.writerow(("layer", "input"))
    self.csvOut.writerow([json.dumps(cells.items())])

    decodings = [k
                 for k, sdr in self.inputRepresentations.iteritems()
                 if np.intersect1d(activeCells, sdr).size == sdr.size]
    self.csvOut.writerow([json.dumps(decodings)])


  def afterReset(self):
    pass



class ObjectLayerMonitor(ColumnPoolerMonitor):
  """
  Outputs a recording of the ObjectLayer to a CSV file. At each timestep, it
  logs the active cells, and it logs each active feedforward synapse on each
  active cell.
  """

  def __init__(self, objectLayer, csvOut, objectRepresentations):
    self.objectLayer = objectLayer
    self.csvOut = csvOut
    self.objectRepresentations = objectRepresentations
    self.subscriberToken = objectLayer.addMonitor(self)


  def __enter__(self, *args):
    pass


  def __exit__(self, *args):
    self.unsubscribe()


  def unsubscribe(self):
    self.objectLayer.removeMonitor(self.subscriberToken)
    self.subscriberToken = None


  def afterCompute(self, feedforwardInput, lateralInputs, learn):
    activeCells = self.objectLayer.getActiveCells()

    cells = dict((cell, [])
                 for cell in activeCells.tolist())

    for cell in activeCells:
      connectedSynapses = np.where(
        self.objectLayer.proximalPermanences.getRow(cell)
        >= self.objectLayer.connectedPermanenceProximal)[0]

      activeSynapses = np.intersect1d(connectedSynapses, feedforwardInput)

      segmentData = [
        ["input", activeSynapses.tolist()]
      ]
      cells[cell].append(segmentData)

    self.csvOut.writerow(("layer", "object"))
    self.csvOut.writerow([json.dumps(cells.items())])

    decodings = [k
                 for k, sdr in self.objectRepresentations.iteritems()
                 if np.intersect1d(activeCells, sdr).size == sdr.size]
    self.csvOut.writerow([json.dumps(decodings)])


  def afterReset(self):
    pass



class SingleLayer2DExperimentVisualizer(SingleLayer2DExperimentMonitor):
  """
  Attaches monitors to each layer in the experiment, and logs the state of the
  world and the inputs to each layer to a CSV.
  """

  def __init__(self, exp, csvOut):
    self.exp = exp
    self.csvOut = csvOut

    self.locationLayerMonitor = LocationLayerMonitor(exp.locationLayer, csvOut,
                                                     exp.locations)
    self.inputLayerMonitor = InputLayerMonitor(exp.inputLayer, csvOut,
                                               exp.inputRepresentations)
    self.objectLayerMonitor = ObjectLayerMonitor(exp.objectLayer, csvOut,
                                                 exp.objectRepresentations)

    self.subscriberToken = exp.addMonitor(self)

    # Make it compatible with JSON -- can only use strings as dict keys.
    objects = dict((objectName, featureLocationPairs.items())
                   for objectName, featureLocationPairs in exp.objects.iteritems())

    self.csvOut.writerow((exp.diameter,))
    self.csvOut.writerow((json.dumps({"A": "red",
                                      "B": "blue",
                                      "C": "gray"}),))
    self.csvOut.writerow((json.dumps(objects),))


  def __enter__(self, *args):
    pass


  def __exit__(self, *args):
    self.unsubscribe()


  def unsubscribe(self):
    self.locationLayerMonitor.unsubscribe()
    self.inputLayerMonitor.unsubscribe()
    self.objectLayerMonitor.unsubscribe()

    self.exp.removeMonitor(self.subscriberToken)
    self.subscriberToken = None


  def beforeTimestep(self, locationSDR, transitionSDR, featureSDR,
                     egocentricLocation, learn):
    self.csvOut.writerow(("t",))

    self.csvOut.writerow(("input", "newLocation"))
    self.csvOut.writerow([json.dumps(locationSDR.tolist())])
    self.csvOut.writerow([json.dumps(
      [decoding
       for decoding, sdr in self.exp.locations.iteritems()
       if np.intersect1d(locationSDR, sdr).size == sdr.size])])

    self.csvOut.writerow(("input", "deltaLocation"))
    self.csvOut.writerow([json.dumps(transitionSDR.tolist())])
    self.csvOut.writerow([json.dumps(
      [decoding
       for decoding, sdr in self.exp.transitions.iteritems()
       if np.intersect1d(transitionSDR, sdr).size == sdr.size])])

    self.csvOut.writerow(("input", "feature"))
    self.csvOut.writerow([json.dumps(featureSDR.tolist())])
    self.csvOut.writerow([json.dumps(
      [k
       for k, sdr in self.exp.features.iteritems()
       if np.intersect1d(featureSDR, sdr).size == sdr.size])])

    self.csvOut.writerow(("egocentricLocation",))
    self.csvOut.writerow([json.dumps(egocentricLocation)])


  def afterReset(self):
    self.csvOut.writerow(("reset",))


  def afterPlaceObjects(self, objectPlacements):
    self.csvOut.writerow(('objectPlacements',))
    self.csvOut.writerow([json.dumps(objectPlacements)])
