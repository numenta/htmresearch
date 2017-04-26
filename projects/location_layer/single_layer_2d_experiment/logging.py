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

from runner import SingleLayer2DExperimentMonitor



class SingleLayer2DExperimentVisualizer(SingleLayer2DExperimentMonitor):
  """
  Logs the state of the world and the state of each layer to a CSV.
  """

  def __init__(self, exp, csvOut):
    self.exp = exp
    self.csvOut = csvOut

    self.locationRepresentations = exp.locations
    self.inputRepresentations = exp.inputRepresentations
    self.objectRepresentations = exp.objectRepresentations

    self.locationLayer = exp.locationLayer
    self.inputLayer = exp.inputLayer
    self.objectLayer = exp.objectLayer

    self.subscriberToken = exp.addMonitor(self)

    # Make it compatible with JSON -- can only use strings as dict keys.
    objects = dict((objectName, featureLocationPairs.items())
                   for objectName, featureLocationPairs in exp.objects.iteritems())

    self.csvOut.writerow((exp.diameter,))
    self.csvOut.writerow((json.dumps({"A": "red",
                                      "B": "blue",
                                      "C": "gray"}),))
    self.csvOut.writerow((json.dumps(objects),))

    self.prevActiveLocationCells = ()


  def __enter__(self, *args):
    pass


  def __exit__(self, *args):
    self.unsubscribe()


  def unsubscribe(self):

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
    self.prevActiveLocationCells = ()


  def afterPlaceObjects(self, objectPlacements):
    self.csvOut.writerow(('objectPlacements',))
    self.csvOut.writerow([json.dumps(objectPlacements)])


  def afterLocationCompute(self, deltaLocation, newLocation,
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
                                              self.prevActiveLocationCells)
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

    self.prevActiveLocationCells = activeCells


  def afterInputCompute(self, activeColumns, basalInput, apicalInput,
                        basalGrowthCandidates=None, apicalGrowthCandidates=None,
                        learn=True):
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


  def afterObjectCompute(self, feedforwardInput, lateralInputs=(),
                         feedforwardGrowthCandidates=None, learn=True):
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
