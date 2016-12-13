#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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
Method to register regions in the nupic.research repository. Regions
specified here must be registered before NuPIC's Network API can locate them.
"""
from nupic.engine import Network
from nupic.engine import pyRegions

# The default NuPIC regions
_PY_REGIONS = [r[1] for r in pyRegions]

def registerAllResearchRegions():
  """
  Register all known research regions.
  """
  # TODO: add self discovery of all regions in nupic.research.regions

  for regionName in ["LanguageSensor", "TemporalPoolerRegion",
                     "TMRegion", "RawSensor", "ColumnPoolerRegion",
                     "ExtendedTMRegion", "CoordinateSensorRegion"]:
    registerResearchRegion(regionName)


def registerResearchRegion(regionTypeName, moduleName=None):
  """
  Register this region so that NuPIC can later find it.

  @param regionTypeName: (str) type name of the region. E.g LanguageSensor.
  @param moduleName: (str) location of the region class, only needed if
    registering a region that is outside the expected "regions/" dir.
  """
  global _PY_REGIONS

  if moduleName is None:
    # the region is located in the regions/ directory
    moduleName = "htmresearch.regions." + regionTypeName
  if regionTypeName not in _PY_REGIONS:
    # Add new region class to the network.
    module = __import__(moduleName, {}, {}, regionTypeName)
    unregisteredClass = getattr(module, regionTypeName)
    Network.registerRegion(unregisteredClass)
    # Add region to list of registered PyRegions
    _PY_REGIONS.append(regionTypeName)


