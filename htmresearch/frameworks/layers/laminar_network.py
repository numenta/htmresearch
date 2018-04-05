# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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
Overall factory methods to create networks of multiple layers and for
experimenting with different laminar structures.

There are two main types of networks:

  L2L4 networks, and,
  L2456 networks.

Either type can be created as a single column or with multiple columns. Each
type has its own creation file (imported here) - see that file for detailed
descriptions.
"""
from nupic.engine import Network
from htmresearch.frameworks.layers.l2_l4_network_creation import (
  createL4L2Column, createMultipleL4L2Columns,
  createMultipleL4L2ColumnsWithTopology)
from htmresearch.frameworks.layers.l2456_network_creation import (
  createL2456Columns
)
from htmresearch.frameworks.layers.combined_sequence_network_creation import (
  createL4L2TMColumn
)
from htmresearch.frameworks.layers.combined_sequence_network_creation2 import (
  createCombinedSequenceColumn
)
from htmresearch.support.register_regions import registerAllResearchRegions


def createNetwork(networkConfig):
  """
  Create and initialize the specified network instance.

  @param networkConfig: (dict) the configuration of this network.
  @return network: (Network) The actual network
  """

  registerAllResearchRegions()

  network = Network()

  if networkConfig["networkType"] == "L4L2Column":
    return createL4L2Column(network, networkConfig, "_0")
  elif networkConfig["networkType"] == "MultipleL4L2Columns":
    return createMultipleL4L2Columns(network, networkConfig)
  elif networkConfig["networkType"] == "MultipleL4L2ColumnsWithTopology":
    return createMultipleL4L2ColumnsWithTopology(network, networkConfig)
  elif networkConfig["networkType"] == "L2456Columns":
    return createL2456Columns(network, networkConfig)
  elif networkConfig["networkType"] == "L4L2TMColumn":
    return createL4L2TMColumn(network, networkConfig, "_0")
  elif networkConfig["networkType"] == "CombinedSequenceColumn":
    return createCombinedSequenceColumn(network, networkConfig, "_0")



def printNetwork(network):
  """
  Given a network, print out regions sorted by phase
  """
  print "The network has",len(network.regions.values()),"regions"
  for p in range(network.getMaxPhase()):
    print "=== Phase",p
    for region in network.regions.values():
      if network.getPhases(region.name)[0] == p:
        print "   ",region.name
