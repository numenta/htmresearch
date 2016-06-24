#!/usr/bin/env python
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

import unittest

from htmresearch.algorithms.extended_temporal_memory import ExtendedTemporalMemory
from extensive_tm_test_base import ExtensiveTemporalMemoryTest


class ExtensiveExtendedTemporalMemoryTest(ExtensiveTemporalMemoryTest, unittest.TestCase):
  """
  Subclasses the tests for Temporal Memory to port them to Extended Temporal Memory.
  Tests specific to ETM should also be implemented in this class.
  """

  def getTMClass(self):
    return ExtendedTemporalMemory


  def init(self, overrides=None):
    """
    Overrides the base method to add the learnOnOneCell parameter for ETM.
    :param overrides: dict of parameters to pass to constructor
    """
    if overrides is None:
      overrides = {}
    if 'learnOnOneCell' not in overrides:
      # if not specified, behave like regular TM
      overrides["learnOnOneCell"] = False
    super(ExtensiveExtendedTemporalMemoryTest, self).init(overrides)
