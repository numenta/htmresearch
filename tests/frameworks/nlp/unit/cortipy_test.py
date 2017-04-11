#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import os
import unittest

from mock import patch

from cortipy.cortical_client import CorticalClient



class TestCortipy(unittest.TestCase):

  def testAPIKeyPresent(self):
    with patch.dict("os.environ", {"CIO_API_KEY": "apikey123"}):
      cClient = CorticalClient()

  @patch("os.environ")
  def testExceptionIfAPIKeyNotPresent(self, mockOS):
    with self.assertRaises(Exception) as e:
      cClient = CorticalClient()
    self.assertIn("Missing API key.", e.exception)


if __name__ == "__main__":
  unittest.main()
