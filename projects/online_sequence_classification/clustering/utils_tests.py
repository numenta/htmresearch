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

import copy
import unittest
import numpy as np
from utils import find_cluster_repetitions, find_cluster_assignments



class UtilsTest(unittest.TestCase):
  """
  Tests for utils.py
  """


  def setUp(self):
    # Sensor data params
    self.reps = 5
    self.num_categories = 3
    self.signal_amplitude = 5
    self.sequence_length = 10
    self.periods_per_sequence = 5
    self.weight_tm_active_cells = 1
    self.weight_tm_pred_active_cells = 0
    self.period_length = self.sequence_length / self.periods_per_sequence
    self.num_points = self.sequence_length * self.num_categories * self.reps

    # TM params
    self.sparsity = 0.02
    self.input_width = 2048 * 32

    # Base TM active cells for each category. 
    base_category_tm_active_cells = []
    for _ in range(self.num_categories):
      sdr = np.random.choice([0, 1],
                             size=(self.input_width,),
                             p=[1 - self.sparsity, self.sparsity])
      base_category_tm_active_cells.append(sdr)

    # Base TM active cells for each category. 
    base_category_tm_pred_active_cells = []
    for _ in range(self.num_categories):
      sdr = np.random.choice([0, 1],
                             size=(self.input_width,),
                             p=[1 - self.sparsity, self.sparsity])
      base_category_tm_pred_active_cells.append(sdr)

    self.sensor_values = []
    self.tm_active_cells = []
    self.categories = []
    for _ in range(self.reps):
      for i in range(self.num_categories):
        for j in range(self.sequence_length):
          value = ((i + 1) * (j + 1)) % self.period_length
          # Flip only two bits for SDRs in the same category, but at different
          # time in the sequence.
          value_sdr = copy.copy(base_category_tm_active_cells[i])
          value_sdr[np.where(sdr == 1)[0][value]] = 0
          value_sdr[np.where(sdr == 0)[0][value]] = 1

          self.sensor_values.append(value)
          self.tm_active_cells.append(value_sdr)
          self.categories.append(i)

    # Note: for now, clusters = categories and TM active = predictive cells
    self.tm_predictive_active_cells = self.tm_active_cells
    self.clusters = self.categories

    self.sdrs = (
      self.weight_tm_active_cells * self.tm_active_cells +
      self.weight_tm_pred_active_cells * self.tm_predictive_active_cells)


  def testInputData(self):
    """
     Make sure we have the right number of elements and that their 
     value alternates with a frequency equals to self.sequence_length.
    """
    self.assertEqual(self.num_points, len(self.sensor_values))
    self.assertEqual(self.num_points, len(self.tm_active_cells))
    self.assertEqual(self.num_points, len(self.categories))

    # Make sure the data is periodic
    periodic_indices = [
      self.period_length,
      self.period_length + (self.sequence_length * self.num_categories)
    ]
    for start_idx in range(self.period_length):
      for periodic_idx in periodic_indices:
        non_periodic_idx = periodic_idx - 1
        # Sensor values
        self.assertEqual(self.sensor_values[start_idx],
                         self.sensor_values[start_idx + periodic_idx])
        self.assertNotEqual(self.sensor_values[start_idx],
                            self.sensor_values[start_idx + non_periodic_idx])

        # Actual categories
        self.assertEqual(self.categories[start_idx],
                         self.categories[start_idx + periodic_idx])
        self.assertEqual(self.categories[start_idx],
                         self.categories[start_idx + non_periodic_idx])

        # Clusters
        self.assertEqual(self.categories[start_idx],
                         self.categories[start_idx + periodic_idx])
        self.assertEqual(self.categories[start_idx],
                         self.categories[start_idx + non_periodic_idx])

        # TM active cells
        self.assertEqual(
          list(self.sdrs[start_idx].nonzero()[0]),
          list(self.sdrs[start_idx + periodic_idx].nonzero()[0]))
        self.assertNotEqual(
          list(self.sdrs[start_idx].nonzero()[0]),
          list(self.sdrs[start_idx + non_periodic_idx].nonzero()[0]))


  def testClusterRepetitions(self):

    cluster_repetitions, sdr_clusters = find_cluster_repetitions(self.sdrs,
                                                                 self.clusters)
    self.assertEqual(len(cluster_repetitions), self.num_points)
    self.assertEqual(len(sdr_clusters), self.num_categories)

    num_points_per_category = self.sequence_length * self.reps
    for i in range(self.num_categories):
      self.assertEqual(len(sdr_clusters[i]), num_points_per_category)


  def testClustersAssignments(self):
    # Ignore noise
    ignore_noise = True
    num_categories = self.num_categories - 1
    (cluster_assignments,
     sdr_slices) = find_cluster_assignments(self.sdrs,
                                            self.clusters,
                                            ignore_noise)
    self.assertEqual(len(cluster_assignments), num_categories * self.reps)
    self.assertEqual(len(sdr_slices), num_categories * self.reps)

    # Don't ignore the noise
    ignore_noise = False
    num_categories = self.num_categories
    (cluster_assignments,
     sdr_slices) = find_cluster_assignments(self.sdrs,
                                            self.clusters,
                                            ignore_noise)
    self.assertEqual(len(cluster_assignments), num_categories * self.reps)
    self.assertEqual(len(sdr_slices), num_categories * self.reps)
