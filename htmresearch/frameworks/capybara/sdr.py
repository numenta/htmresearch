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


import pandas as pd
import json
import numpy as np


def load_sdrs(file_path, sp_output_width, tm_output_width):

  return pd.read_csv(file_path, converters={
    'spActiveColumns': sdr_converter(sp_output_width),
    'tmPredictedActiveCells': sdr_converter(tm_output_width)})


def sdr_converter(sdr_width):

  def convert_sdr(patternNZ_strings):
    patternNZs = json.loads(patternNZ_strings)
    sdrs = []
    for patternNZ in patternNZs:
      sdr = np.zeros(sdr_width)
      sdr[patternNZ] = 1
      sdrs.append(sdr.tolist())
    return np.array(sdrs)


  return convert_sdr
