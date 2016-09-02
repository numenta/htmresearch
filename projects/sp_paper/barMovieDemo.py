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

from htmresearch.support.generate_sdr_dataset import getMovingBar
import matplotlib.pyplot as plt
import numpy as np
plt.ion()

if __name__ == "__main__":
  bars = getMovingBar(startLocation=(10, 1),
                      direction=(0, 1),
                      imageSize=(20, 20),
                      steps=19)

  plt.figure(1)
  i = 0
  while True:
    plt.imshow(np.transpose(bars[i]), cmap='gray')
    plt.pause(.05)
    i += 1
    if i >= len(bars):
      i = 0