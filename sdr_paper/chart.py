#!/usr/bin/env python
# ----------------------------------------------------------------------
# Copyright (C) 2015, Numenta Inc. All rights reserved.
#
# The information and source code contained herein is the
# exclusive property of Numenta Inc.  No part of this software
# may be used, reproduced, stored or distributed in any form,
# without explicit written authorization from Numenta Inc.
# ----------------------------------------------------------------------

"""TODO"""

import sys

from matplotlib import pyplot


if __name__ == "__main__":
  lines = [line.strip().split(",") for line in sys.stdin]
  data = [(int(pair[0]), float(pair[1])) for pair in lines]
  thetas, probs = zip(*data)
  pyplot.plot(thetas, probs)
  pyplot.show()
