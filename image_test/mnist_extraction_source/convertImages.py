# ----------------------------------------------------------------------
#  Copyright (C) 2006-2008 Numenta Inc. All rights reserved.
#
#  The information and source code contained herein is the
#  exclusive property of Numenta Inc. No part of this software
#  may be used, reproduced, stored or distributed in any form,
#  without explicit written authorization from Numenta Inc.
# ----------------------------------------------------------------------

import os
import numpy
from PIL import Image

def doConversion():

  # Inner function to process each directory
  def _visitorProc(state, dirname, names):

    # Inner function to convert one image
    def _convertData(srcPath, dstPath):
      fpSrc = open(srcPath, 'r')
      rawLines = fpSrc.readlines()
      numRows, numCols = [int(token) for token in rawLines[0].split()]
      pixelRows = rawLines[1:]
      pixels = []
      for pixelRow in pixelRows:
        pixels += [int(token) for token in pixelRow.strip().split()]
      fpSrc.close()
      # Create array
      numpyImg = numpy.array(pixels, dtype=numpy.uint8).reshape(numRows, numCols)
      image = Image.fromarray(numpyImg, "L")
      image.save(dstPath)
      # Destroy original text version
      os.remove(srcPath)

    # Process the contents of the directory
    for name in names:
      imgName, imgExt = os.path.splitext(name)
      if imgExt == '.txt':
        srcPath = os.path.join(dirname, name)
        dstPath = os.path.join(dirname, imgName + ".png")
        print "%s ==> %s" % (srcPath, dstPath)
        _convertData(srcPath, dstPath)
        state['numImages'] += 1

  # Perform final conversion
  state = dict(numImages=0)
  os.path.walk("training", _visitorProc, state)
  os.path.walk("testing", _visitorProc, state)
  print "Total images: %d" % state['numImages']


if __name__ == "__main__":
  doConversion()
