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
import csv
import sys



def run(filename):
  with open(filename, 'rU') as infile:
    reader = csv.reader(infile)
    outfilename = "trivial_" + filename

    with open(outfilename, 'wb') as outfile:
      writer = csv.writer(outfile)

      writer.writerow(reader.next())

      for row in reader:
        writer.writerow([row[0], row[1], row[1]])

      print "Wrote to:", outfilename



if __name__ == "__main__":
  run(sys.argv[1])
