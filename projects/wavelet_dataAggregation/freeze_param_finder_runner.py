# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
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

import cx_Freeze
import os



def main():
  """
  Package the script. Warning: This assumes that this script is in the 
  same directory as target script to freeze.
  """
  # Initial cleanup.
  scriptDir = os.path.dirname(os.path.realpath(__file__))
  buildDir = os.path.join(scriptDir, "build")
  distDir = os.path.join(scriptDir, "dist")
  os.system("rm -rf build %s %s" % (buildDir, distDir))


  # Freeze the script
  executables = [cx_Freeze.Executable(os.path.join(scriptDir,
                                                   "param_finder_test.py"),
                                      targetName="param_finder_test")]

  freezer = cx_Freeze.Freezer(executables,
                              silent=True)

  freezer.Freeze()



if __name__ == "__main__":
  main()
