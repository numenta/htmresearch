# Copyright (C) 2015-2016, Numenta, Inc.  Unless you have purchased from
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

from setuptools import setup, find_packages

installRequires = []
dependencyLinks = []

with open("requirements.txt", "r") as reqfile:
  for line in reqfile:
    line = line.strip()
    (link, _, package) = line.rpartition("#egg=")
    if link:
      # e.g., "-e git+http://github.com/numenta/nupic.fluent.git@0.2.1#egg=nupic.fluent-0.2.1"
      if line.startswith("-e"):
        line = line[2:].strip()

      dependencyLinks.append(line)

      (packageName, _, packageVersion) = package.partition("-")

      package = packageName + "==" + packageVersion

    installRequires.append(package)

setup(
  name="Imbu",
  version="0.1.0",
  description=("Sample application using nnupic.fluent platform"),
  url="https://github.com/numenta/numenta-apps/imbu",
  package_dir = {"": "engine"},
  packages = find_packages("engine"),
  include_package_data=True,
  install_requires = installRequires,
  dependency_links = dependencyLinks,
)
