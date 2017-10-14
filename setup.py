# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014-15, Numenta, Inc.  Unless you have purchased from
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
import os
import pkg_resources
import sys

from setuptools import find_packages, setup

REPO_DIR = os.path.dirname(os.path.realpath(__file__))

def getVersion():
  """
  Get version from local file.
  """
  with open(os.path.join(REPO_DIR, "VERSION"), "r") as versionFile:
    return versionFile.read().strip()


def nupicPrereleaseInstalled():
  """
  Make an attempt to determine if a pre-release version of nupic is
  installed already.

  @return: boolean
  """
  try:
    nupicDistribution = pkg_resources.get_distribution("nupic")
    if pkg_resources.parse_version(nupicDistribution.version).is_prerelease:
      # A pre-release dev version of nupic is installed.
      return True
  except pkg_resources.DistributionNotFound:
    pass  # Silently ignore.  The absence of nupic will be handled by
          # setuptools by default

  return False


def htmresearchCorePrereleaseInstalled():
  """
  Make an attempt to determine if a pre-release version of htmresearch-core is
  installed already.

  @return: boolean
  """
  try:
    coreDistribution = pkg_resources.get_distribution("htmresearch-core")
    if pkg_resources.parse_version(coreDistribution.version).is_prerelease:
      # A pre-release dev version of htmresearch-core is installed.
      return True
  except pkg_resources.DistributionNotFound:
    pass  # Silently ignore.  The absence of htmresearch-core will be handled by
          # setuptools by default

  return False

def parse_file(requirementFile):
  try:
    return [
      line.strip()
      for line in open(requirementFile).readlines()
      if not line.startswith("#")
    ]
  except IOError:
    return []

def findRequirements():
  """
  Read the requirements.txt file and parse into requirements for setup's
  install_requirements option.
  """
  requirementsPath = os.path.join(REPO_DIR, "requirements.txt")
  requirements = parse_file(requirementsPath)

  # User has a pre-release version of numenta packages installed, which is only
  # possible if the user installed and built the packages from source and
  # it is up to the user to decide when to update these packages.  We'll
  # quietly remove the entries in requirements.txt so as to not conflate the
  # two.
  if nupicPrereleaseInstalled():
    requirements = [req for req in requirements if "nupic" not in req]

  if htmresearchCorePrereleaseInstalled():
    requirements = [req for req in requirements if "htmresearch-core" not in req]

  return requirements

if __name__ == "__main__":

  requirements = findRequirements()

  setup(
    name="htmresearch",
    version=getVersion(),
    install_requires=requirements,
    packages=find_packages(include=["htmresearch", "htmresearch.*"]),
    package_data={
      "htmresearch": ["README.md", "LICENSE.txt"],
    },
    include_package_data=True,
    description="Numenta's HTM research code",
    author="Numenta",
    author_email="help@numenta.org",
    url="https://github.com/numenta/htmresearch",
    classifiers=[
      "Programming Language :: Python",
      "Programming Language :: Python :: 2",
      "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
      "Operating System :: MacOS :: MacOS X",
      "Operating System :: POSIX :: Linux",
      "Operating System :: Microsoft :: Windows",
      # It has to be "5 - Production/Stable" or else pypi rejects it!
      "Development Status :: 5 - Production/Stable",
      "Environment :: Console",
      "Intended Audience :: Science/Research",
      "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    long_description=(
      "This package contains code for experimental algorithm work done "
      "internally at Numenta.\n\n"
      "For more information, see http://numenta.org")
)
