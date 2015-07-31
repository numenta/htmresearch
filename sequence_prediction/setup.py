import os
from setuptools import setup, find_packages



def read(fname):
  """
  Utility function to read specified file.
  """
  path = os.path.join(os.path.dirname(__file__), fname)
  return open(path).read()



setup(name="sequence_prediction",
      version="0.0",
      description="Compare HTM with other algorithms on time series prediction",
      packages=find_packages(),
      long_description=read("README.md"))
