import os
from setuptools import setup, find_packages



def read(fname):
  """
  Utility function to read specified file.
  """
  path = os.path.join(os.path.dirname(__file__), fname)
  return open(path).read()



setup(name="union_pooling",
      version="0.0",
      description="Union Temporal Pooler and related experiments.",
      packages=find_packages(),
      long_description=read("README.md"))
