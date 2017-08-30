import platform
import sys

from setuptools import find_packages, setup

setup(name="htmresearchviz0",
      version="0.0.1",
      description="",
      packages=find_packages(),
      package_data={'htmresearchviz0': ['htmresearchviz0/package_data/*',]},
      include_package_data=True,
      zip_safe=False,
      )
