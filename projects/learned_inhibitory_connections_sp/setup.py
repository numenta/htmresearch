
from setuptools import setup, find_packages

setup(name='LateralPooler',
      version='1.0',
      description='Experimental spatial pooler implementation',
      author='Mirko Klukas',
      author_email='mklukas@numenta.com',
      package_dir={"": "src"},
      packages=find_packages("src")
     )