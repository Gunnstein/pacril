# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


# Parse the version from the module.
with open('pacril/__init__.py') as f:
    for line in f:
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            break

setup(name='pacril',
      version=version,
      url='https://github.com/Gunnstein/pacril',
      license='MIT',
      description='Package for generating responses from influence lines and loads. Inverse methods are also included',
      author='Gunnstein T. Froeseth',
      author_email='gunnstein.t.froseth@ntnu.no',
      packages=find_packages(exclude=["test"]),
      install_requires=['numpy', 'scipy', 'nose']
     )
