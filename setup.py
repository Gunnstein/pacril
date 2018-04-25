# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(name='pacril',
      version='1.3.4',
      url='https://github.com/Gunnstein/pacril',
      license='MIT',
      description='Package for generating responses from influence lines and loads. Inverse methods are also included',
      author='Gunnstein T. Froeseth',
      author_email='gunnstein.t.froseth@ntnu.no',
      packages=find_packages(exclude=["test"]),
      install_requires=['numpy', 'scipy', 'nose']
     )
