# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 21:51:19 2020

@author: Troshin Mikhail
m.troshin.ml@gmail.com
"""


from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

version = '1.0.0'

requirements = [
    'seaborn',
    'numpy',
    'pandas',
    'matplotlib',
    'lightgbm',
    'nltk',
    'scikit_learn'
    ]

with open(f'{here}/README.md', 'r') as f:
    readme = f.read()

setup(
      #Metadata
      name='auto_ml_basic',
      version=version,
      description='AutoML package that allows to solve a classification and regression tesks',
      license="MIT",
      long_description=readme,
      author='Troshin Mikhail',
      author_email='m.troshin.ml@gmail.com',
      package_dir={'': 'auto_ml_basic'},
      packages=find_packages(where='auto_ml_basic'),  #same as name
      install_requires=requirements #external packages as dependencies
      )


'''
      entry_points={
          'console_scripts': [
              'sample=sample:main',
              ],
          },
      
'''