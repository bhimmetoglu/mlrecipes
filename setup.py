"""
Burak Himmetoglu
Setup file for package MLBook package
"""

from setuptools import setup

setup(name='mlbook',
      version='0.1',
      description='ML book codes',
      author='Burak Himmetoglu',
      author_email='burakhmmtgl@gmail.com',
      url = 'https://www.burakhimmetoglu.com',
      packages = ['mlbook', 'mlbook.linear_regression', 'mlbook.utils'],
      install_requires = ['numpy', 'pandas', 'joblib', 'scipy', 'scikit-learn'],
      classifier = [ 'Programming Language :: Python :: 3']
      )
