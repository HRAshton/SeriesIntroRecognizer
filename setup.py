# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='series_opening_recogniser',
    version='0.1.0',
    description='Series Opening Recogniser',
    long_description=readme,
    author='HRAshton',
    url='https://github.com/HRAshton/SeriesOpeningRecogniser',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
