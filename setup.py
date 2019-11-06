#!/usr/bin/python3

from io import open
from distutils.core import setup


def read(filename):
    with open(filename, encoding='utf-8') as file:
        return file.read()


def requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as file:
        return file.readlines()


setup(name='pyRC',
      version='0.1',
      description='Reservoir Computing Library',
      long_description=read('README.md'),
      long_description_content_type="text/markdown",
      author='Lucas Burger',
      author_email='Lucas.Burger@uni-konstanz.de',
      url='https://lucasburger.github.io/pyRC',
      packages=['pyRC'],
      install_requires=requirements(),
      )
