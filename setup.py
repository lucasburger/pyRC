import setuptools
import json

with open("README.md", "r") as fh:
    long_description = fh.read()

setup = json.load(open('setup.json', 'r'))

with open("requirements.txt", "r") as f:
    requirements = f.readlines()

setuptools.setup(
    long_description=long_description,
    packages=setuptools.find_packages(),
    install_requires=requirements,
    **setup
)
