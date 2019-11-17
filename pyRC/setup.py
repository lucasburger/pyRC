import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = f.readlines()

setuptools.setup(
    name="pyRC",
    version="0.1.0",
    author="Lucas Burger",
    author_email="Lucas.Burger@uni-konstanz.de",
    description="Reservoir Computing Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://lucasburger.github.io/pyRC",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    install_requires=requirements,
    python_requires=">=3.6"
)
