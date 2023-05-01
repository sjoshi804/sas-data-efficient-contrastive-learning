from setuptools import setup, find_packages

VERSION = '1.0'
DESCRIPTION = 'A python package implementing the data-efficient contrastive learning proposed in https://arxiv.org/abs/2302.09195 by S. Joshi and B. Mirzasoleiman'

setup(
    name="subcl",
    version=VERSION,
    author="Siddharth Joshi",
    author_email="sjoshi804@cs.ucla.edu",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=DESCRIPTION,
    packages=find_packages(),
    install_requires=['torch', 'torchvision', 'numpy', 'fast-pytorch-kmeans'],
)