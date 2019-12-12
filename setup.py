from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup
from setuptools import find_packages

with open('requirements.txt') as fp:
    install_requires = fp.read().split('\n')

setup(
    name='numba-neighbors',
    version='0.0.1',
    description='Numba implementation of scikit-learn neighbors',
    url='https://github.com/jackd/numba-neighbors.git',
    author='Dominic Jack',
    author_email='thedomjack@gmail.com',
    license='MIT',
    packages=find_packages(),
    requirements=install_requires,
    zip_safe=True,
)
