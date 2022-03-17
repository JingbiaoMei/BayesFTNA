from platform import version
from setuptools import setup
from setuptools import find_packages
setup(
    name="Fault Neural Network Architecture",
    version='1.0.0',
    description='This package provides ways to search for Robust Neural Architecture',
    author='Jingbiao Mei',
    author_email='jm2245@cam.ac.uk',
    packages=find_packages(exclude=('tests','data','results')),
    url='https://github.com/jingbiaoMei/BayeFT'

)