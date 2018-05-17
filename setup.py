"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

setup(
    name='phdplanner',
    version='1.0.0',

    description='A PHD based motion planner',
    url='https://github.com/jonatanolofsson/phdplanner',

    # Author details
    author='Jonatan Olofsson',
    author_email='jonatan.olofsson@gmail.com',

    # Choose your license
    license='GPLv3',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='lbm tracking multi-target',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
)
