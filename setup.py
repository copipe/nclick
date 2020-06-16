from os import path
from setuptools import find_packages, setup


def get_version():
    version_filepath = path.join(path.dirname(__file__), 'nclick', 'version.py')
    with open(version_filepath) as f:
        for line in f:
            if line.startswith('__version__'):
                return line.strip().split()[-1][1:-1]

setup(
    name='nclick',
    packages=find_packages(),
    version=get_version(),
    license='MIT',
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'sklearn',
        'tqdm',
        'openpyxl',
        'autopep8',
        'flake8',
        'pytest',
        'lightgbm',
        'xgboost',
        'catboost>=0.17',
        'torch',
        'ipython',
        'dill',
    ],
    author='copypaste',
    description='Code for Laziness.',
)
