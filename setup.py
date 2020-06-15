from setuptools import find_packages, setup


setup(
    name='nclick',
    packages=find_packages(),
    version='0.0.0',
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
    ],
    author='copypaste',
    description='Code for Laziness.',
)
