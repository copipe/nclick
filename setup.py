from setuptools import find_packages, setup


setup(
    name='nclick',
    packages=find_packages(),
    version='0.0.0'
    license='MIT',
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'sklearn',
        'tqdm',
        'openpyxl',
    ],
    author='copypaste',
    description='Code for Laziness.',
)
