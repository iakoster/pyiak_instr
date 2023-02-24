from setuptools import setup, find_packages

from pyiak_instr_deprecation import __version__

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    version=__version__,
    packages=find_packages(where="src"),
    install_requires=required,
    package_dir={"": "src"},
)
