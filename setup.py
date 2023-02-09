import setuptools

from pyiak_instr import __version__

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    version=__version__,
    packages=setuptools.find_packages(
        exclude=['*tests*']
    ),
    install_requires=required
)
