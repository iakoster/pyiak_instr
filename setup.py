import shutil
import setuptools
from pathlib import Path

if (Path() / 'dist').exists():
    shutil.rmtree(Path() / 'dist')

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    packages=setuptools.find_packages(
        exclude=['*tests*']
    ),
    install_requires=required
)

shutil.rmtree(Path() / 'build')
shutil.rmtree(Path() / 'pyinstr_iakoster.egg-info')
