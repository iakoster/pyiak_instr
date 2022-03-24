import shutil
import setuptools
from pathlib import Path

if (Path() / 'dist').exists():
    shutil.rmtree(Path() / 'dist')

setuptools.setup(
    packages=setuptools.find_packages(
        exclude=['*tests*']
    ),
    zip_file=False
)

shutil.rmtree(Path() / 'build')
shutil.rmtree(Path() / 'pyinstr_iakoster.egg-info')
