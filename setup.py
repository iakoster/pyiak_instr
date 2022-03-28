import setuptools

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    packages=setuptools.find_packages(
        exclude=['*tests*']
    ),
    install_requires=required
)
