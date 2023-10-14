from setuptools import setup, find_packages


def read_requirements():
    with open('requirements.txt') as req:
        return req.read().splitlines()


setup(
    name='dumb_cfd',
    version='0.0.1',
    packages=find_packages(),
    author='Liam Lamont',
    description='A dumb implementation of some cfd equations.',
    long_description=open('README.md').read(),
    install_requires=read_requirements(),
    keywords="cfd computational fluid dynamics"
)
