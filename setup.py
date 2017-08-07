import versioneer
from setuptools import setup, find_packages

setup(
    name='seigen',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="A finite element-based elastic wave equation solver
    for seimological problems",
    long_descritpion="""Seigen is an elastic wave equation solver for
    seimological problems based on the Firedrake finite element
    framework. It forms part of the OPESCI seismic imaging project.""",
    url='http://www.opesci.org/',
    author="Imperial College London",
    author_email='opesci@imperial.ac.uk',
    license='MIT',
    packages=find_packages(exclude=['tests']),
    install_requires=[],
    test_requires=[]
)
