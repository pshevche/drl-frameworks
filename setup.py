#!/usr/bin/env python3
from setuptools import find_packages, setup


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="drl_fw",
    version="0.0.1",
    long_description=readme(),
    url="https://github.com/pshevche/drl-frameworks",
    packages=['drl_fw'],
    install_requires=[],
    dependency_links=[],
)
