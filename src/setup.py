#!/usr/bin/env python3
from setuptools import find_packages, setup


setup(
    name="drl_fw",
    version="0.0.1",
    url="https://github.com/pshevche/drl-frameworks",
    packages=find_packages(),
    package_data={
        'park': ['join-order-benchmark/*.sql', 'query-optimizer/*', 'query-optimizer/src/main/java/*']
    },
    install_requires=[],
    dependency_links=[],
)
