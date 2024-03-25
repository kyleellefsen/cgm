# Causal Graphical Models

[![CGM Tests](https://github.com/kyleellefsen/cgm/actions/workflows/cgm_tests.yml/badge.svg)](https://github.com/kyleellefsen/cgm/actions/workflows/cgm_tests.yml)
[![PyPi Publish](https://github.com/kyleellefsen/cgm/actions/workflows/publish_to_pypi.yml/badge.svg?event=release)](https://github.com/kyleellefsen/cgm/actions/workflows/publish_to_pypi.yml?query=event%3Arelease)
[![PyPi Version](https://img.shields.io/pypi/v/cgm)](https://pypi.org/project/cgm/)
![PyPI - Status](https://img.shields.io/pypi/status/cgm)
![PyPI - Format](https://img.shields.io/pypi/format/cgm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/kyleellefsen/cgm/blob/master/LICENSE)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fkyleellefsen%2Fcgm%2Fmaster%2Fpyproject.toml)](https://www.python.org/)
![GitHub last commit](https://img.shields.io/github/last-commit/kyleellefsen/cgm)




A python library for building causal graphical models, closely following Daphne 
Koller's Coursera course on Probabilistic Graphical Models, and her 2009 book 
_Probabilistic Graphical Models: Principles and Techniques_. 
The source for this project is available [here][src].

## Installation
[NumPy][numpy] is the only dependency. Python version must be >= 3.7. 

    pip install cgm

## Usage

```python
import cgm

# Define all nodes
A = cgm.CG_Node('A', num_states=3)
B = cgm.CG_Node('B', 3)
C = cgm.CG_Node('C', 3)
D = cgm.CG_Node('D', 3)
# Specify all parents of nodes
cgm.CPD([B, A])
cgm.CPD([B, C])
cgm.CPD([D, A, B])
# Create causal graph
graph = cgm.CG([A, B, C, D])
print(graph)
# A ← []
# B ← [C]
# C ← []
# D ← [A, B]
```

[src]: https://github.com/kyleellefsen/cgm
[numpy]: https://numpy.org/

## Documentation
[kyleellefsen.github.io/cgm](https://kyleellefsen.github.io/cgm/)