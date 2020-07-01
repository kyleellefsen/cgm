# Causal Graphical Models

A python library for building causal graphical models, closely following Daphne 
Koller's Coursera course on Probabilistic Graphical Models, and her 2009 book 
_Probabilistic Graphical Models: Principles and Techniques_. 
The source for this project is available [here][src].

## Installation
[NumPy][numpy] is the only dependency. Python version must be >= 3.7. 

    pip install cgm

## Usage

```python
import numpy as np
import cgm

np.random.seed(30)
# Define all nodes
A = cgm.CG_Node('A', nStates=3)
B = cgm.CG_Node('B', 3)
C = cgm.CG_Node('C', 3)
D = cgm.CG_Node('D', 3)
# Specify all parents of nodes
cgm.CPD(child=B, parents=[A])
cgm.CPD(C, [B])
cgm.CPD(D, [A, B])
# Create causal graph
graph = cgm.CG([A, B, C, D])
```

[src]: https://github.com/kyleellefsen/cgm
[numpy]: https://numpy.org/