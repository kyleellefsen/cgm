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
A = DAG_Node('A', 3)
B = DAG_Node('B', 3)
C = DAG_Node('C', 3)
D = DAG_Node('D', 3)
# Specify all parents of nodes
CPD(B, [A])
CPD(C, [B])
CPD(D, [A, B])
nodes = [A, B, C, D]
# Create graph
graph = DAG(nodes)
```

[src]: https://github.com/kyleellefsen/cgm
[numpy]: https://numpy.org/