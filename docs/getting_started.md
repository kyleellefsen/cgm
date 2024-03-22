# Getting Started

```bash
pip install cgm
```

```python
import cgm

# Define all nodes
A = cgm.CG_Node('A', num_states=3)
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