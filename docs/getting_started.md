# Getting Started

```bash
pip install cgm
```

```python
import cgm
import numpy as np

# Define all nodes
g = cgm.CG()
A = g.node('A', num_states=3)
B = g.node('B', 3)
C = g.node('C', 2)
D = g.node('D', 3)

# Define values for one of the CPDs
values = np.array([[0.1, 0.2, 0.7],
                   [0.3, 0.4, 0.3]]).T

# Specify all parents of nodes
phi1 = g.P(B | A)
phi2 = g.P(B | C, values=values)
phi3 = g.P(D | [A, B])

# Print the graph
print(g)
# A ← []
# B ← [C]
# C ← []
# D ← [A, B]

# Print the CPDs
print(phi2.table)
# 𝑃(B | C)  |    B⁰    B¹    B²
# ─────────────────────────────────
# C⁰        |  0.100  0.200  0.700
# C¹        |  0.300  0.400  0.300
```