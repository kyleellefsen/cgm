# Factors

## Factor Overview

The definitions used in this library, unless otherwise specified, come from ["Probailistic Graphical Models: Principles and Techniques" by Daphne Koller and Nir Friedman (2009)](https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/).

A **factor** is mapping from variables to $\mathbb{R}$. Suppose you have some 
variables $A$, $B$, and $C$ that can take on some finite set of states. You 
might want to count the number of times every particular combination 
$(A=a, B=b, C=c)$ of these variables co-occurs. You can store that count in a 
factor.

A **conditional probability distribution (CPD)** is a factor where the parent 
variables are mapped to a number $p \in (0, 1)$ for all possible values of the 
child variable, with the additional constraint that sum over the range equals 1. 
For example, if the parent variables are $B$ and $C$, the child variable $A$, 
and all variables take on two possible values, then a CPD $\phi$ might be 
expressed in the table:

| B | C | A | P(A &#124; B, C) |
|---|---|---|-------------|
| 0 | 0 | 0 | P(A=0 &#124; B=0, C=0) |
| 0 | 0 | 1 | P(A=1 &#124; B=0, C=0) |
| 0 | 1 | 0 | P(A=0 &#124; B=0, C=1) |
| 0 | 1 | 1 | P(A=1 &#124; B=0, C=1) |
| 1 | 0 | 0 | P(A=0 &#124; B=1, C=0) |
| 1 | 0 | 1 | P(A=1 &#124; B=1, C=0) |
| 1 | 1 | 0 | P(A=0 &#124; B=1, C=1) |
| 1 | 1 | 1 | P(A=1 &#124; B=1, C=1) |

The diagram below shows the inheritance hierchy between types of factors and 
types of variables. DAG (Directed Acyclic Graph) nodes are constrained to 
ensure there are no cycles in the associated factors. CG Nodes are associated 
with only a single CPD. This additional constraints are why 
{py:class}`cgm.core.CPD` uses {py:class}`cgm.core.CG_Node`.

```{mermaid}
flowchart TB
    cgm.Factor-.->cgm.Variable
    cgm.CPD-.->cgm.CG_Node
    subgraph Variable Hierarchy
        direction TB
        cgm.Variable-->cgm.DAG_Node
        cgm.DAG_Node-->cgm.CG_Node
    end
    subgraph Factor Hierarchy
        direction TB
        cgm.Factor-->cgm.CPD
    end
    click cgm.Factor "core.html#cgm.core.Factor" _blank
    click cgm.Variable "core.html#cgm.core.Variable" _blank
    click cgm.DAG_Node "core.html#cgm.core.DAG_Node" _blank
    click cgm.CPD "core.html#cgm.core.CPD" _blank
    click cgm.CG_Node "core.html#cgm.core.CG_Node" _blank
```

## Factor Operations
`````{tabs}
````{group-tab} Multiplication

### Multiplication

#### Factor Multiplication
The factor product is defined in Probabilistic Graphical Models
Definition 4.2 (Koller 2009). That definition is quoted below.

> Let $X$, $Y$, $Z$ be three disjoint sets of variables, and let $\phi_1(X, Y)$
> and $\phi_2(Y, Z)$ be two factors. We define the factor product
> $\phi_1 \times \phi_2$ to be a factor $\psi: Val(X, Y, Z) \to \mathbb{R}$ as 
> follows:
>> $$\psi(X, Y, Z) \doteq  \phi_1(X, Y) \cdot \phi_2(Y, Z)$$


{py:mod}`cgm` will line up the overlapping dimensions automatically, and 
multiply the factors along these dimensions.

```python
import cgm

X = cgm.Variable('X', num_states=2)
Y = cgm.Variable('Y', num_states=3)
Z = cgm.Variable('Z', num_states=4)
phi_1 = cgm.Factor(scope=[X, Y])
phi_2 = cgm.Factor(scope=[Y, Z])
psi = phi_1 * phi_2

print(psi)
# ϕ(X, Y, Z)
```

#### CPD Multiplication

CPD multiplication works exactly the same way as factor multiplcation.
However, the API for creating CPDs is slightly different.

```python
import cgm

g = cgm.CG()
X = g.node('X', num_states=2)
Y = g.node('Y', num_states=3)
Z = g.node('Z', num_states=4)
phi_1 = g.P(X | [Y, Z])
phi_2 = g.P(Y | Z)
phi_3 = phi_1 * phi_2
print(phi_3)
# ϕ(X, Y, Z)
print(type(phi_3))  # The CPD is now an unnormalized Factor.
# <class 'cgm.Factor'>
print(phi_3.table)
# ϕ(X⁰, Y, Z) |    Z⁰    Z¹    Z²    Z³
# ─────────────────────────────────────
# Y⁰     |  0.136  0.099  0.126  0.141
# Y¹     |  0.020  0.182  0.010  0.289
# Y²     |  0.233  0.213  0.113  0.249
```






````
````{group-tab} Conditioning

## Conditioning

## Factor Conditioning

You can condition a factor on a set of values for variables.

$$
f(\phi_{X, Y}(X, Y), y=0) \doteq \phi(x, y=0)
$$

```python
import cgm

X = cgm.Variable('X', num_states=2)
Y = cgm.Variable('Y', num_states=3)
Z = cgm.Variable('Z', num_states=4)
phi_1 = cgm.Factor(scope=[X, Y, Z])
phi_2 = phi_1.condition({X: 0, Y: 1})
print(phi_2)
# ϕ(Z)
```

## CPD Conditioning


Conditioning for CPDs is the same as conditioning on a factor, except that 
the only allowed variables to condition on are the parents of the CPD.

```python
import cgm

g = cgm.CG()
X = g.node('X', num_states=2)
Y = g.node('Y', num_states=3)
Z = g.node('Z', num_states=4)
phi_1 = g.P(X | [Y, Z])
print(phi_1.table)
# 𝑃(X | Y, Z)  |    X⁰    X¹
# ─────────────────────────────
# Y⁰, Z⁰       |  0.794  0.206
# Y⁰, Z¹       |  0.526  0.474
# Y⁰, Z²       |  0.626  0.374
# Y⁰, Z³       |  0.522  0.478
# Y¹, Z⁰       |  0.747  0.253
# Y¹, Z¹       |  0.514  0.486
# Y¹, Z²       |  0.635  0.365
# Y¹, Z³       |  0.418  0.582
# Y², Z⁰       |  0.447  0.553
# Y², Z¹       |  0.371  0.629
# Y², Z²       |  0.371  0.629
# Y², Z³       |  0.416  0.584

phi_2 = phi_1.condition({Y: 0, Z: 1})
print(phi_2)
# 𝐏(X)
print(type(phi_2))  # The result is a CPD
# <class 'cgm.CPD'>
print(phi_2.table)
# 𝑃(X) |    X⁰    X¹
# ──────────────────
#      |  0.526  0.474
```

````
````{group-tab} Marginalization

#### Marginalization

#### Factor Marginalization
The marginalization of a factor over its variable eliminates that variable 
from the factor by summing over all its possible states. {py:mod}`cgm`'s 
factor.marginalize() function allows simultaneous marginalization over 
multiple variables.

$$
\begin{aligned}
f(\phi(X, Y), Y) &\doteq \sum_{y \in Y} \phi(X, Y=y) \\
                 &\to \psi(X)
\end{aligned}
$$

```python
import cgm

X = cgm.Variable('X', num_states=2)
Y = cgm.Variable('Y', num_states=3)
Z = cgm.Variable('Z', num_states=4)
phi_1 = cgm.Factor(scope=[X, Y, Z])
phi_2 = phi_1.marginalize([Y, Z])
print(phi_2)
# ϕ(X)
```

One can also marginalize a factor over a CPD:


$$
\begin{aligned}
f(\phi(X, Y), P_{Y|X}(y|x)) &\doteq \sum_{y \in Y} \phi(X, Y=y) P_{Y|X}(Y=y | X) \\
                            &\to \phi_{new}(X)
\end{aligned}
$$

```python
import cgm

g = cgm.CG()
X = g.node('X', num_states=2)
Y = g.node('Y', num_states=3)
phi1 = cgm.Factor([X, Y])
cpd = g.P(Y | X)
phi2 = phi1.marginalize_cpd(cpd)
print(phi2)
# ϕ(X)
```

#### CPD Marginalization

It's common to want to marginalize over a parent variable in a conditional 
probability distribution.

$$
\begin{aligned}
f(P_{X|Y}(x|y), P_Y(y)) &\doteq \sum_{y \in Y} P_{X|Y}(X | Y=y) P_Y(Y=y)\\
                        &\to P_X(X)
\end{aligned}
$$

Or in general, when the $Y$ variable also has parents:

$$
\begin{aligned}
f(P_{X|Y, \mathbf{Z}}(x|y), P_{Y|\mathbf{Z}}(y)) &\doteq \sum_{y \in Y} 
                                P_{X|Y}(X | Y=y, Z_1,..., Z_j) 
                                P_{Y|\mathbf{Z}}(Y=y| Z_1, Z_2, ... Z_j) \\
                        &\to P_X(X|Z_1...Z_j)
\end{aligned}
$$

Currently, {py:mod}`cgm` only allows CPDs to have a single child variable; there is 
no joint conditional distribution $P(X, Y | Z)$. As a result, It's only
possible to marginalize over a single variable at a time.

```python
import cgm

g = cgm.CG()
X = g.node('X', 2)
Y = g.node('Y', 3)
Z = g.node('Z', 4)
phi_1 = g.P(X | [Y, Z])
phi_2 = g.P(Y | Z)
phi_3 = phi_1.marginalize_cpd(phi_2)
print(phi_3)
# 𝐏(X | Z)
print(phi_3.table)
# 𝑃(X | Z)  |    X⁰    X¹
# ──────────────────────────
# Z⁰        |  0.377  0.623
# Z¹        |  0.227  0.773
# Z²        |  0.388  0.612
# Z³        |  0.539  0.461

```

````
`````