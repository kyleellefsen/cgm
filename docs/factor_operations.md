# Factor Operations

## Factor Multiplication
The factor product is defined in Probabilistic Graphical Models
Definition 4.2 (Koller 2009). That definition is quoted below.

> Let $X$, $Y$, $Z$ be three disjoint sets of variables, and let $\phi_1(X, Y)$
> and $\phi_2(Y, Z)$ be two factors. We define the factor product
> $\phi_1 \times \phi_2$ to be a factor $\psi: Val(X, Y, Z) \to \mathbb{R}$ as 
> follows:
> $$ \psi(X, Y, Z) \doteq  \phi_1(X, Y) \cdot \phi_2(Y, Z) $$


cgm will line up the overlapping dimensions automatically, and multiply the factors along these dimensions.

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