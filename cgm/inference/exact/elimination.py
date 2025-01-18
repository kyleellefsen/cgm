"""Variable Elimination Algorithm for Bayesian Networks

Let G be a Bayesian network with nodes V and factors Φ. For any subset of variables 
X ⊆ V, variable elimination computes P(X) by successively eliminating variables 
Y ∈ V\X according to some elimination ordering π.

For each variable Y to eliminate:
1. Let Φ_Y be the set of all factors φ containing Y
2. Let φ_Y = ∏_{φ ∈ Φ_Y} φ be their product 
3. Let μ_Y = ∑_Y φ_Y be the marginalization of Y from φ_Y
4. Update Φ = (Φ \ Φ_Y) ∪ {μ_Y}

The final set of factors Φ gives the desired marginal probability P(X).
"""

from typing import TypeAlias
import cgm

FactorSet: TypeAlias = set[cgm.Factor[cgm.CG_Node]]


def eliminate(cg: cgm.CG, elimination_order: list[cgm.CG_Node]) -> FactorSet:
    """Eliminate variables from a Bayesian network in the given order.
    
    Args:
        cg: A Bayesian network represented as a Causal Graph
        elimination_order: Sequence of variables to eliminate
        
    Returns:
        The set of remaining factors after elimination
    """
    # Initial set of factors Φ is all CPDs in the network
    factors: FactorSet = {node.cpd for node in cg.nodes if node.cpd is not None}

    # Eliminate each variable in order
    for var in elimination_order:
        # Find all factors containing this variable: Φ_Y
        relevant_factors = {f for f in factors if var in f.scope}

        # Remove these factors and add their marginalized product
        if relevant_factors:
            # Multiply factors: φ_Y = ∏ Φ_Y
            product = next(iter(relevant_factors))
            for factor in list(relevant_factors)[1:]:
                product = product * factor

            # Marginalize out the variable: μ_Y = ∑_Y φ_Y
            marginal = product.marginalize([var])

            # Update factor set: Φ = (Φ \ Φ_Y) ∪ {μ_Y}
            factors = (factors - relevant_factors) | {marginal}

    return factors
