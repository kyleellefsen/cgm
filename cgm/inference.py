import numpy as np
from typing import List
import logging
from .core import Variable, Factor, DAG_Node, CPD, DAG

def eliminate_node(node_to_eliminate: List[DAG_Node], factors: List[Factor]):
    factors_to_combine = [f for f in factors if node_to_eliminate in f.scope]
    intermediate_factor = factors_to_combine[0]
    for f in factors_to_combine[1:]:
        intermediate_factor = intermediate_factor * f
    marginal_factor = intermediate_factor.marginalize([node_to_eliminate])
    reduced_factors = [f for f in factors if f not in factors_to_combine] + [marginal_factor]
    return reduced_factors

def eliminate_nodes(nodes_to_eliminate: List[DAG_Node], dag: DAG):
    """
    Implements the variable elimination algorithm over a DAG.
    Specify a list of nodes to eliminate


    """
    factors = [n.cpd for n in dag.nodes]
    for n in nodes_to_eliminate:
        logging.info(f"Eliminating {n}")
        factors = eliminate_node(n, factors)
        logging.info(factors)
    return factors

    
