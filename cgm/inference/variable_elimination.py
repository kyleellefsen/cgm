import numpy as np
from typing import List
import logging
from ..core import Variable, Factor, DAG_Node, CPD, DAG


class DagVariableElimination:
    def __init__(self, dag: DAG):
        self.dag = dag

    def variable_elimination(self, nodes_to_eliminate: List[DAG_Node]):
        """Implements the variable elimination algorithm over a DAG.
        Specify a list of nodes to eliminate, in order of elimination
        """
        factors = [n.cpd for n in self.dag.nodes]
        for n in nodes_to_eliminate:
            logging.debug(f"Eliminating {n}")
            factors = self._eliminate_node(n, factors)
            logging.debug(factors)
        return factors

    def _eliminate_node(self, node_to_eliminate: DAG_Node, factors: List[Factor]):
        factors_to_combine = [f for f in factors if node_to_eliminate in f.scope]
        intermediate_factor = factors_to_combine[0]
        for f in factors_to_combine[1:]:
            intermediate_factor = intermediate_factor * f
        marginal_factor = intermediate_factor.marginalize([node_to_eliminate])
        reduced_factors = [f for f in factors if f not in factors_to_combine] + [marginal_factor]
        return reduced_factors

    def get_scope_of_new_factor_after_variable_elimination(self):
        """Returns a dictionary containing the new factor created after 
        eliminating a node, for every node. To reduce computation, we want to
        create new intermediate factors that have the smallest scope.
        """
        nodes = self.dag.nodes
        return {n_target : set.union(*[set(n.cpd.scope) for n in nodes if n_target in n.cpd.scope]) \
                - set([n_target]) for n_target in nodes}


