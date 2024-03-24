import logging
import cgm


class DagVariableElimination:
    def __init__(self, cg: cgm.CG):
        self.cg = cg

    def variable_elimination(self, nodes_to_eliminate: list[cgm.CG_Node]):
        """Implements the variable elimination algorithm over a CG.
        Specify a list of nodes to eliminate, in order of elimination
        """
        factors: set[cgm.CPD] = set(n.cpd for n in self.cg.nodes)
        for n in nodes_to_eliminate:
            logging.debug("Eliminating %s", n)
            factors = self._eliminate_node(n, factors)
            logging.debug(factors)
        return factors

    def _eliminate_node(self, node_to_eliminate: cgm.CG_Node, factors: set[cgm.CPD]):
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
        nodes = self.cg.nodes
        return {n_target : set.union(*[set(n.cpd.scope) for n in nodes if n_target in n.cpd.scope]) \
                - set([n_target]) for n_target in nodes}

