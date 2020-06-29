import numpy as np
from typing import List
import logging
from ..core import Variable, Factor, DAG_Node, CPD, DAG

class Cluster:
    def __init__(self, factors: List[Factor]):
        self.factors = factors
        self.scope = sorted(list(set.union(*[set(f.scope) for f in self.factors])))
        self.edges = []
        self.messages = dict()  # incoming messages, where each edge is the key
        self.potential = self._get_potential()

    def add_edge(self, edge):
        assert set(edge.scope).issubset(set(self.scope))
        self.edges.append(edge)
        self.messages[edge] = Factor.getNull()
    
    def send_message(self, edge):
        target_node = edge.get_other_node(self)
        variables_to_marginalize = \
            sorted(list(set(self.scope) - set(edge.scope)))
        marginalized_belief = self.get_belief() \
            .marginalize(variables_to_marginalize)
        msg = marginalized_belief / self.messages[edge]
        target_node.messages[edge] = msg

    def __repr__(self):
        return "ψ(" + ", ".join([f"{s}" for s in self.scope]) + ")"
    
    def _get_potential(self):
        product = self.factors[0]
        for f in self.factors[1:]:
            product = product * f
        return product
    
    def get_belief(self):
        belief = self.potential
        for msg in self.messages.values():
            belief = belief * msg
        return belief


class ClusterEdge:
    def __init__(self, nodes: List[Cluster], scope: List[Factor]):
        assert len(nodes) == 2
        self.nodes = nodes
        self.scope = scope
        for n in self.nodes:
            n.add_edge(self)
    
    def get_other_node(self, node):
        return (set(self.nodes) - set([node])).pop()
    
    def __repr__(self):
        n0 = self.nodes[0]
        n1 = self.nodes[1]
        return f"{n0} <-- {self.scope} --> {n1}"


class ClusterGraph:
    """A cluster graph is used for computing marginal probability distributions. 
    It is an undirected graph with edges between clusters. """
    def __init__(self, nodes: List[Cluster]):
        self.nodes = nodes
        self.variables = self._get_variables()
        self.factors = self._get_factors()

    def _get_variables(self):
        variables = set()
        for n in self.nodes:
            for f in n.factors:
                variables.update(f.scope)
        return variables
    
    def get_variable(self, name: str):
        return [v for v in self.variables if v.name==name][0]
    
    def _get_factors(self):
        factors = set()
        for n in self.nodes:
            factors.update(n.factors)
        return factors


    def propagate_beliefs_round_robin(self, nTimes):
        for _ in range(nTimes):
            for cluster in self.nodes:
                for edge in cluster.edges:
                    cluster.send_message(edge)

    def forward_backward_algorithm(self):
        """ This only runs on chains with no branches """
        def make_pass(nodes):
            previous_node = None
            for node in nodes[:-1]:
                edge = node.edges[0]
                if previous_node == edge.get_other_node(node):
                    edge = node.edges[1]
                previous_node = node
                node.send_message(edge)
        make_pass(self.nodes) # forward pass
        make_pass(self.nodes[::-1]) # backward pass

