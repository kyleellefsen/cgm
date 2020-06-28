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
        self.factor_product = self._get_factor_product()
        self.belief = self.factor_product

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
    
    def _get_factor_product(self):
        product = self.factors[0]
        for f in self.factors[1:]:
            product = product * f
        return product
    
    def get_belief(self):
        belief = self.factor_product
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
    def __init__(self, clusters: List[Cluster]):
        self.clusters = clusters

    def propagate_beliefs_round_robin(self, nTimes):
        for _ in range(nTimes):
            for cluster in self.clusters:
                for edge in cluster.edges:
                    cluster.send_message(edge)


