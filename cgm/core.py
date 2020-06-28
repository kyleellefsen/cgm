import numpy as np
from typing import List

class Variable:
    def __init__(self, name: str, nStates: int):
        self.name = name
        self.nStates = nStates
    def __repr__(self):
        return self.name
    def __lt__(self, other):
        return self.name < other.name

class Factor:
    def __init__(self, scope: List[Variable], factor=None):
        self.scope = scope
        self._check_input()
        if factor is None:
            self.factor = self.makeFactor()
        else:
            self.factor = factor

    def _check_input(self):
        # all variable names have to be unique
        assert len(set([s.name for s in self.scope])) == len(self.scope)
        # all variable names must be in order
        assert sorted(self.scope) == self.scope

    def __repr__(self):
        return "ϕ(" + ", ".join([f"{s}" for s in self.scope]) + ")"
            
    def makeFactor(self):
        nDims = tuple(s.nStates for s in self.scope)
        return np.random.uniform(size=nDims)
    
    def __mul__(self, other: 'Factor'):
        """ 
        Factor product as defined in PGM Definition 4.2 (Koller 2009)
        """
        scope1 = self.scope
        scope2 = other.scope
        scope_union = sorted(list(set(scope1).union(scope2)))
        dims2insert1 = np.where([s not in scope1 for s in scope_union])[0]
        dims2insert2 = np.where([s not in scope2 for s in scope_union])[0]
        aa = self.factor
        for i in dims2insert1:
            aa = np.expand_dims(aa, i)
        bb = other.factor
        for i in dims2insert2:
            bb = np.expand_dims(bb, i)
        return Factor(scope_union, np.multiply(aa, bb))
    
    def __truediv__(self, other: 'Factor'):
        scope1 = self.scope
        scope2 = other.scope
        scope_intersection = sorted(list(set(scope1).intersection(scope2)))
        # The scope of the denominator must be a subset of that of the numerator
        assert scope_intersection == scope2  
        dims2insert = np.where([s not in scope2 for s in scope1])[0]
        bb = other.factor
        for i in dims2insert:
            bb = np.expand_dims(bb, i)
        return Factor(scope1, np.divide(self.factor, bb))
    
    def marginalize(self, variables: list):
        """ 
        Sum over all possible states of a list of variables
        example: phi3.marginalize([A, B]) 
        """
        axes = tuple(np.where([s in variables for s in self.scope])[0])
        reduced_scope = [s for s in self.scope if s not in variables]
        return Factor(reduced_scope, np.sum(self.factor, axis=axes))

class DAG_Node(Variable):
    def __init__(self, name: str, nStates: int):
        super().__init__(name, nStates)
        # by default the cpd has no parents; the node is unconnected
        self.update_cpd(CPD(self))
    
    def update_cpd(self, cpd):
        self.cpd = cpd
        self.parents = set(self.cpd.scope) - set([self])
    
    def get_ancestors(self):
        parents_remaining = self.parents.copy()
        ancestors = set()
        while len(parents_remaining) > 0:
            node = parents_remaining.pop()
            ancestors.add(node)
            ancestors.update(node.get_ancestors())
            parents_remaining = parents_remaining - ancestors
        return ancestors

class CPD(Factor):
    def __init__(self, child: DAG_Node, parents: List[DAG_Node]=[], factor=None):
        scope = sorted(list(set([child] + parents)))
        super().__init__(scope, factor)
        self.set_child(child)
        self._normalize()
    
    def set_child(self, child):
        self.child = child
        self._nocycles()
        child.update_cpd(self)

    def _nocycles(self):
        child = self.child
        parents = set(self.scope) - set([child])
        if len(parents) == 0:
            return
        ancestors = set.union(*[p.get_ancestors() for p in parents])
        assert child not in ancestors
    
    def _normalize(self):
        # Normalize so it is a distribution that sums to 1
        self.factor = (self / self.marginalize([self.child])).factor
        margin = self.marginalize([self.child]).factor
        np.testing.assert_allclose(margin, np.ones_like(margin))

class DAG():
    def __init__(self, nodes: List[DAG_Node]):
        self.nodes = sorted(nodes)
        
    def __repr__(self):
        s = ''
        for n in self.nodes:
            parents = sorted(list(n.parents))
            s += f"{n} <- {parents}\n"
        return s
    
    def get_scope_of_factors_after_variable_elimination(self):
        nodes = self.nodes
        return {n_target : set.union(*[set(n.cpd.scope) for n in nodes if n_target in n.cpd.scope]) \
                - set([n_target]) for n_target in nodes}
    
    def eliminate_variable(self, node: DAG_Node):
        pass

