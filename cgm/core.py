# -*- coding: utf-8 -*-
"""



"""
import numpy as np
from typing import List

class Variable:
    """A variable has a name and can taken on a finite number of states. 
    """
    def __init__(self, name: str, nStates: int):
        self.name = name
        self.nStates = nStates
    def __repr__(self):
        return self.name
    def __lt__(self, other):
        return self.name < other.name


class DAG_Node(Variable):
    """A DAG (Directed Acyclic Graph) node is a variable in a Bayesian Network. 
    A node can have multiple parents and multiple children, but no cycles can be
    created. A node is associated with a single conditional probability 
    distribution (CPD), which is a distribution over the variable given its 
    parents. If the node has no parents, this CPD is a distribution over all the
    states of the variable. 

    """
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


class Factor:
    """A factor is a function that has a list of variables in its scope, and 
    maps every combination of variable values to a real number. In this 
    implementation the mapping is stored as a np.ndarray. For example, if this
    factor's scope is the variables {A, B, C}, and each of these is a binary 
    variable, then to access the value of the factor for [A=1, B=0, C=1], the 
    entry can be accessed at self.factor[1, 0, 1]. If the ndarray isn't 
    specified, a random one will be created. 

    The scope of a factor must is sorted by the name of the variables. All 
    variables must hav unique names. 

    Factors ϕ1 and ϕ2 can be multiplied and divided by ϕ1 * ϕ2 and ϕ1 / ϕ2. 
    A factor can be marginalized over a subset of its scope. For example, to 
    marginalize out variables A and B, call ϕ.marinalize([A, B]).
    """
    def __init__(self, scope: List[Variable], factor: np.ndarray = None):
        self.scope = scope
        if factor is None:
            self.factor = self.makeFactor()
        else:
            self.factor = factor
        self._check_input()
    
    @classmethod
    def getNull(Factor):
        return Factor(scope=[], factor=np.float64(1))

    def _check_input(self):
        # all variable names have to be unique
        assert len(set([s.name for s in self.scope])) == len(self.scope)
        # all variable names must be in order
        assert sorted(self.scope) == self.scope
        # size of scope much match nDims of factor
        assert len(self.scope) == len(self.factor.shape)
        

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
    
    def marginalize(self, variables: List[Variable]):
        """ 
        Sum over all possible states of a list of variables
        example: phi3.marginalize([A, B]) 
        """
        axes = tuple(np.where([s in variables for s in self.scope])[0])
        reduced_scope = [s for s in self.scope if s not in variables]
        return Factor(reduced_scope, np.sum(self.factor, axis=axes))
    
    def normalize(self):
        """Returns a factor with the same distribution whose sum is 1"""
        factor = (self / self.marginalize(self.scope)).factor
        return Factor(self.scope, factor)

class CPD(Factor):
    """This is a type of factor with additional constraints. One variable in its
    scope is the child node, the others are the parents. The CPD must sum to 1
    for every particular value of the child node. Additionally, the CPD cannot
    introduce cycles in the DAG.
    """
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
    
    def normalize(self):
        msg = "CPD has no method 'normalize', since it is already normalized."
        raise AttributeError(msg)

class DAG():
    """ Contains a list of DAG_Nodes. The information about connectivity is
    stored at each node.
    """
    def __init__(self, nodes: List[DAG_Node]):
        self.nodes = sorted(nodes)
        
    def __repr__(self):
        s = ''
        for n in self.nodes:
            parents = sorted(list(n.parents))
            s += f"{n} <- {parents}\n"
        return s
