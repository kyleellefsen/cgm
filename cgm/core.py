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
    """A DAG (Directed Acyclic Graph) node is a variable in a DAG. 
    A node can have multiple parents and multiple children, but no cycles can be
    created.
    """
    def __init__(self, name: str, nStates: int):
        super().__init__(name, nStates)
        self.parents = set()
    
    def get_ancestors(self):
        parents_remaining = self.parents.copy()
        ancestors = set()
        while len(parents_remaining) > 0:
            node = parents_remaining.pop()
            ancestors.add(node)
            ancestors.update(node.get_ancestors())
            parents_remaining = parents_remaining - ancestors
        return ancestors


class CG_Node(DAG_Node):
    """A CG (Causal Graph) node is a variable in a Bayesian Network. 
    A node is associated with a single conditional probability 
    distribution (CPD), which is a distribution over the variable given its 
    parents. If the node has no parents, this CPD is a distribution over all the
    states of the variable. 

    """
    def __init__(self, name: str, nStates: int):
        super().__init__(name, nStates)
        # by default the cpd has no parents; the node is unconnected
        self.setCpd(CPD(self))
    
    def setCpd(self, cpd):
        self.cpd = cpd
        self.parents = set(self.cpd.scope) - set([self])


class Factor:
    """A factor is a function that has a list of variables in its scope, and 
    maps every combination of variable values to a real number. In this 
    implementation the mapping is stored as a np.ndarray. For example, if this
    factor's scope is the variables {A, B, C}, and each of these is a binary 
    variable, then to access the value of the factor for [A=1, B=0, C=1], the 
    entry can be accessed at self.getValues()[1, 0, 1]. If the ndarray isn't 
    specified, a random one will be created. 

    The scope of a factor must is sorted by the name of the variables. All 
    variables must have unique names. 

    Factors ϕ1 and ϕ2 can be multiplied and divided by ϕ1 * ϕ2 and ϕ1 / ϕ2. 
    A factor can be marginalized over a subset of its scope. For example, to 
    marginalize out variables A and B, call ϕ.marginalize([A, B]).
    """
    def __init__(self, scope: List[Variable], values: np.ndarray = None):
        self.scope = scope
        if values is None:
            self._values = self._getRandomValues()
        else:
            self._values = values    
        self._check_input()

    @classmethod
    def getNull(Factor):
        return Factor(scope=[], values=np.float64(1))
    
    def getValues(self):
        return self._values

    def setValues(self, values: np.ndarray):
        # the dimension of the factor cannot be changed using this method
        assert self._values.shape == values.shape
        self._values = values
        self._check_input()

    def randomizeValues(self):
        self._values = self._getRandomValues()
        self._check_input()

    def _check_input(self):
        # all variable names have to be unique
        assert len(set([s.name for s in self.scope])) == len(self.scope)
        # all variable names must be in order
        assert sorted(self.scope) == self.scope
        # size of scope much match nDims of factor
        assert len(self.scope) == len(self._values.shape)
        

    def __repr__(self):
        return "ϕ(" + ", ".join([f"{s}" for s in self.scope]) + ")"
            
    def _getRandomValues(self):
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
        aa = self._values
        for i in dims2insert1:
            aa = np.expand_dims(aa, i)
        bb = other._values
        for i in dims2insert2:
            bb = np.expand_dims(bb, i)
        return Factor(scope_union, np.multiply(aa, bb))
    
    def __truediv__(self, other: 'Factor'):
        scope1 = self.scope
        if isinstance(other, (int, float)):
            return Factor(scope1, np.divide(self._values, other))
        scope2 = other.scope
        scope_intersection = sorted(list(set(scope1).intersection(scope2)))
        # The scope of the denominator must be a subset of that of the numerator
        assert scope_intersection == scope2  
        dims2insert = np.where([s not in scope2 for s in scope1])[0]
        bb = other._values
        for i in dims2insert:
            bb = np.expand_dims(bb, i)
        return Factor(scope1, np.divide(self._values, bb))
    
    def marginalize(self, variables: List[Variable]):
        """ 
        Sum over all possible states of a list of variables
        example: phi3.marginalize([A, B]) 
        """
        axes = tuple(np.where([s in variables for s in self.scope])[0])
        reduced_scope = [s for s in self.scope if s not in variables]
        return Factor(reduced_scope, np.sum(self._values, axis=axes))
    
    def normalize(self):
        """Returns a factor with the same distribution whose sum is 1"""
        return Factor(self.scope, (self / self.marginalize(self.scope))._values)
    
    def condition(self, values: dict):
        """
        Condition on a set of variables at particular values of those variables.
        values is a dictionary where each key is a variable to condition on and
        the value is an integer representing the index to condition on. 

        The scope of the returned factor will exclude all the variables 
        conditioned on. 
        """
        raise NotImplementedError  # implementation needs to be added
    
    def increment_at_index(self, index: tuple, amount):
        self._values[index] += amount


class CPD(Factor):
    """This is a type of factor with additional constraints. One variable in its
    scope is the child node, the others are the parents. The CPD must sum to 1
    for every particular value of the child node. Additionally, the CPD cannot
    introduce cycles in the DAG.
    """
    def __init__(self, child: CG_Node, parents: List[CG_Node]=[], values=None):
        scope = sorted(list(set([child] + parents)))
        super().__init__(scope, values)
        self.set_child(child)
        self._normalize()
    
    def set_child(self, child):
        self.child = child
        self._nocycles()
        child.setCpd(self)

    def _nocycles(self):
        child = self.child
        parents = set(self.scope) - set([child])
        if len(parents) == 0:
            return
        ancestors = set.union(*[p.get_ancestors() for p in parents])
        assert child not in ancestors
    
    def _normalize(self):
        # Normalize so it is a distribution that sums to 1
        self._values = (self / self.marginalize([self.child]))._values
        margin = self.marginalize([self.child])._values
        np.testing.assert_allclose(margin, np.ones_like(margin))
    
    def normalize(self):
        msg = "CPD has no method 'normalize', since it is already normalized."
        raise AttributeError(msg)

    def sample(self, parent_states: dict = {}, nSamples: int = 1):
        parents = set(self.scope) - set([self.child])
        assert parents == set(parent_states.keys())
        index = []
        for var in self.scope:
            if var in parents:
                index.append(parent_states[var])
            else:
                index.append(slice(None))
        index = tuple(index)
        dist = self._values[index]
        np.testing.assert_almost_equal(dist.sum(), 1.0)
        samples = np.random.choice(a=len(dist), size=nSamples, p=dist)
        return samples

    def randomizeValues(self):
        super().randomizeValues()
        self._normalize()
    
    def setValues(self, values):
        super().setValues(values)
        self._normalize()



class DAG:
    def __init__(self, nodes: List[DAG_Node]):
        self.nodes = sorted(nodes)

    def __repr__(self):
        s = ''
        for n in self.nodes:
            parents = sorted(list(n.parents))
            s += f"{n} <- {parents}\n"
        return s


class CG(DAG):
    """ Causal Graph
    Contains a list of CG_Nodes. The information about connectivity is stored 
    at each node.
    """
    def __init__(self, nodes: List[CG_Node]):
        super().__init__(nodes)
        

