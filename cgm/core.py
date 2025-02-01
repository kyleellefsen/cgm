# -*- coding: utf-8 -*-
"""
The core module contains the basic building blocks of a Causal Graphical Model.

Typical usage:

```python
    g = cgm.CG()
    A = g.node('A', 2)
    B = g.node('B', 2)
    C = g.node('C', 2)
    phi1 = g.P(A | B) # if no values are provided, the CPD is random
    phi2 = g.P(B | C, values=np.array([[0.1, 0.9], [0.3, 0.7]]))
    theta = g.P(C | B, virtual=True) # virtual nodes are not added to the graph
```

"""
from typing import List, Sequence, TypeVar, Generic, Protocol, FrozenSet, Optional
import functools
import collections
from collections import OrderedDict
import dataclasses
import numpy as np
import numpy.typing as npt
from . import _utils
from . import _format


DCovariant = TypeVar('DCovariant', bound='ComparableHasParents', covariant=True)
D = TypeVar('D', bound='ComparableHasParents')

class HasComparison(Protocol):
    def __lt__(self: 'HasComparison', other: 'HasComparison') -> bool: ...

class HasParents(Protocol[DCovariant]):
    @property
    def parents(self) -> FrozenSet[DCovariant]: ...
    @property 
    def ancestors(self) -> FrozenSet[DCovariant]: ...

class ComparableHasParents(HasParents[DCovariant], HasComparison, Protocol[DCovariant]):
    pass

class HasVariable(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def num_states(self) -> int: ...
    def __lt__(self, other: 'HasVariable') -> bool: ...

V = TypeVar('V', bound='HasVariable')

@_utils.set_module('cgm')
@dataclasses.dataclass(frozen=True)
class Variable(HasVariable):
    """A variable has a name and can taken on a finite number of states."""
    _name: str  # The name of the variable
    _num_states: int  # The number of states the variable can take on

    def __init__(self, name: str, num_states: int):
        object.__setattr__(self, '_name', name)
        object.__setattr__(self, '_num_states', num_states)

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_states(self) -> int:
        return self._num_states

    def __repr__(self) -> str:
        return self.name

    def __lt__(self, other) -> bool:
        return self.name < other.name

    def __eq__(self, other) -> bool:
        if not isinstance(other, Variable):
            return NotImplemented
        return (self.name == other.name and self.num_states == other.num_states)

    def __hash__(self) -> int:
        return hash((self.name, self.num_states))

@_utils.set_module('cgm')
@dataclasses.dataclass(frozen=True)
class DAG_Node(HasParents[D], HasVariable, Generic[D]):
    """A DAG (Directed Acyclic Graph) node is a variable in a DAG. 
    A node can have multiple parents and multiple children, but no cycles can be
    created.
    """
    variable: Variable
    dag: 'DAG[D]'

    def __post_init__(self):
        self.dag.add_node(self, parents=set(), replace=False)

    @property
    def name(self) -> str:
        """Return the name of the variable."""
        return self.variable.name

    @property
    def parents(self) -> FrozenSet:
        return self.dag.get_parents(self)

    @property
    def ancestors(self) -> FrozenSet:
        return self.dag.get_ancestors(self)

    @property
    def num_states(self) -> int:
        """Return the number of states the variable can take on."""
        return self.variable.num_states

    def __repr__(self) -> str:
        return f"{self.name}"

    def __lt__(self, other) -> bool:
        return self.variable.__lt__(other.variable)

    def __eq__(self, other) -> bool:
        if not isinstance(other, DAG_Node):
            return NotImplemented
        return self.variable == other.variable and self.dag == other.dag

    def __hash__(self) -> int:
        return hash((self.variable, self.dag))


@dataclasses.dataclass(frozen=True)
class CPDSpec:
    """Helper class to hold specification of a CPD created using | operator"""
    child: 'CG_Node'
    parents: list['CG_Node']

@_utils.set_module('cgm')
@dataclasses.dataclass(frozen=True)
class CG_Node(HasParents, HasVariable):
    """A Causal Graph Node

    A CG_Node is a variable in a Bayesian Network. 
    A node is associated with a single conditional probability 
    distribution (CPD), which is a distribution over the variable given its 
    parents. If the node has no parents, this CPD is a distribution over all the
    states of the variable. 

    Example:
        g = cgm.CG()
        A = g.node('A', 2)
        B = g.node('B', 2)
        C = g.node('C', 2)
        phi1 = g.P(A | B)
        phi2 = g.P(B | C)
        phi3 = g.P(C)
    """
    dag_node: DAG_Node['CG_Node']
    cg: 'CG'

    @classmethod
    def from_params(cls, name: str, num_states: int, cg: 'CG') -> 'CG_Node':
        """Create a new CG_Node with default CPD."""
        variable = Variable(name=name, num_states=num_states)
        dag_node = DAG_Node(variable=variable, dag=cg.dag)
        node = cls(dag_node=dag_node, cg=cg)
        CPD(scope=[node])  # Creates default CPD
        return node

    @property
    def parents(self) -> frozenset['CG_Node']:
        return frozenset(CG_Node(dag_node=p, cg=self.cg) for p in self.dag_node.parents)

    @property
    def ancestors(self) -> frozenset['CG_Node']:
        return frozenset(CG_Node(dag_node=a, cg=self.cg) for a in self.dag_node.ancestors)

    @property
    def variable(self) -> Variable:
        return self.dag_node.variable

    @property
    def name(self) -> str:
        return self.variable.name

    @property
    def num_states(self) -> int:
        return self.variable.num_states

    @property
    def dag(self) -> 'DAG[CG_Node]':
        return self.dag_node.dag

    @property
    def cpd(self) -> Optional['CPD']:
        """Get the CPD associated with this node."""
        return self.cg.get_cpd(self)

    def __repr__(self) -> str:
        return f"{self.name}"

    def __lt__(self, other) -> bool:
        return self.variable.__lt__(other.variable)

    def __eq__(self, other) -> bool:
        if not isinstance(other, CG_Node):
            return NotImplemented
        return self.variable == other.variable and self.dag == other.dag

    def __hash__(self) -> int:
        return hash((self.variable, self.dag))

    def __or__(self, parents) -> CPDSpec:
        """Enable syntax like A | [B, C] for CPD creation"""
        if isinstance(parents, CG_Node):
            parents = [parents]
        return CPDSpec(child=self, parents=parents)

@_utils.set_module('cgm')
class ScopeShapeMismatchError(Exception):
    """Exception raised when the shape of a factor's scope does not match 
    the shape of its stored values array."""
    def __init__(self, expected_shape, actual_shape):
        message = f"Expected shape {expected_shape}, but got {actual_shape}. " \
                   "The factor's scope must correspond to the shape of the " \
                   "stored values array."
        super().__init__(message)

@_utils.set_module('cgm')
class NonUniqueVariableNamesError(Exception):
    """Exception raised when the variables in a factor's scope do not have unique names."""
    def __init__(self, non_unique_names):
        message = "Variables in the scope have non-unique names: "\
                  f"{non_unique_names}. All variables in the scope must have "\
                   "unique names."
        super().__init__(message)




@_utils.set_module('cgm')
class Factor(Generic[V]):
    """A factor is a function that has a list of variables in its scope, and 
    maps every combination of variable values to a real number. In this 
    implementation the mapping is stored as a np.ndarray. For example, if this
    factor's scope is the variables {A, B, C}, and each of these is a binary 
    variable, then to access the value of the factor for [A=1, B=0, C=1], the 
    entry can be accessed at self.values[1, 0, 1]. If the ndarray isn't 
    specified, a random one will be created.

    All variables in the scope must have unique names. 

    Factors ϕ1 and ϕ2 can be multiplied and divided by ϕ1 * ϕ2 and ϕ1 / ϕ2. 
    A factor can be marginalized over a subset of its scope. For example, to 
    marginalize out variables A and B, call ϕ.marginalize([A, B]).

    Example:
    
        A = cgm.Variable('A', 2)
        B = cgm.Variable('B', 2)
        C = cgm.Variable('C', 2)
        phi1 = cgm.Factor([A, B, C])
        phi2 = cgm.Factor([B, C])
        phi3 = cgm.Factor([B, C])
        phi1 * phi2
        phi1 / phi2
        phi1.marginalize([A, B])

    Args:
        scope: A list of variables that are in the scope of the factor.
        values: The values of the factor. If None, random values will be 
            generated. If a scalar, all values will be set to that scalar.
        rng: A numpy random number generator. Only used if values is None.
    """

    def __init__(self,
                 scope: Sequence[V],
                 values: np.ndarray | int | float | None = None,
                 rng: np.random.Generator | None = None):
        self._scope: tuple[V, ...] = tuple(scope)
        if values is None:
            if rng is None:
                rng = np.random.default_rng()
            self._values = self._get_random_values(rng)
        elif isinstance(values, (int, float)):
            self._values = np.full(tuple(s.num_states for s in self.scope), values)
        else:
            self._values = values
        self._check_input()

    @classmethod
    def get_null(cls):
        """Return a factor with no scope and a single value of 1.0."""
        return cls[V](scope=tuple(), values=np.float64(1))

    @property
    def values(self) -> np.ndarray:
        """Return the values of the factor."""
        return self._values

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the factor's values array."""
        return tuple(s.num_states for s in self.scope)

    @property
    def scope(self) -> tuple[V, ...]:
        """Return the scope of the factor."""
        return self._scope

    def permute_scope(self, new_scope: Sequence[V]) -> 'Factor':
        """Set the scope of the factor according to the specified permutation.
        
        Must be a permutation of the original scope."""
        assert set(new_scope) == set(self.scope)
        dst = [new_scope.index(s) for s in self.scope]
        new_vals = np.moveaxis(self.values,
                               source=range(len(new_scope)),
                               destination=dst)
        return Factor(new_scope, new_vals.copy())

    def set_scope(self, new_scope: Sequence[V]) -> 'Factor':
        """Set the scope of the factor to the specified scope."""
        return Factor(new_scope, self.values.copy())

    def _check_input(self):
        # all variable names have to be unique
        var_counts = collections.Counter([el.name for el in self.scope])
        if any([v > 1 for v in var_counts.values()]):
            non_unique_names = [k for k, v in var_counts.items() if v > 1]
            raise NonUniqueVariableNamesError(non_unique_names)
        assert len({s.name for s in self.scope}) == len(self.scope)

        # shape of scope much match shape of factor
        scope_shape = np.array([s.num_states for s in self.scope])
        if not np.array_equal(self.values.shape, scope_shape):
            raise ScopeShapeMismatchError(expected_shape=scope_shape,
                                          actual_shape=self.values.shape)

    def __repr__(self):
        return "ϕ(" + ", ".join([f"{s}" for s in self.scope]) + ")"

    @property
    def table(self) -> _format.FactorTableView:
        """Access the factor's table representation.
        
        Returns:
            FactorTableView object that can be used either as a property (for default view)
            or as a method (for custom views)
        """
        return _format.FactorTableView(self)

    def _get_random_values(self, rng: np.random.Generator):
        num_dimensions = tuple(s.num_states for s in self.scope)
        return rng.uniform(size=num_dimensions)

    def _normalize_dimensions(self, other: 'Factor') -> tuple[np.ndarray, np.ndarray, tuple[V, ...]]:
        """Expand and permute the dimensions of the two factors to match.
        
        This is required for factor multiplication, division, addition, 
        and subtraction.
        """
        scope1 = self.scope
        scope2 = other.scope
        scope_2_but_not_1 = tuple(sc for sc in scope2 if sc not in scope1)
        result_scope = scope1 + scope_2_but_not_1
        scope2_padded = list(scope2) + [sc for sc in scope1 if sc not in scope2]
        arr1 = np.expand_dims(self.values, axis=tuple(range(len(scope1), len(result_scope))))
        arr2 = np.expand_dims(other.values, axis=tuple(range(len(scope2), len(result_scope))))
        destination_mapping = {value: index for index, value in enumerate(result_scope)}
        arr2_dst = [destination_mapping[element] for element in scope2_padded]
        arr2 = np.moveaxis(arr2, source=range(len(result_scope)), destination=arr2_dst)
        return arr1, arr2, result_scope

    def __mul__(self, other: 'Factor | int | float') -> 'Factor':
        """Factor product as defined in PGM Definition 4.2 (Koller 2009)."""
        if isinstance(other, (int, float)):
            return Factor(self.scope, np.multiply(self.values, other))
        arr1, arr2, result_scope = self._normalize_dimensions(other)
        return Factor(result_scope, np.multiply(arr1, arr2))

    def __rmul__(self, other: int | float) -> 'Factor':
        return Factor(self.scope, np.multiply(self.values, other))

    def __add__(self, other: 'Factor | int | float') -> 'Factor':
        if isinstance(other, (int, float)):
            return Factor(self.scope, np.add(self.values, other))
        arr1, arr2, result_scope = self._normalize_dimensions(other)
        return Factor(result_scope, np.add(arr1, arr2))

    def __radd__(self, other: int | float) -> 'Factor':
        return Factor(self.scope, np.add(self.values, other))

    def __sub__(self, other: 'Factor | int | float') -> 'Factor':
        if isinstance(other, (int, float)):
            return Factor(self.scope, np.subtract(self.values, other))
        arr1, arr2, result_scope = self._normalize_dimensions(other)
        return Factor(result_scope, np.subtract(arr1, arr2))

    def __truediv__(self, other: 'Factor') -> 'Factor':
        scope1 = self.scope
        if isinstance(other, (int, float)):
            return Factor(scope1, np.divide(self.values, other))
        scope2 = other.scope
        # The scope of the denominator must be a subset of that of the numerator
        assert set(scope1).intersection(scope2) == set(scope2)
        result_scope = scope1
        scope2_padded = list(scope2) + [sc for sc in scope1 if sc not in scope2]
        arr1 = self.values
        arr2 = np.expand_dims(other.values, axis=tuple(range(len(scope2), len(result_scope))))
        destination_mapping = {value: index for index, value in enumerate(result_scope)}
        arr2_dst = [destination_mapping[element] for element in scope2_padded]
        arr2 = np.moveaxis(arr2, source=range(len(result_scope)), destination=arr2_dst)
        return Factor(result_scope, np.divide(arr1, arr2))

    def marginalize(self, variables: List[V]) -> 'Factor[V]':
        """ 
        Sum over all possible states of a list of variables
        example: phi3.marginalize([A, B]) 
        """
        axes = tuple(np.where([s in variables for s in self.scope])[0])
        reduced_scope = [s for s in self.scope if s not in variables]
        return Factor(reduced_scope, np.sum(self.values, axis=axes))

    def marginalize_cpd(self, cpd: 'CPD') -> 'Factor':
        """Marginalize out a conditional probability distribution.

        Sum over all possible states of a set of the cpd variables, weighted
        by how probable the c is.

        Example:

            X = cgm.cgm.CG_Node.from_params('X', 2)
            Y = cgm.cgm.CG_Node.from_params('Y', 2)
            phi1 = cgm.Factor([X, Y])
            cpd = cgm.CPD(Y, [X])
            phi2 = phi1.marginalize_cpd(cpd)
            print(phi2)
            # ϕ(X)
        """
        summand_var: CG_Node = cpd.child  # This variable will be eliminated
        assert set(self.scope) == set(cpd.scope)
        prod: 'Factor[CG_Node]' = self * cpd
        summand = prod.marginalize([summand_var])
        return summand

    def max(self, variable: V) -> 'Factor':
        """ 
        Returns the maximum along the  the state of the variables that maximizes 
        the factor. 
        example: phi3.max(A) 
        """
        axis = self.scope.index(variable)
        reduced_scope = [s for s in self.scope if s != variable]
        max_indices = np.max(self.values, axis=axis)
        return Factor[V](reduced_scope, max_indices)

    def argmax(self, variable: V) -> 'Factor':
        """ 
        Find the state of the variables that maximizes the factor
        example: phi3.argmax(A) 
        """
        axis = self.scope.index(variable)
        reduced_scope = [s for s in self.scope if s != variable]
        max_indices = np.argmax(self.values, axis=axis)
        return Factor[V](reduced_scope, max_indices)

    def abs(self) -> 'Factor':
        """Returns the absolute value of the factor."""
        return Factor[V](self.scope, np.abs(self.values))

    def normalize(self) -> 'Factor':
        """Returns a new factor with the same distribution whose sum is 1."""
        return Factor(self.scope, (self / self.marginalize(list(self.scope))).values)

    def increment_at_index(self, index: tuple[int, ...], amount) -> None:
        """Increment the value of the factor at a particular index by amount."""
        self._values[index] += amount

    def condition(self, condition_dict: dict[V, int]) -> 'Factor':
        """Condition on a set of variables.

        Condition on a set of variables at particular values of those variables.
        condition_dict is a dictionary where each key is a variable to condition 
        on and the value is an integer representing the index to condition on. 

        The scope of the returned factor will exclude all the variables 
        conditioned on. 
        """

        specified_vars = set(condition_dict.keys())
        new_scope = [v for v in self.scope if v not in specified_vars]
        assert specified_vars.issubset(set(self.scope))
        index: list[int | slice] = []
        for var in self.scope:
            if var in specified_vars:
                index.append(condition_dict[var])
            else:
                index.append(slice(None))
        index_tuple = tuple(index)
        cond_values = self.values[index_tuple]
        return Factor(list(new_scope), cond_values)





@_utils.set_module('cgm')
class CPD(Factor[CG_Node]):
    """Conditional Probability Distribution

    This is a type of factor with additional constraints. One variable in its
    scope is the child node, the others are the parents. The CPD must sum to 1
    for every particular value of the child node. Additionally, the CPD cannot
    introduce cycles in the DAG.

    Example:
    ```
      g = cgm.CG()
      A = g.node('A', 2)
      B = g.node('B', 2)
      C = g.node('C', 2)
      phi1 = g.P(A | B)
      phi2 = g.P(B | C)
      phi3 = g.P(C)
      print(g)
    ```

    """

    def __init__(self,
                 scope: Sequence[CG_Node],
                 values: np.ndarray | None = None,
                 child: CG_Node | None = None,
                 rng: np.random.Generator | None = None,
                 virtual: bool = False):
        """Create a conditional probability distribution.
        
        Args:
            scope: The scope of the CPD. The scope sets the order of the
              dimensions in the underlying array.
            values: The values of the CPD. If None, random values will be
              generated.
            child: The child node of the CPD. If child is None, the first
              variable in the scope is assumed to be the child.
            rng: A numpy random number generator used to set the values.
              Only used if values is None.
            virtual: If True, the CPD is not added to the DAG. This is useful
                for creating derived CPDs
              
        """

        super().__init__(scope, values, rng=rng)
        if child is None:
            child = scope[0]
        object.__setattr__(self, '_child', child)
        parents = frozenset(v for v in scope if v != child)
        object.__setattr__(self, '_parents', parents)
        
        if not virtual:
            # Add node to DAG *before* checking cycles
            dag: DAG[CG_Node] = child.dag_node.dag
            dag.add_node(child.dag_node, parents=self._parents, replace=True)            
            # Register the CPD with the CG
            child.cg.set_cpd(child, self)
            # Only check cycles for non-virtual CPDs
            self._assert_nocycles()
        
        self._normalize()

    @property
    def child(self) -> CG_Node:
        """Return the child node of the CPD."""
        return self._child

    @property
    def parents(self) -> frozenset[CG_Node]:
        """Return the parents of the CPD."""
        return self._parents

    @property
    def dag(self) -> 'DAG[CG_Node]':
        """Return the DAG that the CPD is associated with."""
        return self.child.dag_node.dag

    def _assert_nocycles(self):
        child = self.child
        parents = set(self.scope) - {child}
        if len(parents) == 0:
            return
        ancestors = set().union(*(set(p.ancestors) for p in parents))
        assert child not in ancestors

    def _normalize(self):
        # Normalize so it is a distribution that sums to 1
        normalized_values = (self / self.marginalize([self.child])).values
        self._values = normalized_values
        margin = self.marginalize([self.child]).values
        np.testing.assert_allclose(margin, np.ones_like(margin), rtol=1e-10)

    def normalize(self):
        msg = "CPD has no method 'normalize', since it is already normalized."
        raise AttributeError(msg)

    def sample(self,
               num_samples: int,
               rng: np.random.Generator) -> tuple[np.ndarray,
                                                  np.random.Generator]:
        """Sample from the distribution"""
        samples = rng.choice(a=len(self.values),
                             size=num_samples,
                             p=self.values)
        return samples, rng

    def condition(self, condition_dict: dict[CG_Node, int]) -> 'CPD':
        """Condition on a set of variables.

        Condition on a set of variables at particular values of those variables.
        condition_dict is a dictionary where each key is a variable to condition 
        on and the value is an integer representing the index to condition on. 

        The scope of the returned factor will exclude all the variables 
        conditioned on. 
        """
        specified_parents = set(condition_dict.keys())
        assert specified_parents.issubset(set(self.parents))
        index: list[int | slice] = []
        new_scope = [v for v in self.scope if v not in specified_parents]
        for var in self.scope:
            if var in specified_parents:
                index.append(condition_dict[var])
            else:
                index.append(slice(None))
        index_tuple = tuple(index)
        cond_values = self.values[index_tuple]
        # Create new conditioned CPD without modifying original node's CPD
        return CPD(scope=new_scope, values=cond_values, child=self.child, virtual=True)

    def marginalize_cpd(self, cpd: 'CPD') -> 'CPD':
        """Marginalize out a distribution over a parent variable.

        Sum over all possible states of a set of parent variables, weighted
        by how probable the parent is.
        """
        summand_var = cpd.child  # This variable will be eliminated
        assert cpd.child in self.parents
        assert set(self.parents) == set(cpd.scope)
        prod: Factor[CG_Node] = self * cpd
        summand = prod.marginalize([summand_var])
        new_scope = [v for v in self.scope if v != summand_var]
        return CPD(scope=new_scope, values=summand.values, child=self.child)

    def set_scope(self, new_scope: Sequence[CG_Node]) -> 'CPD':
        """Set the scope of the factor to the specified scope."""
        child_idx = self.scope.index(self.child)
        new_child = new_scope[child_idx]
        return CPD(scope=new_scope, child=new_child, values=self.values.copy())

    def permute_scope(self, new_scope: Sequence[CG_Node]) -> 'CPD':
        """Set the scope of the factor according to the specified permutation.
        
        Must be a permutation of the original scope.
        """
        assert set(new_scope) == set(self.scope)
        dst_map = {value: idx for idx, value in enumerate(new_scope)}
        dst = [dst_map[element] for element in self.scope]
        new_vals = np.moveaxis(self.values,
                               source=range(len(self.scope)),
                               destination=dst)
        return CPD(scope=new_scope, values=new_vals.copy(), child=self.child)

    def __repr__(self):
        parents = tuple(p for p in self.scope if p != self.child)
        if len(parents) > 0:
            return f"𝑃({self.child} | {', '.join(str(s) for s in parents)})"
        return f"𝑃({self.child})"

    @property
    def table(self) -> _format.FactorTableView:
        """Access the CPD's table representation."""


        return _format.CPDTableView(self)

@_utils.set_module('cgm')
class DAG(Generic[D]):
    """Mutable Directed Acyclic Graph."""

    def __init__(self, nodes: Sequence[D | None] | None = None):
        if nodes is None:
            nodes = []
        self._parent_dict: OrderedDict[DAG_Node[D], FrozenSet[DAG_Node[D]]] = OrderedDict()
        filtered_nodes = [n for n in nodes if n is not None]
        for node in filtered_nodes:
            self.add_node(node, frozenset(), replace=False)

    @property
    def nodes(self) -> list[DAG_Node[D]]:
        """Return nodes in topological order.
        
        The nodes are returned in a deterministic topological order, where each node
        appears after all of its parents. This ordering is cached and recomputed only
        when the graph structure changes.
        """
        return self._topological_order

    @functools.cached_property
    def _topological_order(self) -> list[DAG_Node[D]]:
        """Return nodes in topological order. This is important for sampling.
        
        Uses Kahn's algorithm to compute a deterministic topological ordering.
        The ordering is cached and recomputed only when the graph structure changes.
        
        Returns:
            List of nodes in topological order (parents before children)
        
        Raises:
            ValueError: If the graph contains cycles
        """
        # Make a copy of the parent dictionary and count incoming edges
        in_degree = {node: len(parents) for node, parents in self._parent_dict.items()}
        ready = [node for node, degree in in_degree.items() if degree == 0]
        
        # Sort ready nodes by name for deterministic ordering of tied nodes
        ready.sort(key=lambda x: x.name)
        
        result = []
        while ready:
            node = ready.pop(0)
            result.append(node)
            
            # Find children of this node (nodes where this is a parent)
            children = [child for child, parents in self._parent_dict.items() 
                       if node in parents]
            
            for child in children:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    ready.append(child)
            # Sort ready nodes by name for deterministic ordering of tied nodes
            ready.sort(key=lambda x: x.name)
        
        if len(result) != len(self._parent_dict):
            raise ValueError("Graph contains cycles")
            
        return result

    def get_parents(self, node: DAG_Node[D]) -> frozenset[DAG_Node[D]]:
        """Return the parents of a node."""
        return self._parent_dict.get(node, frozenset())

    def add_node(
        self,
        node: DAG_Node[D] | D,  # Allow both DAG_Node and CG_Node
        parents: FrozenSet[DAG_Node[D] | D] | set[DAG_Node[D] | D],
        replace: bool = False
    ) -> None:
        """Add a node to the graph."""
        # Convert CG_Node to its dag_node if needed
        node_to_add = node.dag_node if hasattr(node, 'dag_node') else node
        # Convert parent CG_Nodes to their dag_nodes
        parents_frozen = frozenset(
            p.dag_node if hasattr(p, 'dag_node') else p
            for p in parents
        )
        # Clear cached properties
        if hasattr(self, '_ancestor_dict'):
            delattr(self, '_ancestor_dict')
        if hasattr(self, '_topological_order'):
            delattr(self, '_topological_order')
        # First ensure node exists in parent dict
        if node_to_add not in self._parent_dict:
            self._parent_dict[node_to_add] = frozenset()
        # Then add parents if they don't exist
        for parent in parents_frozen:
            if parent not in self._parent_dict:
                self._parent_dict[parent] = frozenset()
        if not replace and self._parent_dict[node_to_add] != frozenset():
            if self._parent_dict[node_to_add] == parents_frozen:
                return
            raise ValueError(f"Node {node_to_add} already exists with different parents")
        self._parent_dict[node_to_add] = parents_frozen

    @functools.cached_property
    def _ancestor_dict(self) -> dict[DAG_Node[D], frozenset[DAG_Node[D]]]:
        """Return a dictionary of ancestors for each node."""
        ancestors: dict[DAG_Node[D], set[DAG_Node[D]]] = {node: set() for node in self._parent_dict}
        for node in self._parent_dict:
            current_parents = self._parent_dict[node]
            stack = list(current_parents)
            visited = set()
            while stack:
                parent = stack.pop()
                if parent not in visited:
                    visited.add(parent)
                    ancestors[node].add(parent)
                    stack.extend(self._parent_dict[parent])
                    
        return {node: frozenset(ancs) for node, ancs in ancestors.items()}

    def get_ancestors(self, node: DAG_Node[D]) -> frozenset[DAG_Node[D]]:
        """Return the ancestors of a node."""
        if node not in self._parent_dict:
            return frozenset()
        return self._ancestor_dict[node]

    def __repr__(self):
        s = ''
        for n in self.nodes:
            parents = sorted(list(n.parents))
            s += f"{n} ← {parents}\n"
        return s

@_utils.set_module('cgm')
@dataclasses.dataclass(frozen=True)
class CG:
    """Causal Graph
    Contains a list of CG_Nodes. The information about connectivity is stored 
    in the DAG. 
    """
    dag: DAG[CG_Node] = dataclasses.field(default_factory=DAG[CG_Node])
    _cpd_dict: dict[CG_Node, CPD] = dataclasses.field(default_factory=dict)

    def get_cpd(self, node: CG_Node) -> CPD | None:
        """Get the CPD associated with a node."""
        return self._cpd_dict.get(node)

    def set_cpd(self, node: CG_Node, cpd: CPD) -> None:
        """Associate a CPD with a node."""
        self._cpd_dict[node] = cpd

    def node(self, name: str, num_states: int) -> CG_Node:
        """Create a new node and return it.
        
        Args:
            name: Name of the node. Must be unique in the graph.
            num_states: Number of states for this node.
            
        Raises:
            ValueError: If a node with the same name already exists.
        """
        # Check if node with this name already exists
        existing_names = {node.name for node in self.nodes}
        if name in existing_names:
            raise ValueError(f"Node with name '{name}' already exists in the graph. Node names must be unique.")
        return CG_Node.from_params(name, num_states, self)

    @property
    def nodes(self) -> list[CG_Node]:
        """Returns the list of CG_Nodes in the graph.
        
        While the underlying DAG stores DAG_Node objects, this property reconstructs 
        and returns the original CG_Node objects. The list is guaranteed to be in
        topological order (parents before children). For nodes at the same level in
        the graph (no parent-child relationship), the order is determined by node names
        for determinism.
        """
        return [CG_Node(dag_node=node, cg=self) for node in self.dag.nodes]

    def __repr__(self):
        return repr(self.dag)


    def P(self,  # pylint: disable=invalid-name
          spec_or_node: CPDSpec | CG_Node,
          values: np.ndarray | None = None,
          virtual: bool = False,
          **kwargs) -> CPD:
        """Create a CPD using probability notation.
        
        Args:
            spec_or_node: Either a CPDSpec from the | operator or a single node for priors
            values: Optional values for the CPD
            virtual: If True, create a virtual CPD that doesn't modify graph structure
            **kwargs: Additional arguments passed to CPD constructor
        """
        if isinstance(spec_or_node, CPDSpec):
            # Handle A | [B, C] case
            scope = [spec_or_node.child] + spec_or_node.parents
            return CPD(scope=scope, child=spec_or_node.child, values=values, 
                      virtual=virtual, **kwargs)
        else:
            # Handle prior case - single node
            return CPD(scope=[spec_or_node], child=spec_or_node, values=values,
                      virtual=virtual, **kwargs)


del _utils


@dataclasses.dataclass(frozen=True)
class GraphSchema:
    """Schema defining the structure of data for a Causal Graph (Bayesian Network)
    
    This class creates a fixed mapping between network variables and array 
    indices, enabling efficient data access without dictionary lookups.
    The schema is computed once when loading the dataset and reused for all
    operations.

    Each variable in the schema has a fixed number of states, numbered from 
    0 to num_states-1. This allows -1 to be safely used as a sentinel value 
    for missing/unset data.
    
    The mapping from variables to indices follows the graph's topological order,
    ensuring that parent nodes always have lower indices than their children.
    This property can be useful for sampling and inference algorithms.
    """
    # Maps each variable name to its position in the data arrays
    var_to_idx: dict[str, int]
    # Maps each variable name to its number of possible states
    var_to_states: dict[str, int]
    # Total number of variables in the schema
    num_vars: int

    @classmethod
    def from_network(cls, network: CG) -> 'GraphSchema':
        """Creates a schema from a network, establishing a fixed variable order.
        
        The variables are ordered according to the graph's topological structure,
        with parent nodes receiving lower indices than their children. For nodes
        at the same level (no parent-child relationship), ordering is determined
        by node names for determinism.
        """
        # Get nodes in topological order
        nodes = network.nodes  # nodes are now in topological order
        var_to_idx = {node.name: idx for idx, node in enumerate(nodes)}
        var_to_states = {node.name: node.num_states for node in nodes}
        return cls(
            var_to_idx=var_to_idx,
            var_to_states=var_to_states,
            num_vars=len(nodes)
        )

    def validate_states(self, data: npt.NDArray[np.number]) -> None:
        """Validates that all values in data are within valid state ranges.
        
        Valid states are non-negative integers from 0 to num_states-1.
        The value -1 is allowed as a sentinel for missing/unset data.
        """
        for var_name, num_states in self.var_to_states.items():
            idx = self.var_to_idx[var_name]
            if not np.all((data[idx] < num_states) | (data[idx] == -1)):
                raise ValueError(
                    f"Invalid states for variable {var_name}. "
                    f"Must be < {num_states}"
                )


@dataclasses.dataclass(frozen=True)
class GraphState:
    """Graph state for a Causal Graph (Bayesian Network)
    
    Since the graph is immutable, to condition on a sample, we create a new
    GraphState which contains the information about the current state.

    Each variable in the state can take on values from 0 to num_states-1,
    with -1 representing a missing/unset value. The mask array indicates 
    which variables have been set (True) vs unset (False).
    """
    network: CG
    schema: GraphSchema
    # Store conditions as a single array and mask for efficiency
    # Array contains values, mask indicates which variables are conditioned
    # We use -1 as a sentinel value for missing/unset values
    _values: npt.NDArray[np.number]
    _mask: npt.NDArray[np.bool_]

    @property
    def values(self) -> npt.NDArray[np.number]:
        """Get the current values array. -1 indicates missing/unset values."""
        return self._values

    @property
    def mask(self) -> npt.NDArray[np.bool_]:
        """Get the current mask array. True indicates set values."""
        return self._mask

    @classmethod
    def create(cls, network: CG) -> 'GraphState':
        """Creates a new GraphState with no conditions."""
        schema = GraphSchema.from_network(network)
        return cls(
            network=network,
            schema=schema,
            _values=np.full(schema.num_vars, -1, dtype=np.float_),
            _mask=np.zeros(schema.num_vars, dtype=np.bool_)
        )

    def condition_on_sample(
        self,
        sample: npt.NDArray[np.number]
    ) -> 'GraphState':
        """Creates a new state by conditioning on a sample.
        
        Args:
            sample: Array of shape (num_vars,) containing observed values.
                   Values should be integers from 0 to num_states-1.
                   Use -1 for missing/unset values.
                   
        Returns:
            New GraphState with additional conditions from sample.
        """
        # Create mask for valid (non-missing) values in sample
        sample_mask = sample != -1

        # Combine existing conditions with new sample
        new_mask = self._mask | sample_mask
        new_values = np.where(
            sample_mask,
            sample,
            self._values
        )

        return GraphState(
            network=self.network,
            schema=self.schema,
            _values=new_values,
            _mask=new_mask
        )
