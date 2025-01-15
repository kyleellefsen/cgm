# -*- coding: utf-8 -*-
"""
The core module contains the basic building blocks of a Causal Graphical Model.
"""
from typing import List, Sequence, TypeVar, Generic, Protocol, FrozenSet
import collections
import dataclasses
import numpy as np
from . import _utils

DCovariant = TypeVar('DCovariant', bound='ComparableHasParents', covariant=True)
D = TypeVar('D', bound='ComparableHasParents')

class HasComparison(Protocol):
    def __lt__(self: 'HasComparison', other: 'HasComparison') -> bool: ...

class HasParents(Protocol[DCovariant]):
    @property
    def parents(self) -> FrozenSet[DCovariant]: ...
    @property 
    def ancestors(self) -> FrozenSet[DCovariant]: ...

class ComparableHasParents(HasParents[DCovariant], HasComparison, Protocol):
    pass

class HasVariable(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def num_states(self) -> int: ...
    def __lt__(self, other) -> bool: ...

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
    _parents: frozenset[D] = dataclasses.field(default_factory=frozenset)
    _ancestors: frozenset[D] = dataclasses.field(init=False)

    def __init__(self, variable: Variable, _parents: frozenset[D] = frozenset()):
        # Create new variable instance
        new_variable = Variable(name=variable.name, num_states=variable.num_states)
        object.__setattr__(self, 'variable', new_variable)
        object.__setattr__(self, '_parents', _parents)

        # Compute ancestors
        parents_remaining = _parents.copy()
        ancestors_acc = set()
        while parents_remaining:
            node = next(iter(parents_remaining))
            parents_remaining = parents_remaining - {node}
            ancestors_acc.add(node)
            ancestors_acc.update(node.ancestors)
            parents_remaining = parents_remaining - ancestors_acc
        object.__setattr__(self, '_ancestors', frozenset(ancestors_acc))

    @property
    def name(self) -> str:
        """Return the name of the variable."""
        return self.variable.name

    def __repr__(self) -> str:
        return f"{self.name}"

    @property
    def num_states(self) -> int:
        """Return the number of states the variable can take on."""
        return self.variable.num_states

    def __lt__(self, other) -> bool:
        return self.variable.__lt__(other.variable)

    def __eq__(self, other) -> bool:
        if not isinstance(other, DAG_Node):
            return NotImplemented
        return (self.variable == other.variable and self.parents == other.parents)

    def __hash__(self) -> int:
        return hash((self.variable, frozenset(self.parents)))

    @property
    def parents(self) -> frozenset[D]:  # Now matches protocol with concrete type D
        return self._parents

    @property
    def ancestors(self) -> frozenset[D]:  # Now matches protocol with concrete type D
        return self._ancestors

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

        A = cgm.cgm.CG_Node.from_params('A', 2)
        B = cgm.cgm.CG_Node.from_params('B', 2)
        C = cgm.cgm.CG_Node.from_params('C', 2)
        phi1 = cgm.CPD(A, [B])
        phi2 = cgm.CPD(B, [C])
        phi3 = cgm.CPD(C, [])


    """
    variable: Variable
    dag_node: DAG_Node['CG_Node']
    cpd: 'CPD'

    @classmethod
    def from_params(cls, name: str, num_states: int) -> 'CG_Node':
        """Create a new CG_Node with default CPD."""
        variable = Variable(name=name, num_states=num_states)
        dag_node = DAG_Node[CG_Node](variable=variable)
        # Use object.__new__ to create instance without calling __init__
        # This allows us to manually set fields even though class is frozen
        temp_node = cls.__new__(cls)
        object.__setattr__(temp_node, 'variable', variable)
        object.__setattr__(temp_node, 'dag_node', dag_node)
        object.__setattr__(temp_node, 'cpd', None)  # Add this line to prevent attribute error
        # Create default CPD using the temporary node
        cpd = CPD([temp_node])
        # Create the real node with all attributes
        return cls(variable=variable, dag_node=dag_node, cpd=cpd)

    @property
    def parents(self) -> frozenset[HasParents]:
        return self.dag_node.parents

    @property
    def ancestors(self) -> frozenset[HasParents]:
        return self.dag_node.ancestors

    @property
    def name(self) -> str:
        return self.variable.name

    def __repr__(self) -> str:
        return f"{self.name}"

    @property
    def num_states(self) -> int:
        return self.variable.num_states

    def __lt__(self, other) -> bool:
        return self.variable.__lt__(other.variable)

    def __eq__(self, other) -> bool:
        if not isinstance(other, CG_Node):
            return NotImplemented
        # For basic node equality, just compare the variables
        return self.variable == other.variable

    def __hash__(self) -> int:
        # Hash should be consistent with equality
        return hash(self.variable)

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
      A = cgm.cgm.CG_Node.from_params('A', 2)
      B = cgm.cgm.CG_Node.from_params('B', 2)
      C = cgm.cgm.CG_Node.from_params('C', 2)
      phi1 = cgm.CPD(A, [B])
      phi2 = cgm.CPD(B, [C])
      phi3 = cgm.CPD(C, [])
    ```

    """

    def __init__(self,
                 scope: Sequence[CG_Node],
                 values: np.ndarray | None = None,
                 child: CG_Node | None = None,
                 rng: np.random.Generator | None = None):
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
              
        """
        super().__init__(scope, values, rng)
        if child is None:
            child = scope[0]
        self._child = child
        self._parents = frozenset(s for s in scope if s != child)
        self._assert_nocycles()
        # Create new child node with updated parent relationships
        self._child = CG_Node(
            variable=child.variable,
            dag_node=DAG_Node[CG_Node](
                variable=child.variable,
                _parents=self._parents
            ),
            cpd=self
        )
        self._normalize()

    @property
    def child(self) -> CG_Node:
        """Return the child node of the CPD."""
        return self._child

    @property
    def parents(self) -> frozenset[CG_Node]:
        """Return the parents of the CPD."""
        return self._parents

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
        np.testing.assert_allclose(margin, np.ones_like(margin))

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
        return CPD(scope=new_scope, values=cond_values, child=self.child)

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

@_utils.set_module('cgm')
class DAG(Generic[D]):
    """Directed Acyclic Graph"""

    def __init__(self, nodes: Sequence[D]):
        self.nodes = sorted(nodes)

    def __repr__(self):
        s = ''
        for n in self.nodes:
            parents = sorted(list(n.parents))
            s += f"{n} ← {parents}\n"
        return s

@_utils.set_module('cgm')
class CG(DAG[CG_Node]):
    """Causal Graph
    Contains a list of CG_Nodes. The information about connectivity is stored 
    at each node.
    """

    def __init__(self, nodes: list[CG_Node]):
        super().__init__(nodes)


del _utils