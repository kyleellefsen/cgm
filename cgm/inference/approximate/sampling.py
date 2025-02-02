"""This module provides functions for performing forward sampling on graphical models.

The main sampling functions are:
- forward_sample: Generate a single sample from a graph
- get_n_samples: Generate multiple samples and return as a Factor
- get_conditioned_samples: Generate samples respecting a GraphState's conditions

All functions are pure and maintain immutability of inputs. Random state is 
explicitly passed and split following JAX conventions for reproducibility.
"""
import dataclasses
import numpy as np
import cgm
from typing import Optional, Union
import typing


class InvalidForwardSamplingError(Exception):
    """Raised when attempting to forward sample with invalid conditions.
    
    This error indicates that forward sampling cannot proceed because some
    conditioned nodes have unconditioned ancestors. Forward sampling requires
    all ancestors of conditioned nodes to also be conditioned.
    """
    pass


@dataclasses.dataclass(frozen=True)
class SampleArray:
    """High-performance storage for samples from a graphical model.
    
    Stores samples in a simple (num_samples, num_vars) array where each column
    corresponds to a variable in the graph's schema. Similar to JAX DeviceArray
    and PyTorch Tensor, this is a minimal container focused on efficient storage
    and basic operations.
    
    The samples array is immutable (frozen) to maintain functional purity and
    enable future optimizations like parallelization and JIT compilation.
    
    Args:
        values: (num_samples, num_vars) array of samples
        schema: GraphSchema defining the variable ordering
    """
    values: np.ndarray  # Shape: (num_samples, num_vars) 
    schema: cgm.GraphSchema

    def __post_init__(self):
        if self.values.ndim != 2:
            raise ValueError("Samples must be a 2D array")
        if self.values.shape[1] != self.schema.num_vars:
            raise ValueError(
                f"Sample array has {self.values.shape[1]} variables but schema "
                f"has {self.schema.num_vars}"
            )


    @property
    def n(self) -> int:
        """Return the number of samples in the SampleArray."""
        return self.values.shape[0]

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the samples array."""
        return self.values.shape

    def to_factor(self) -> cgm.Factor[cgm.CG_Node]:
        """Convert samples to a Factor for compatibility with existing code.
        
        Uses the actual nodes from the schema instead of creating temporary ones,
        ensuring that node identity is preserved for operations like marginalization.
        """
        # Initialize factor with zeros using the actual nodes from schema
        factor = cgm.Factor[cgm.CG_Node](self.schema.nodes, values=0)
        
        # Count samples into factor
        for i in range(self.shape[0]):
            # Get sample and ensure it's a tuple of ints for indexing
            sample_row: np.ndarray = self.values[i]
            index_tuple: tuple[int, ...] = tuple(map(int, sample_row))
            factor.increment_at_index(index_tuple, 1)
            
        return factor


@dataclasses.dataclass(frozen=True)
class ForwardSamplingCertificate:
    """Certificate that validates a graph state is suitable for forward sampling.
    
    This certificate ensures that any conditioned nodes in the graph state do not
    have unconditioned ancestors. This is required for forward sampling to be valid,
    since we must sample nodes in topological order.
    
    The certificate can be passed to sampling functions that require this validation.
    
    Example:
        >>> cg = cgm.example_graphs.get_chain_graph(3)  # A->B->C
        >>> state = cg.condition(B=1)  # Condition on B without conditioning A
        >>> try:
        ...     certificate = ForwardSamplingCertificate(state)
        ... except InvalidForwardSamplingError as e:
        ...     print(e)  # Node B is conditioned but its ancestor A is not...
    
    Raises:
        InvalidForwardSamplingError: If any conditioned node has an unconditioned
            ancestor. The error message will specify which node is conditioned and
            which ancestor is not.
    """
    _state: cgm.GraphState

    def __init__(self, state: cgm.GraphState):
        """Create a certificate by validating the graph state.
        
        Args:
            state: The graph state to validate
            
        Raises:
            InvalidForwardSamplingError: If the state has conditioned nodes with
                unconditioned ancestors.
        """
        self._validate_state(state)
        # Use object.__setattr__ since the dataclass is frozen
        object.__setattr__(self, '_state', state)
    
    @property
    def state(self) -> cgm.GraphState:
        """Get the validated graph state."""
        return self._state
    
    def _validate_state(self, state: cgm.GraphState) -> None:
        """Check that no conditioned nodes have unconditioned ancestors.
        
        Args:
            state: The graph state to validate
            
        Raises:
            InvalidForwardSamplingError: If validation fails
        """
        # Get all conditioned nodes
        conditioned_nodes = []
        for node in state.network.nodes:
            idx = state.schema.var_to_idx[node.name]
            if state.mask[idx]:
                conditioned_nodes.append(node)
                
        # Check ancestors of each conditioned node
        for node in conditioned_nodes:
            ancestors = node.ancestors
            for ancestor in ancestors:
                idx = state.schema.var_to_idx[ancestor.name]
                if not state.mask[idx]:
                    raise InvalidForwardSamplingError(
                        f"Node {node.name} is conditioned but its ancestor "
                        f"{ancestor.name} is not conditioned. Forward sampling "
                        "requires all ancestors of conditioned nodes to also be "
                        "conditioned."
                    )


def forward_sample(
    cg: cgm.CG,
    key: np.random.Generator,
    schema: cgm.GraphSchema | None = None,
    state: cgm.GraphState | None = None,
    certificate: Optional[ForwardSamplingCertificate] = None
) -> tuple[np.ndarray, np.random.Generator]:
    """Generate a single sample from a graphical model.
    
    Args:
        cg: The graphical model to sample from
        key: Random number generator key
        schema: Optional GraphSchema defining variable ordering
        state: Optional GraphState with conditions to respect
        certificate: Optional certificate validating the state
        
    Returns:
        Tuple of:
        - Array of sampled values, indexed by schema
        - New random key for subsequent operations
        
    Raises:
        InvalidForwardSamplingError: If state is provided without a valid certificate
            and validation fails
    """
    # Validate state if provided without certificate
    if state is not None and certificate is None:
        certificate = ForwardSamplingCertificate(state)
    elif state is not None and certificate is not None and certificate.state != state:
        raise ValueError("Certificate does not match provided state")
        
    # Initialize schema if not provided
    if schema is None:
        schema = cgm.GraphSchema.from_network(cg)
        
    # Pre-allocate array for this sample
    sample = np.full(schema.num_vars, -1, dtype=np.int32)
    
    # If we have conditions, apply them first
    if state is not None:
        sample[state.mask] = state.values[state.mask]
    
    # Sample each unconditioned node in topological order
    for node in cg.nodes:  # Already in topological order
        node_idx = schema.var_to_idx[node.name]
        
        # Skip if this node is conditioned
        if state is not None and state.mask[node_idx]:
            continue
            
        # Get parent values
        parent_states = {}
        for parent in node.parents:
            parent_idx = schema.var_to_idx[parent.name]
            parent_states[parent] = sample[parent_idx]
        
        # Get node's CPD and sample
        cpd = cg.get_cpd(node)
        if cpd is None:
            raise ValueError(f"Node {node.name} has no CPD")
            
        # Get conditional distribution for these parent values
        if parent_states:
            cpd = cpd.condition(parent_states)
            
        # Sample from the (conditional) distribution
        node_sample, key = cpd.sample(1, key)
        sample[node_idx] = node_sample.item()
        
    return sample, key


def get_n_samples(
    cg: cgm.CG,
    key: np.random.Generator,
    num_samples: int,
    state: cgm.GraphState | None = None,
    certificate: Optional[ForwardSamplingCertificate] = None,
    return_array: bool = False
) -> tuple[Union[cgm.Factor[cgm.CG_Node], SampleArray], np.random.Generator]:
    """Generate multiple samples from a graphical model.
    
    Args:
        cg: The graphical model to sample from
        key: Random number generator key
        num_samples: Number of samples to generate
        state: Optional GraphState with conditions to respect
        certificate: Optional certificate validating the state
        return_array: If True, return a SampleArray instead of Factor
        
    Returns:
        Tuple of:
        - Either Factor containing counts or SampleArray containing raw samples
        - New random key for subsequent operations
    """
    # Initialize schema
    schema = cgm.GraphSchema.from_network(cg)
    
    # Pre-allocate array for all samples
    samples_array = np.full((num_samples, schema.num_vars), -1, dtype=np.int32)
    
    # Generate all samples
    for i in range(num_samples):
        samples_array[i], key = forward_sample(cg, key, schema, state, certificate)
    
    # Create sample array
    samples = SampleArray(samples_array, schema)
    
    if return_array:
        return samples, key
    else:
        return samples.to_factor(), key


def get_marginal_samples(
    cg: cgm.CG,
    key: np.random.Generator,
    nodes: set[cgm.CG_Node],
    num_samples: int,
    state: cgm.GraphState | None = None,
    certificate: Optional[ForwardSamplingCertificate] = None,
    return_array: bool = False
) -> tuple[Union[cgm.Factor[cgm.CG_Node], SampleArray], np.random.Generator]:
    """Generate samples and compute their marginal distribution over specified nodes.
    
    Args:
        cg: The graphical model to sample from
        key: Random number generator key
        nodes: Set of nodes to compute marginal for
        num_samples: Number of samples to generate
        state: Optional GraphState with conditions to respect
        certificate: Optional certificate validating the state
        return_array: If True, return a SampleArray instead of Factor
        
    Returns:
        Tuple of:
        - Either Factor containing marginal counts or SampleArray containing raw samples
        - New random key for subsequent operations
    """
    # Always get samples as array first
    result, key = get_n_samples(cg, key, num_samples, state, certificate, return_array=True)
    samples = typing.cast(SampleArray, result)  # We know this is true because return_array=True
    
    if return_array:
        # For array return, we need to keep only the columns corresponding to the requested nodes
        node_indices = [samples.schema.var_to_idx[node.name] for node in nodes]
        marginal_values = samples.values[:, node_indices]
        
        # Create new schema with only the requested nodes
        marginal_schema = cgm.GraphSchema(
            var_to_idx={node.name: i for i, node in enumerate(nodes)},
            var_to_states={node.name: node.num_states for node in nodes},
            num_vars=len(nodes),
            nodes=list(nodes)  # Convert set to list for schema
        )
        return SampleArray(marginal_values, marginal_schema), key
    else:
        # For factor return, use existing marginalization
        factor = samples.to_factor()
        return factor.marginalize(list(set(cg.nodes) - nodes)), key
