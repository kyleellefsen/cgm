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
from typing import Optional


class InvalidForwardSamplingError(Exception):
    """Raised when attempting to forward sample with invalid conditions.
    
    This error indicates that forward sampling cannot proceed because some
    conditioned nodes have unconditioned ancestors. Forward sampling requires
    all ancestors of conditioned nodes to also be conditioned.
    """
    pass


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
        schema: Optional pre-computed schema (will be created if None)
        state: Optional GraphState with conditions to respect
        certificate: Optional certificate validating the state is suitable for
            forward sampling. If state is provided but certificate is None,
            validation will be performed.
        
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
            
        # Get parent values using schema indexing
        parent_states = {}
        for parent in node.parents:
            parent_idx = schema.var_to_idx[parent.name]
            parent_states[parent] = sample[parent_idx]
        
        # Get node's CPD and sample
        cpd = cg.get_cpd(node)
        if cpd is None:
            raise ValueError(f"Node {node.name} has no CPD")
            
        conditioned_node = cpd.condition(parent_states)
        node_sample, key = conditioned_node.sample(1, key)
        sample[node_idx] = node_sample.item()
        
    return sample, key


def get_n_samples(
    cg: cgm.CG,
    key: np.random.Generator,
    num_samples: int,
    state: cgm.GraphState | None = None,
    certificate: Optional[ForwardSamplingCertificate] = None
) -> tuple[cgm.Factor[cgm.CG_Node], np.random.Generator]:
    """Generate multiple samples from a graphical model.
    
    Args:
        cg: The graphical model to sample from
        key: Random number generator key
        num_samples: Number of samples to generate
        state: Optional GraphState with conditions to respect
        certificate: Optional certificate validating the state is suitable for
            forward sampling. If state is provided but certificate is None,
            validation will be performed.
        
    Returns:
        Tuple of:
        - Factor containing counts of all generated samples
        - New random key for subsequent operations
        
    Raises:
        InvalidForwardSamplingError: If state is provided without a valid certificate
            and validation fails
    """
    # Initialize
    schema = cgm.GraphSchema.from_network(cg)
    
    # Pre-allocate array for all samples
    samples_array = np.full((num_samples, schema.num_vars), -1, dtype=np.int32)
    
    # Generate all samples
    for i in range(num_samples):
        samples_array[i], key = forward_sample(cg, key, schema, state, certificate)
    
    # Convert to factor format
    scope = cg.nodes  # Already in topological order
    num_dims = tuple(n.num_states for n in scope)
    samples_factor = cgm.Factor[cgm.CG_Node](scope, np.zeros(num_dims, dtype=int))
    
    # Count occurrences
    for i in range(num_samples):
        idx = tuple(samples_array[i, schema.var_to_idx[n.name]] for n in samples_factor.scope)
        samples_factor.increment_at_index(idx, 1)
        
    return samples_factor, key


def get_marginal_samples(
    cg: cgm.CG,
    key: np.random.Generator,
    nodes: set[cgm.CG_Node],
    num_samples: int,
    state: cgm.GraphState | None = None,
    certificate: Optional[ForwardSamplingCertificate] = None
) -> tuple[cgm.Factor[cgm.CG_Node], np.random.Generator]:
    """Generate samples and compute their marginal distribution over specified nodes.
    
    Args:
        cg: The graphical model to sample from
        key: Random number generator key
        nodes: Set of nodes to compute marginal for
        num_samples: Number of samples to generate
        state: Optional GraphState with conditions to respect
        certificate: Optional certificate validating the state is suitable for
            forward sampling. If state is provided but certificate is None,
            validation will be performed.
        
    Returns:
        Tuple of:
        - Factor containing marginal counts over specified nodes
        - New random key for subsequent operations
        
    Raises:
        InvalidForwardSamplingError: If state is provided without a valid certificate
            and validation fails
    """
    samples, key = get_n_samples(cg, key, num_samples, state, certificate)
    return samples.marginalize(list(set(cg.nodes) - nodes)), key
