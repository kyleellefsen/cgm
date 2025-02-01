"""This module provides functions for performing forward sampling on graphical models.

The main sampling functions are:
- forward_sample: Generate a single sample from a graph
- get_n_samples: Generate multiple samples and return as a Factor
- get_conditioned_samples: Generate samples respecting a GraphState's conditions

All functions are pure and maintain immutability of inputs. Random state is 
explicitly passed and split following JAX conventions for reproducibility.
"""
import numpy as np
import cgm


def forward_sample(
    cg: cgm.CG,
    key: np.random.Generator,
    schema: cgm.GraphSchema | None = None,
    state: cgm.GraphState | None = None
) -> tuple[np.ndarray, np.random.Generator]:
    """Generate a single sample from a graphical model.
    
    Args:
        cg: The graphical model to sample from
        key: Random number generator key
        schema: Optional pre-computed schema (will be created if None)
        state: Optional GraphState with conditions to respect
        
    Returns:
        Tuple of:
        - Array of sampled values, indexed by schema
        - New random key for subsequent operations
    """
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
) -> tuple[cgm.Factor[cgm.CG_Node], np.random.Generator]:
    """Generate multiple samples from a graphical model.
    
    Args:
        cg: The graphical model to sample from
        key: Random number generator key
        num_samples: Number of samples to generate
        state: Optional GraphState with conditions to respect
        
    Returns:
        Tuple of:
        - Factor containing counts of all generated samples
        - New random key for subsequent operations
    """
    # Initialize
    schema = cgm.GraphSchema.from_network(cg)
    
    # Pre-allocate array for all samples
    samples_array = np.full((num_samples, schema.num_vars), -1, dtype=np.int32)
    
    # Generate all samples
    for i in range(num_samples):
        samples_array[i], key = forward_sample(cg, key, schema, state)
    
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
) -> tuple[cgm.Factor[cgm.CG_Node], np.random.Generator]:
    """Generate samples and compute their marginal distribution over specified nodes.
    
    Args:
        cg: The graphical model to sample from
        key: Random number generator key
        nodes: Set of nodes to compute marginal for
        num_samples: Number of samples to generate
        state: Optional GraphState with conditions to respect
        
    Returns:
        Tuple of:
        - Factor containing marginal counts over specified nodes
        - New random key for subsequent operations
    """
    samples, key = get_n_samples(cg, key, num_samples, state)
    return samples.marginalize(list(set(cg.nodes) - nodes)), key
