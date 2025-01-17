"""Module for serializing and deserializing objects in `cgm.core`.

Currently this only supports conversion of `cgm.CG` objects to and from a 
numpy-based representation based off of jax's `jraph` library.
"""

import dataclasses
from typing import Optional, Dict, Any
import numpy as np
import cgm

@dataclasses.dataclass(frozen=True)
class GraphTuple:
    """Array-based graph tuple representation inspired by jraph."""
    # Core graph structure
    nodes: np.ndarray  # [num_nodes, node_feat_dim]
    senders: np.ndarray  # [num_edges]
    receivers: np.ndarray  # [num_edges]
    # Node features
    node_types: np.ndarray  # [num_nodes] - categorical type per node
    node_states: np.ndarray  # [num_nodes] - number of states per node
    # CPD arrays - packed into a single contiguous array
    cpd_values: np.ndarray  # [total_cpd_entries] - flattened CPD probability values
    cpd_offsets: np.ndarray  # [num_nodes + 1] - start/end indices for each node's CPD
    cpd_shapes: np.ndarray  # [num_nodes, max_parents + 1] - shape of each CPD tensor
    cpd_parents: np.ndarray  # [num_nodes, max_parents] - parent indices for each node
    cpd_var_order: np.ndarray  # [num_nodes, max_parents + 1] - variable ordering for each CPD
    # Optional attributes
    edge_attrs: Optional[Dict[str, np.ndarray]] = None
    globals: Optional[Dict[str, Any]] = None

    @classmethod
    def from_cg(cls, cg: cgm.CG) -> 'GraphTuple':
        """Convert a CG (Causal Graph) to array-based GraphTuple format."""
        nodes = sorted(cg.nodes)
        num_nodes = len(nodes)
        node_index = {n: i for i, n in enumerate(nodes)}

        # Build edge arrays from actual edges in the graph
        edges = [(node_index[p], node_index[n])
                for n in nodes 
                for p in n.parents]
        senders = np.array([s for s, _ in edges], dtype=np.int32)
        receivers = np.array([r for _, r in edges], dtype=np.int32)

        # Node features
        node_types = np.zeros(num_nodes)  # Can be extended for different node types
        node_states = np.array([n.num_states for n in nodes], dtype=np.int32)

        # Calculate max parents for CPD array allocation
        max_parents = max(len(n.parents) for n in nodes)

        # Initialize CPD arrays
        total_cpd_entries = sum(np.prod([n.num_states] + 
                              [p.num_states for p in n.parents])
                              for n in nodes)

        cpd_values = np.zeros(total_cpd_entries)
        cpd_offsets = np.zeros(num_nodes + 1, dtype=np.int32)
        cpd_shapes = np.zeros((num_nodes, max_parents + 1), dtype=np.int32)
        cpd_parents = np.full((num_nodes, max_parents), -1, dtype=np.int32)
        cpd_var_order = np.full((num_nodes, max_parents + 1), -1, dtype=np.int32)

        # Pack CPDs into arrays
        current_offset = 0
        for i, node in enumerate(nodes):
            cpd = node.cpd
            if cpd is None:
                continue

            # Get CPD shape and values according to original scope order
            scope_order = list(cpd.scope)  # Preserve original variable ordering
            cpd_shape = [s.num_states for s in scope_order]
            num_entries = int(np.prod(cpd_shape))

            # Store values
            cpd_values[current_offset:current_offset + num_entries] = cpd.values.flatten()

            # Store metadata
            cpd_offsets[i] = current_offset
            cpd_offsets[i + 1] = current_offset + num_entries

            # Store shape (padded to max_parents + 1)
            cpd_shapes[i, :len(cpd_shape)] = cpd_shape

            # Store variable ordering (indices in the node list)
            var_indices = [node_index[v] for v in scope_order]
            cpd_var_order[i, :len(var_indices)] = var_indices

            # Store parent indices (padded to max_parents)
            parent_indices = [node_index[p] for p in node.parents]
            cpd_parents[i, :len(parent_indices)] = parent_indices

            current_offset += num_entries

        return cls(
            nodes=np.zeros((num_nodes, 0)),  # Placeholder for node features
            senders=senders,
            receivers=receivers,
            node_types=node_types,
            node_states=node_states,
            cpd_values=cpd_values,
            cpd_offsets=cpd_offsets,
            cpd_shapes=cpd_shapes,
            cpd_parents=cpd_parents,
            cpd_var_order=cpd_var_order
        )

    def get_node_cpd(self, node_idx: int) -> np.ndarray:
        """Retrieve CPD tensor for a given node."""
        start_idx = self.cpd_offsets[node_idx]
        end_idx = self.cpd_offsets[node_idx + 1]
        cpd_values = self.cpd_values[start_idx:end_idx]

        # Get actual shape (remove padding)
        shape = []
        var_order = []
        for s, v in zip(self.cpd_shapes[node_idx], self.cpd_var_order[node_idx]):
            if s == 0:
                break
            shape.append(s)
            var_order.append(v)

        # Reshape according to original shape
        reshaped = cpd_values.reshape(shape)

        # Return values in original CPD order by transposing if needed
        return reshaped

    def to_cg(self) -> cgm.CG:
        """Convert array-based GraphTuple back to CG format."""
        # Implementation would create CG nodes and reconstruct CPDs
        # from the array representation
        raise NotImplementedError("Conversion back to CG not yet implemented")
