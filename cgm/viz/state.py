"""State management for the visualization module."""
from typing import Optional, Dict, List, Any, Literal
from dataclasses import dataclass, field
import time
import numpy as np

import cgm
from ..inference.approximate import sampling

@dataclass
class VizState:
    """Global visualization state with convenient property accessors."""
    _current_graph: Optional[cgm.CG] = None
    _current_graph_state: Optional[cgm.GraphState] = None
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())
    _current_seed: Optional[int] = None
    _current_samples: Optional[sampling.SampleArray] = None
    _samples_metadata: Dict[str, Any] = field(default_factory=lambda: {
        'seed': None,
        'timestamp': None,
        'num_samples': 0,
        'conditions': {}
    })
    
    def set_seed(self, seed: Optional[int] = None) -> int:
        """Set the random seed and return the seed used."""
        if seed is None:
            seed = int(time.time() * 1000) & 0xFFFFFFFF
        
        self._current_seed = seed
        self._rng = np.random.default_rng(seed)
        return seed
    
    def clear_samples(self):
        """Clear the current samples and metadata."""
        self._current_samples = None
        self._samples_metadata = {
            'seed': None,
            'timestamp': None,
            'num_samples': 0,
            'conditions': {}
        }
    
    def set_current_graph(self, graph: cgm.CG, graph_state: Optional[cgm.GraphState] = None) -> None:
        """Set the current graph and optionally its state.
        
        Args:
            graph: The graph to visualize
            graph_state: Optional state of the graph (e.g., conditioned values)
        """
        self._current_graph = graph
        # Create a new graph state if none provided
        self._current_graph_state = graph_state if graph_state is not None else cgm.GraphState.create(graph)
        self.clear_samples()  # Clear any existing samples as they're no longer valid
    
    def get_node_samples(self, node_name: str) -> Optional[List[int]]:
        """Get samples for a specific node from the cached samples.
        
        Args:
            node_name: Name of the node to get samples for
            
        Returns:
            List of samples if available, None if no samples exist
        """
        if self._current_samples is None:
            return None
            
        try:
            node_idx = self._current_samples.schema.var_to_idx[node_name]
            return self._current_samples.values[:, node_idx].tolist()
        except (KeyError, IndexError):
            return None

    def get_node_distribution(self,
                              node_name: str,
                              codomain: str = "normalized_counts") -> Optional[tuple[list[int], list[float|int]]]:
        """Get the distribution for a specific node from cached samples.
        
        The form of the result are two lists, one for the x values and one for
        the y values.

        The codomain can be one of "counts" or "normalized_counts".
        """
        if self._current_samples is None:
            return None
        schema = self._current_samples.schema
        node_idx = schema.var_to_idx[node_name]
        samples = self._current_samples.values[:, node_idx]
        num_node_states = schema.var_to_states[node_name]
        counts = np.bincount(samples, minlength=num_node_states)
        if codomain == "counts":
            y = counts.tolist()
        elif codomain == "normalized_counts":
            y = (counts / len(samples)).tolist()
        else:
            raise ValueError(f"Invalid codomain: {codomain}")
        x = list(range(num_node_states))
        return x, y
    
    def store_samples(self, samples: sampling.SampleArray, metadata: Dict[str, Any]):
        """Store new samples and their metadata."""
        self._current_samples = samples
        self._samples_metadata = metadata
    
    @property
    def n_samples(self) -> int:
        """Get the number of samples in the current samples."""
        if self._current_samples is None:
            return 0
        return self._current_samples.n

    @property
    def current_graph(self) -> Optional[cgm.CG]: return self._current_graph
    
    @property
    def current_graph_state(self) -> Optional[cgm.GraphState]: return self._current_graph_state
    
    @property
    def current_seed(self) -> Optional[int]:
        """Get the currently used random seed."""
        return self._current_seed
    
    @property
    def conditioned_nodes(self) -> Dict[str, int]:
        if not self._current_graph_state: return {}
        return {
            name: int(self._current_graph_state.values[idx])
            for name, idx in self._current_graph_state.schema.var_to_idx.items()
            if self._current_graph_state.mask[idx]
        }
    
    def condition(self, node: str, value: Optional[int] = None) -> bool:
        """Set or clear node condition and update visualization."""
        if not self._current_graph: return False
        
        # Create new graph state if needed
        if not self._current_graph_state:
            print(f"Creating new graph state for condition: {node}={value}")
            self._current_graph_state = cgm.GraphState.create(self._current_graph)
            
        if node not in self._current_graph_state.schema.var_to_idx: return False
        
        # Get current conditions
        conditions = self.conditioned_nodes.copy()
        print(f"Current conditions before update: {conditions}")
        
        # Update or remove the condition for the specified node
        if value is None:
            conditions.pop(node, None)
            print(f"Removed condition for {node}")
        else:
            conditions[node] = value
            print(f"Added condition: {node}={value}")
            
        # Create a new state with the updated conditions
        print(f"Creating new state with conditions: {conditions}")
        self._current_graph_state = self._current_graph.condition(**conditions)
        return True

# Initialize global state
# _vizstate = VizState()

# Expose internal state instance
# vizstate_instance = _vizstate 
