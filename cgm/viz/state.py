"""State management for the visualization module."""
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
import time
import numpy as np

import cgm
from ..inference.approximate import sampling

@dataclass
class State:
    """Global visualization state with convenient property accessors."""
    _current_graph: Optional[cgm.CG] = None
    _graph_state: Optional[cgm.GraphState] = None
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
    
    def store_samples(self, samples: sampling.SampleArray, metadata: Dict[str, Any]):
        """Store new samples and their metadata."""
        self._current_samples = samples
        self._samples_metadata = metadata
    
    @property
    def graph(self): return self._current_graph
    
    @property
    def state(self): return self._graph_state
    
    @property
    def current_seed(self) -> Optional[int]:
        """Get the currently used random seed."""
        return self._current_seed
    
    @property
    def conditioned_nodes(self) -> Dict[str, int]:
        if not self._graph_state: return {}
        return {
            name: int(self._graph_state.values[idx])
            for name, idx in self._graph_state.schema.var_to_idx.items()
            if self._graph_state.mask[idx]
        }
    
    def condition(self, node: str, value: Optional[int] = None) -> bool:
        """Set or clear node condition and update visualization."""
        if not self._current_graph: return False
        
        # Create new graph state if needed
        if not self._graph_state:
            print(f"Creating new graph state for condition: {node}={value}")
            self._graph_state = cgm.GraphState.create(self._current_graph)
            
        if node not in self._graph_state.schema.var_to_idx: return False
        
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
        self._graph_state = self._current_graph.condition(**conditions)
        return True

# Initialize global state
_state = State()

# Public interface
def graph(): return _state.graph
def state(): return _state.state
def conditioned_nodes(): return _state.conditioned_nodes
def condition(node: str, value: Optional[int] = None) -> bool:
    """Set or clear node condition and update visualization."""
    return _state.condition(node, value)

# Expose internal state instance
state_instance = _state 