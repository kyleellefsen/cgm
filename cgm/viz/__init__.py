"""Visualization module for CGM.

This module provides a simple web-based visualization tool for Bayesian networks.

Example usage:
    import cgm
    import cgm.viz
    g1 = cgm.example_graphs.get_cg1()
    cgm.viz.show(g1, open_new_browser_window=True)  # Open browser for the first graph
"""
from typing import Optional, Dict, List, Any
import pathlib
import threading
import webbrowser
from dataclasses import dataclass, field
import numpy as np
import time

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

import cgm
from ..inference.approximate import sampling
from ..inference.approximate.sampling import ForwardSamplingCertificate

# Pydantic models for request/response validation
class SamplingOptions(BaseModel):
    burn_in: Optional[int] = 100
    thinning: Optional[int] = 1
    random_seed: Optional[int] = None
    cache_results: Optional[bool] = True

class SamplingRequest(BaseModel):
    method: str = "forward"
    num_samples: int = 1000
    conditions: Dict[str, int] = {}
    target_variable: Optional[str] = None  # Variable to get samples for
    options: SamplingOptions = SamplingOptions()

class SamplingResponse(BaseModel):
    total_samples: int
    accepted_samples: int
    rejected_samples: int
    samples: List[int]
    seed_used: int
    target_variable: str  # Name of the variable these samples are for

@dataclass
class State:
    """Global visualization state with convenient property accessors."""
    _app: FastAPI = field(default_factory=FastAPI)
    _current_graph: Optional[cgm.CG] = None
    _graph_state: Optional[cgm.GraphState] = None
    _server_thread: Optional[threading.Thread] = None
    _port: int = 5050
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
            self._graph_state = cgm.GraphState.create(self._current_graph)
            
        if node not in self._graph_state.schema.var_to_idx: return False
        
        idx = self._graph_state.schema.var_to_idx[node]
        new_values, new_mask = self._graph_state.values.copy(), self._graph_state.mask.copy()
        
        if value is None:
            new_values[idx], new_mask[idx] = -1, False
        else:
            new_values[idx], new_mask[idx] = value, True
            
        self._graph_state = cgm.GraphState(
            network=self._current_graph,
            schema=self._graph_state.schema,
            _values=new_values,
            _mask=new_mask
        )
        show(self._current_graph, self._graph_state, False)
        return True

# Initialize global state
_state = State()

# Get the development directory path and mount static files
_dev_dir = pathlib.Path(__file__).parent
_static_dir = _dev_dir / "static"
assert _static_dir.exists(), f"Static files directory not found: {_static_dir}"
_state._app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

@_state._app.get("/", response_class=HTMLResponse)
async def home() -> FileResponse:
    """Serve the visualization page."""
    return FileResponse(_static_dir / "viz-layout.html")

@_state._app.get("/state")
async def get_state() -> Dict:
    """Return the current graph state as JSON."""
    if not _state.graph:
        return {"nodes": [], "links": []}

    nodes = [{
        "id": node.name,
        "states": node.num_states,
        "type": "effect",
        "conditioned_state": -1,
        "cpd": node.cpd.table.html() if node.cpd else None,
        **({"conditioned_state": int(_state.state.values[_state.state.schema.var_to_idx[node.name]])}
           if _state.state and _state.state.mask[_state.state.schema.var_to_idx[node.name]] else {})
    } for node in _state.graph.nodes]

    links = [{"source": parent.name, "target": node.name}
            for node in _state.graph.nodes
            for parent in node.parents]

    return {"nodes": nodes, "links": links}

@_state._app.post("/condition/{node_id}/{state}")
async def condition_node(node_id: str, state: int):
    """Update node's conditioned state"""
    if not _state.graph:
        return {"status": "no graph loaded"}
    
    # Ensure graph state exists
    if not _state.state:
        _state._graph_state = cgm.GraphState.create(_state.graph)
    
    if _state.condition(node_id, None if state == -1 else state):
        return {"status": "updated"}
    return {"status": "failed"}

@_state._app.post("/api/sample", response_model=SamplingResponse)
async def generate_samples(request: SamplingRequest) -> SamplingResponse:
    """Generate samples from the current graph state."""
    if not _state.graph:
        raise HTTPException(status_code=400, detail="No graph loaded")
        
    try:
        # Set random seed and get the seed used
        seed_used = _state.set_seed(request.options.random_seed)
        
        # Create the conditioned state
        state = _state.graph.condition(**request.conditions)
        
        # Create the sampling certificate
        try:
            cert = ForwardSamplingCertificate(state)
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid sampling conditions: {str(e)}"
            )
            
        # Generate samples
        try:
            sample_array, _ = sampling.get_n_samples(
                _state.graph,
                _state._rng,
                num_samples=request.num_samples,
                state=state,
                certificate=cert,
                return_array=True
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Sampling failed: {str(e)}"
            )
        
        if not isinstance(sample_array, sampling.SampleArray):
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected sample format: {type(sample_array)}"
            )
        
        # Store samples and metadata
        metadata = {
            'seed': seed_used,
            'timestamp': time.time(),
            'num_samples': request.num_samples,
            'conditions': request.conditions
        }
        _state.store_samples(sample_array, metadata)
        
        # Process results for the target variable (if specified)
        try:
            target_var = request.target_variable
            if target_var is None:
                target_var = _state.graph.nodes[0].name
            
            samples = _state.get_node_samples(target_var)
            if samples is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown variable: {target_var}"
                )
            
            result = SamplingResponse(
                total_samples=len(samples),
                accepted_samples=len(samples),
                rejected_samples=0,
                samples=samples,
                seed_used=seed_used,
                target_variable=target_var
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process sampling results: {str(e)}"
            )
            
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Unexpected error during sampling: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

@_state._app.get("/api/node_distribution/{node_name}")
async def get_node_distribution(node_name: str) -> Dict[str, Any]:
    """Get the distribution for a specific node from cached samples."""
    if not _state.graph:
        raise HTTPException(status_code=400, detail="No graph loaded")
        
    if _state._current_samples is None:
        raise HTTPException(
            status_code=400, 
            detail="No samples available. Generate samples first."
        )
    
    samples = _state.get_node_samples(node_name)
    if samples is None:
        raise HTTPException(status_code=400, detail=f"Unknown node: {node_name}")
    
    return {
        "samples": samples,
        "metadata": _state._samples_metadata,
        "node": node_name
    }

def start_server() -> None:
    """Start the visualization server in a background thread."""
    if _state._server_thread is not None:
        return

    def run_server():
        uvicorn.run(_state._app, host="127.0.0.1", port=_state._port, log_level="warning")

    _state._server_thread = threading.Thread(target=run_server, daemon=True)
    _state._server_thread.start()

def show(graph, graph_state: Optional[cgm.GraphState] = None, open_new_browser_window: bool = True) -> None:
    """Update the visualization with a new graph and optional state."""
    if _state._server_thread is not None and not _state._server_thread.is_alive():
        _state._server_thread = None

    if _state._server_thread is None:
        start_server()

    _state._current_graph = graph
    _state._graph_state = graph_state

    if open_new_browser_window:
        webbrowser.open(f'http://localhost:{_state._port}')

def stop_server() -> None:
    """Stop the visualization server."""
    _state._server_thread = None

# Public interface
def graph(): return _state.graph
def state(): return _state.state
def conditioned_nodes(): return _state.conditioned_nodes
def condition(node: str, value: Optional[int] = None) -> bool:
    """Set or clear node condition and update visualization."""
    result = _state.condition(node, value)
    return result

