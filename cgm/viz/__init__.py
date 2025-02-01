"""Visualization module for CGM.

This module provides a simple web-based visualization tool for Bayesian networks.

Example usage:
    import cgm
    import cgm.viz
    g1 = cgm.example_graphs.get_cg1()
    cgm.viz.show(g1, open_new_browser_window=True)  # Open browser for the first graph
"""
from typing import Optional, Dict, List
import pathlib
import threading
import webbrowser
from dataclasses import dataclass

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

import cgm

@dataclass
class State:
    """Global visualization state with convenient property accessors."""
    _app: FastAPI = FastAPI()
    _current_graph: Optional[cgm.CG] = None
    _graph_state: Optional[cgm.GraphState] = None
    _server_thread: Optional[threading.Thread] = None
    _port: int = 5050
    
    @property
    def graph(self): return self._current_graph
    
    @property
    def state(self): return self._graph_state
    
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

