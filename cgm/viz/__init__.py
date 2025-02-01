"""Visualization module for CGM.

This module provides a simple web-based visualization tool for Bayesian networkspyth

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

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

import cgm

# Data models for type safety
class Node(BaseModel):
    id: str
    states: int
    cpd: Optional[str] = None
    conditioned_state: Optional[int] = None

class Link(BaseModel):
    source: str
    target: str

class GraphState(BaseModel):
    nodes: List[Node]
    links: List[Link]

# Global state
_app = FastAPI()
_current_graph = None
_graph_state = None
_server_thread = None
_port = 5050

# Get the development directory path
_dev_dir = pathlib.Path(__file__).parent
_static_dir = _dev_dir / "static"
print(f"Using static directory: {_static_dir} (exists: {_static_dir.exists()})")
assert _static_dir.exists(), f"Static files directory not found: {_static_dir}"

# Mount static files directory
_app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

@_app.get("/", response_class=HTMLResponse)
async def home() -> FileResponse:
    """Serve the visualization page."""
    html_path = _static_dir / "viz-layout.html"
    print(html_path)
    return FileResponse(html_path)

@_app.get("/state", response_model=GraphState)
async def get_state() -> Dict:
    """Return the current graph state as JSON."""
    if _current_graph is None:
        print("No current graph")
        return {"nodes": [], "links": []}

    print("\n=== SERVER STATE REQUEST START ===")
    # Convert graph to D3 format
    nodes = []
    links = []

    # Add nodes with CPD information and conditioning state
    for node in _current_graph.nodes:
        node_data = {
            "id": node.name,
            "states": node.num_states,
            "type": "effect",
            "conditioned_state": -1  # Default value for all nodes
        }

        # Add CPD table if available
        cpd = node.cpd
        if cpd is not None:
            table_html = cpd.table.html()
            node_data["cpd"] = table_html

        # Override conditioned_state if actually conditioned
        if _graph_state is not None:
            idx = _graph_state.schema.var_to_idx[node.name]
            if _graph_state.mask[idx]:
                value = _graph_state.values[idx]
                if value != -1:
                    node_data["conditioned_state"] = int(value)
                    print(f"Node {node.name} conditioned to {value}")

        nodes.append(node_data)
        print(f"Added node: {node_data['id']}, conditioned_state: {node_data.get('conditioned_state')}")

    # Add edges from parent relationships
    for node in _current_graph.nodes:
        for parent in node.parents:
            link = {
                "source": parent.name,
                "target": node.name
            }
            links.append(link)
            print(f"Added link: {parent.name} -> {node.name}")

    response = {"nodes": nodes, "links": links}
    print("=== SERVER STATE REQUEST END ===\n")
    return response

@_app.post("/condition/{node_id}/{state}")
async def condition_node(node_id: str, state: int):
    """Update node's conditioned state"""
    if _current_graph is None:
        print("No current graph")
        return {"status": "no graph loaded"}
        
    # Create new graph state if needed
    global _graph_state
    if _graph_state is None:
        print("Creating new graph state")
        _graph_state = cgm.GraphState.create(_current_graph)
        
    schema = _graph_state.schema
    if node_id not in schema.var_to_idx:
        print(f"Node {node_id} not found")
        return {"status": "node not found"}
        
    # Update state
    new_values = _graph_state.values.copy()
    new_mask = _graph_state.mask.copy()
    idx = schema.var_to_idx[node_id]
    
    if state == -1:  # Clear conditioning
        print(f"Clearing conditioning for node {node_id}")
        new_values[idx] = -1
        new_mask[idx] = False
    else:  # Set conditioning
        print(f"Setting node {node_id} to state {state}")
        new_values[idx] = state
        new_mask[idx] = True
        
    # Store updated state
    _graph_state = cgm.GraphState(
        network=_current_graph,
        schema=schema,
        _values=new_values,
        _mask=new_mask
    )
    print(f"Updated graph state: values={_graph_state.values}, mask={_graph_state.mask}")
    return {"status": "updated"}

@_app.get("/test-static")
async def test_static():
    """Test route to verify static file access."""
    js_path = _static_dir / "js" / "viz-graph.js"
    print(f"Testing static file access. File exists: {js_path.exists()}")
    print(f"Static directory contents: {list(_static_dir.iterdir())}")
    if "js" in [d.name for d in _static_dir.iterdir()]:
        print(f"JS directory contents: {list((_static_dir / 'js').iterdir())}")
    return {"static_dir": str(_static_dir), "js_exists": js_path.exists()}

def start_server() -> None:
    """Start the visualization server in a background thread."""
    global _server_thread  # pylint: disable=global-statement
    if _server_thread is not None:
        return

    def run_server():
        print("Starting server...")
        print(f"Static directory: {_static_dir}")
        uvicorn.run(_app, host="127.0.0.1", port=_port, log_level="info")

    _server_thread = threading.Thread(target=run_server, daemon=True)
    _server_thread.start()

def show(graph, graph_state: Optional[cgm.GraphState] = None, open_new_browser_window: bool = True) -> None:
    """Update the visualization with a new graph and optional state."""
    global _current_graph  # pylint: disable=global-statement
    global _graph_state   # pylint: disable=global-statement
    global _server_thread  # pylint: disable=global-statement

    # Start server if not running
    if _server_thread is not None and not _server_thread.is_alive():
        _server_thread = None

    if _server_thread is None:
        start_server()

    # Update current graph and state
    _current_graph = graph
    _graph_state = graph_state

    # Open browser if requested
    if open_new_browser_window:
        webbrowser.open(f'http://localhost:{_port}')

def stop_server() -> None:
    """Stop the visualization server."""
    global _server_thread  # pylint: disable=global-statement
    if _server_thread is not None:
        # Cleanup and shutdown
        _server_thread = None
