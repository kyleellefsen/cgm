"""Visualization module for CGM.

This module provides a simple web-based visualization tool for Bayesian networkspyth

Example usage:

    import cgm
    import cgm.viz
    g1 = cgm.example_graphs.get_cg1()
    cgm.viz.show(g1, open_browser=True)  # Open browser for the first graph
    
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

class Link(BaseModel):
    source: str
    target: str

class GraphState(BaseModel):
    nodes: List[Node]
    links: List[Link]

# Global state
_app = FastAPI()
_current_graph = None
_server_thread = None
_port = 5050

_cgm_dir = pathlib.Path(cgm.__file__).parent
_static_dir = _cgm_dir / "viz" / "static"
assert _static_dir.exists(), f"Static files directory not found: {_static_dir}"
print(f"Contents of _static_dir: {list(_static_dir.glob('*'))}") 


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
        return {"nodes": [], "links": []}

    # Convert graph to D3 format
    nodes = []
    links = []

    # Add nodes with CPD information
    for node in _current_graph.nodes:
        node_data = {
            "id": node.name,
            "states": node.num_states
        }

        # Add CPD table if available
        cpd = node.cpd
        if cpd is not None:
            table_html = cpd.table.html()
            node_data["cpd"] = table_html
        nodes.append(node_data)
    # Add edges from parent relationships
    for node in _current_graph.nodes:
        for parent in node.parents:
            links.append({
                "source": parent.name,
                "target": node.name
            })
    return {"nodes": nodes, "links": links}

def start_server() -> None:
    """Start the visualization server in a background thread."""
    global _server_thread  # pylint: disable=global-statement
    if _server_thread is not None:
        return

    def run_server():
        uvicorn.run(_app, host="127.0.0.1", port=_port, log_level="error")

    _server_thread = threading.Thread(target=run_server, daemon=True)
    _server_thread.start()

def show(graph, open_browser: bool = True) -> None:
    """Update the visualization with a new graph."""
    global _current_graph  # pylint: disable=global-statement
    global _server_thread  # pylint: disable=global-statement

    # Start server if not running
    if _server_thread is not None and not _server_thread.is_alive():
        _server_thread = None

    if _server_thread is None:
        start_server()

    # Update current graph
    _current_graph = graph

    # Open browser if requested
    if open_browser:
        webbrowser.open(f'http://localhost:{_port}')

def stop_server() -> None:
    """Stop the visualization server."""
    global _server_thread  # pylint: disable=global-statement
    if _server_thread is not None:
        # Cleanup and shutdown
        _server_thread = None
