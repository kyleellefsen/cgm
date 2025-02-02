"""Application setup and server management for the visualization module."""
import pathlib
import threading
import webbrowser
from typing import Optional

import uvicorn
from fastapi import FastAPI

import cgm
from .state import vizstate_instance
from .routes import setup_routes

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI()
    
    # Setup static files
    static_dir = pathlib.Path(__file__).parent / "static"
    assert static_dir.exists(), f"Static files directory not found: {static_dir}"
    
    # Setup routes
    setup_routes(app, static_dir)
    
    return app

# Server management
_app = create_app()
_server_thread: Optional[threading.Thread] = None
_port = 5050

def start_server() -> None:
    """Start the visualization server in a background thread."""
    global _server_thread
    
    if _server_thread is not None:
        return

    def run_server():
        uvicorn.run(_app, host="127.0.0.1", port=_port, log_level="warning")

    _server_thread = threading.Thread(target=run_server, daemon=True)
    _server_thread.start()

def show(graph: cgm.CG, graph_state: Optional[cgm.GraphState] = None, 
         open_new_browser_window: bool = True) -> None:
    """Update the visualization with a new graph and optional state."""
    global _server_thread
    
    if _server_thread is not None and not _server_thread.is_alive():
        _server_thread = None

    if _server_thread is None:
        start_server()

    # Update state using state instance
    vizstate_instance._current_graph = graph
    vizstate_instance._graph_state = graph_state

    if open_new_browser_window:
        webbrowser.open_new_tab(f'http://localhost:{_port}')

def stop_server() -> None:
    """Stop the visualization server."""
    global _server_thread
    _server_thread = None 