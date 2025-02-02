"""Visualization module for CGM.

This module provides a simple web-based visualization tool for Bayesian networks.

Example usage:
    import cgm
    import cgm.viz
    g1 = cgm.example_graphs.get_cg1()
    cgm.viz.show(g1, open_new_browser_window=True)  # Open browser for the first graph
"""

from .app import show, start_server, stop_server
from .state import vizstate_instance

__all__ = [
    'show', 'start_server', 'stop_server', 'vizstate_instance'
]