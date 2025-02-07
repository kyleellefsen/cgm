# CGM Visualization Library

"""
The CGM visualization library provides real-time visualization of causal graphical models
during development and debugging. It is designed to be minimal, fast, and developer-friendly,
prioritizing quick feedback loops over feature completeness.

## Design Philosophy

The visualization system follows these core principles:

1. **Minimal Setup** - Developers should be able to visualize graphs with minimal 
   ceremony or configuration. The system uses a simple FastAPI server and D3.js for
   visualization.

2. **State Management** - The FastAPI server acts as the single source of truth for
   the current graph state. This allows developers to:
   - Close and reopen browser windows without losing state
   - Get the latest visualization just by refreshing the browser
   - Have the visualization always reflect the current state in Python memory

3. **Simple Updates** - Updating the visualization should be a single line of code.
   The system doesn't try to automatically track changes to graphs; instead, developers
   explicitly request updates when needed.

4. **Layout Independence** - The browser (using D3.js) handles all node positioning
   and layout. The server only provides graph structure, letting D3's force-directed
   layout handle the rest.

## Implementation Details

The system consists of three main components:

1. **FastAPI Server**
   - Single server instance running on a fixed port
   - Two routes:
     - `/` - Serves the HTML/D3.js visualization page
     - `/state` - JSON endpoint providing current graph state
   - Maintains exactly one graph state in memory
   - No persistence between Python sessions

2. **Browser Visualization**
   - Pure D3.js force-directed graph visualization
   - Polls server periodically for state updates
   - Handles all layout/positioning independently
   - Can be refreshed/reopened without losing state

3. **Python API**
   - Simple functions for starting server and updating visualizations
   - No automatic tracking or complex state management
   - Works from any Python environment (REPL, iPython, scripts)

## Usage Example

```python
import cgm
from cgm.example_graphs import get_cg1

# Start visualization server (one time setup)
cgm.viz.start_server()

# Create and visualize a graph
g1 = get_cg1()
cgm.viz.show(g1)  # Opens browser first time

# Modify graph and update visualization
g2 = modify_graph(g1)
cgm.viz.show(g2)  # Updates existing visualization
```

## Design Decisions and Tradeoffs

1. **Single Graph State**
   - The server only maintains state for one graph at a time
   - When you show() a new graph, it replaces the previous visualization
   - Simplifies state management and matches typical development workflow

2. **Manual Updates**
   - No automatic tracking of graph changes
   - Developers must explicitly call show() to update visualization
   - Tradeoff: Less magical, more predictable

3. **Persistent Server**
   - Server runs until explicitly stopped or Python session ends
   - Must be manually started (no auto-start on first show())
   - Tradeoff: More explicit control, slight setup overhead

4. **Browser Independence**
   - Browser is just a view into server state
   - Can be closed/reopened freely
   - Multiple browser windows show same visualization
   - Tradeoff: No browser-specific state persistence

## Limitations

1. No support for multiple simultaneous graph visualizations
2. No persistence between Python sessions
3. No automatic updates when graphs change
4. Layout not preserved between updates
5. Simple visual styling only

## Dependencies

- FastAPI for server
- D3.js (loaded from CDN) for visualization
- Standard Python libraries only
"""
