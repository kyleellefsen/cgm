"""Manual visual inspection tests for the visualization interface."""
import numpy as np
import cgm
import cgm.viz

def test_conditioning_visualization():
    """Interactive test to verify conditioning visualization.
    
    This test guides you through visual inspection of the conditioning interface.
    It will:
    1. Show the initial graph state
    2. Apply conditioning to nodes
    3. Guide you through verifying the visual elements
    """
    # Create a simple test graph
    g = cgm.example_graphs.get_cg2()  # Using rain/sprinkler/grass example
    
    # Start visualization
    cgm.viz.show(g, open_new_browser_window=True)
    
    print("\nVisual Inspection Test Steps:")
    print("1. Initial State")
    print("   - Verify no nodes show conditioning circles")
    print("   - Click 'rain' node and verify CPD table is shown")
    input("Press Enter to continue...")
    
    # Create graph state with rain=True
    state = cgm.GraphState.create(g)
    # Create sample array with all -1s (unset)
    sample = np.full(state.schema.num_vars, -1)
    # Set rain=1 at its index
    sample[state.schema.var_to_idx['rain']] = 1
    state = state.condition_on_sample(sample)
    
    # Update visualization with conditioned state
    cgm.viz.show(g, graph_state=state, open_new_browser_window=False)
    
    print("\n2. Conditioned State")
    print("   - Verify 'rain' node shows blue conditioning circle")
    print("   - Check CPD tables for other nodes - relevant rows should be highlighted")
    print("   - Try clicking 'sprinkler' node to toggle its conditioning")
    input("Press Enter to finish...")
    
    # Verify the final state
    assert state.values[state.schema.var_to_idx['rain']] == 1
    assert bool(state.mask[state.schema.var_to_idx['rain']]) == True

if __name__ == "__main__":
    test_conditioning_visualization() 