# This isn't a pytest, it has to be run manually
# The command to run it is:
# python -m tests.viz.conditioning
import cgm
import cgm.viz
import numpy as np

def test_conditioning_visualization():
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
    
    assert state.values[state.schema.var_to_idx['rain']] == 1
    assert state.mask[state.schema.var_to_idx['rain']] == True

if __name__ == "__main__":
    test_conditioning_visualization()