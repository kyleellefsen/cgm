# This isn't a pytest, it has to be run manually
# The command to run it is:
# python -m tests.viz.conditioning
import numpy as np


import cgm
import cgm.viz
from fastapi.testclient import TestClient
vizstate_instance = cgm.viz.vizstate_instance


def test_conditioning_visualization():
    # Create a simple test graph
    g = cgm.example_graphs.get_cg2()  # Using rain/sprinkler/grass example
    
    # Start visualization
    cgm.viz.show(g, open_new_browser_window=True)
    
    print(vizstate_instance.conditioned_nodes)
    vizstate_instance.condition('season', 0)
    print(vizstate_instance.conditioned_nodes)
    
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

def test_state_passing():
    """This test is to check that the graphstate (conditioned nodes) get 
    passed back and forth correctly.
    """

    g = cgm.example_graphs.get_cg2()  # Using rain/sprinkler/grass example
    cgm.viz.show(g, open_new_browser_window=True)
    assert vizstate_instance.conditioned_nodes == {}
    vizstate_instance.condition('rain', 0)
    assert vizstate_instance.conditioned_nodes == {'rain': 0}

def test_sample_passing():
    g = cgm.example_graphs.get_cg2()  # Using rain/sprinkler/grass example
    cgm.viz.show(g, open_new_browser_window=True)
    vizstate_instance.condition('season', 0)
    client = TestClient(cgm.viz.app._app)
    sampling_request = cgm.viz.models.SamplingRequest(
        num_samples=100,
        options=cgm.viz.models.SamplingOptions(random_seed=42)
    )
    print(sampling_request.model_dump())
    response = client.post(
        "/api/sample",
        json=sampling_request.model_dump()  # Convert Pydantic model to dict
    )
    print(response.json())


    # Then parse the response
    result = cgm.viz.models.SamplingResponse(**response.json())
    node_distribution_request = cgm.viz.models.NodeDistributionRequest(
        node_name='season',
        codomain='counts'
    )
    node_distribution_response = client.post(
        "/api/node_distribution",
        json=node_distribution_request.model_dump()
    )
    print(node_distribution_response.json())

if __name__ == "__main__":
    test_conditioning_visualization()