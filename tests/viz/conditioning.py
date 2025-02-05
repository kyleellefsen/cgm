"""Tests for graph conditioning functionality."""
# pylint: disable=redefined-outer-name
import numpy as np
import pytest
from fastapi.testclient import TestClient
import cgm
import cgm.viz

@pytest.fixture
def client():
    """Create a test client."""
    from cgm.viz.app import create_fastapi_app  # Import inside fixture to avoid side effects
    app = create_fastapi_app()
    return TestClient(app)

@pytest.fixture
def example_graph():
    """Create an example graph for testing."""
    return cgm.example_graphs.get_cg2()  # The sprinkler network

@pytest.fixture
def viz_state(client: TestClient) -> cgm.viz.VizState:
    """Get the visualization state from the test client."""
    return client.app.state.viz_state  # type: ignore

@pytest.fixture(autouse=True)
def cleanup(viz_state: cgm.viz.VizState):
    """Clean up the state after each test."""
    yield
    empty_graph = cgm.CG()
    viz_state.set_current_graph(empty_graph)
    viz_state.clear_samples()

def test_state_passing(viz_state: cgm.viz.VizState, example_graph: cgm.CG):
    """Test that the graph state (conditioned nodes) get passed back and forth correctly."""
    viz_state.set_current_graph(example_graph)
    assert viz_state.conditioned_nodes == {}
    viz_state.condition('rain', 0)
    assert viz_state.conditioned_nodes == {'rain': 0}

def test_sample_passing(viz_state: cgm.viz.VizState, example_graph: cgm.CG, client: TestClient):
    """Test sampling with conditioned nodes."""
    viz_state.set_current_graph(example_graph)
    viz_state.condition('season', 0)
    
    # Create and send sampling request
    sampling_request = cgm.viz.models.SamplingRequest(
        num_samples=100,
        options=cgm.viz.models.SamplingOptions(random_seed=42)
    )
    response = client.post(
        "/api/sample",
        json=sampling_request.model_dump()  # Convert Pydantic model to dict
    )
    assert response.status_code == 200
    sample_data = response.json()
    assert sample_data["success"] is True
    assert sample_data["result"]["total_samples"] == 100

    # Get node distribution
    node_distribution_request = cgm.viz.models.NodeDistributionRequest(
        node_name='season',
        codomain='counts'
    )
    response = client.post(
        "/api/node_distribution",
        json=node_distribution_request.model_dump()
    )
    assert response.status_code == 200
    dist_data = response.json()
    assert dist_data["success"] is True
    assert "x_values" in dist_data["result"]
    assert "y_values" in dist_data["result"]
    assert len(dist_data["result"]["x_values"]) == 4  # season has 4 states

def test_conditioning_state(viz_state: cgm.viz.VizState, example_graph: cgm.CG):
    """Test conditioning state management."""
    viz_state.set_current_graph(example_graph)
    
    # Create graph state with rain=True
    state = cgm.GraphState.create(example_graph)
    sample = np.full(state.schema.num_vars, -1)
    sample[state.schema.var_to_idx['rain']] = 1
    state = state.condition_on_sample(sample)
    
    # Update state with conditioned values
    viz_state.set_current_graph(example_graph, state)
    
    # Verify state
    assert state.values[state.schema.var_to_idx['rain']] == 1
    assert bool(state.mask[state.schema.var_to_idx['rain']]) == True  # Convert numpy bool to Python bool
    assert viz_state.conditioned_nodes == {'rain': 1}