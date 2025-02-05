"""Tests for the visualization API endpoints."""
# pylint: disable=redefined-outer-name
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

def test_empty_state(viz_state: cgm.viz.VizState, client: TestClient):
    """Test state endpoint with no graph loaded."""
    response = client.get("/state")
    assert response.status_code == 200
    data = response.json()
    assert data == {"nodes": [], "links": []}

def test_node_distribution_no_graph(viz_state: cgm.viz.VizState, client: TestClient):
    """Test node distribution endpoint with no graph loaded."""
    response = client.post("/api/node_distribution", json={"node_name": "rain"})
    assert response.status_code == 200  # API returns 200 with error in body
    data = response.json()
    assert data["success"] is False
    assert data["error"]["message"] == "No graph loaded"

def test_node_distribution_no_samples(viz_state: cgm.viz.VizState, example_graph: cgm.CG, client: TestClient):
    """Test node distribution endpoint with graph but no samples."""
    # Load the graph
    viz_state.set_current_graph(example_graph)
    
    response = client.post("/api/node_distribution", json={"node_name": "rain"})
    assert response.status_code == 200  # API returns 200 with error in body
    data = response.json()
    assert data["success"] is False
    assert data["error"]["message"] == "No samples available. Generate samples first."

def test_sampling_flow(viz_state: cgm.viz.VizState, example_graph: cgm.CG, client: TestClient):
    """Test the complete sampling flow."""
    # Load the graph
    viz_state.set_current_graph(example_graph)
    
    # Generate samples
    sample_request = {
        "method": "forward",
        "num_samples": 1000,
        "options": {
            "random_seed": 42,
            "cache_results": True
        }
    }
    
    response = client.post("/api/sample", json=sample_request)
    assert response.status_code == 200
    sample_data = response.json()
    if not sample_data["success"]:  # Add error logging
        print(f"Sampling failed: {sample_data['error']}")
    assert sample_data["success"] is True
    assert sample_data["result"]["total_samples"] == 1000
    assert sample_data["result"]["seed_used"] == 42
    
    # Now get distribution for a node
    response = client.post("/api/node_distribution", json={"node_name": "rain"})
    assert response.status_code == 200
    dist_data = response.json()
    assert dist_data["success"] is True
    assert "x_values" in dist_data["result"]
    assert "y_values" in dist_data["result"]
    assert len(dist_data["result"]["x_values"]) == 2  # rain has 2 states

def test_invalid_node(viz_state: cgm.viz.VizState,  example_graph: cgm.CG, client: TestClient):
    """Test requesting distribution for non-existent node."""
    # Load the graph
    viz_state.set_current_graph(example_graph)
    
    # Generate some samples first
    sample_request = {
        "method": "forward",
        "num_samples": 100,
        "options": {}  # Use default options
    }
    client.post("/api/sample", json=sample_request)
    
    # Try to get distribution for invalid node
    response = client.post("/api/node_distribution", json={"node_name": "nonexistent"})
    assert response.status_code == 200  # API returns 200 with error in body
    data = response.json()
    assert data["success"] is False
    assert "Unknown node" in data["error"]["message"]

def test_sampling_with_conditions(viz_state: cgm.viz.VizState, example_graph: cgm.CG, client: TestClient):
    """Test sampling with conditions set."""
    # Load the graph
    viz_state.set_current_graph(example_graph)
    
    # Set a condition
    response = client.post("/condition/rain/1")
    assert response.status_code == 200
    
    # Generate samples with condition - this should fail because 'season' is not conditioned
    sample_request = {
        "method": "forward",
        "num_samples": 1000,
        "options": {
            "random_seed": 42
        }
    }
    
    response = client.post("/api/sample", json=sample_request)
    assert response.status_code == 200  # API returns 200 with error in body
    error_data = response.json()
    assert error_data["success"] is False
    assert "Cannot use forward sampling with these conditions" in error_data["error"]["message"]
    assert "Node rain is conditioned but its ancestor season is not conditioned" in error_data["error"]["details"]

@pytest.fixture(autouse=True)
def cleanup(viz_state: cgm.viz.VizState):
    """Clean up the state after each test."""
    yield
    # Create an empty graph instead of using None
    empty_graph = cgm.CG()
    viz_state.set_current_graph(empty_graph)
    viz_state.clear_samples()
