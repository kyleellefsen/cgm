"""Tests for the visualization API endpoints."""
import pytest
from fastapi.testclient import TestClient
import cgm
import cgm.viz

@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(cgm.viz._state._app)

@pytest.fixture
def example_graph():
    """Create an example graph for testing."""
    return cgm.example_graphs.get_cg2()  # The sprinkler network

def test_empty_state(client):
    """Test state endpoint with no graph loaded."""
    response = client.get("/state")
    assert response.status_code == 200
    data = response.json()
    assert data == {"nodes": [], "links": []}

def test_node_distribution_no_graph(client):
    """Test node distribution endpoint with no graph loaded."""
    response = client.get("/api/node_distribution/rain")
    assert response.status_code == 400
    assert response.json()["detail"] == "No graph loaded"

def test_node_distribution_no_samples(client, example_graph):
    """Test node distribution endpoint with graph but no samples."""
    # Load the graph
    cgm.viz._state._current_graph = example_graph
    
    response = client.get("/api/node_distribution/rain")
    assert response.status_code == 400
    assert response.json()["detail"] == "No samples available. Generate samples first."

def test_sampling_flow(client, example_graph):
    """Test the complete sampling flow."""
    # Load the graph
    cgm.viz._state._current_graph = example_graph
    
    # Generate samples
    sample_request = {
        "method": "forward",
        "num_samples": 1000,
        "conditions": {},
        "options": {
            "random_seed": 42,  # Use fixed seed for reproducibility
            "cache_results": True
        }
    }
    
    response = client.post("/api/sample", json=sample_request)
    assert response.status_code == 200
    sample_data = response.json()
    assert sample_data["total_samples"] == 1000
    assert sample_data["seed_used"] == 42
    
    # Now get distribution for a node
    response = client.get("/api/node_distribution/rain")
    assert response.status_code == 200
    dist_data = response.json()
    assert "samples" in dist_data
    assert len(dist_data["samples"]) == 1000
    assert "metadata" in dist_data
    assert dist_data["metadata"]["seed"] == 42

def test_invalid_node(client, example_graph):
    """Test requesting distribution for non-existent node."""
    # Load the graph
    cgm.viz._state._current_graph = example_graph
    
    # Generate some samples first
    sample_request = {
        "method": "forward",
        "num_samples": 100,
        "conditions": {}
    }
    client.post("/api/sample", json=sample_request)
    
    # Try to get distribution for invalid node
    response = client.get("/api/node_distribution/nonexistent")
    assert response.status_code == 400
    assert "Unknown node" in response.json()["detail"]

def test_sampling_with_conditions(client, example_graph):
    """Test sampling with conditions set."""
    # Load the graph
    cgm.viz._state._current_graph = example_graph
    
    # Set a condition
    response = client.post("/condition/rain/1")
    assert response.status_code == 200
    
    # Generate samples with condition - this should fail because 'season' is not conditioned
    sample_request = {
        "method": "forward",
        "num_samples": 1000,
        "conditions": {"rain": 1},
        "options": {"random_seed": 42}
    }
    
    response = client.post("/api/sample", json=sample_request)
    assert response.status_code == 400
    error_detail = response.json()["detail"]
    assert "Cannot use forward sampling with these conditions" in error_detail
    assert "Node rain is conditioned but its ancestor season is not conditioned" in error_detail
    assert "Forward sampling requires that if a node is conditioned, all of its ancestors must also be conditioned" in error_detail

@pytest.fixture(autouse=True)
def cleanup():
    """Clean up the state after each test."""
    yield
    cgm.viz._state._current_graph = None
    cgm.viz._state._graph_state = None
    cgm.viz._state.clear_samples() 