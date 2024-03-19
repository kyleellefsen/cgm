"""
Inside the main directory, run with:
`python -m pytest tests/core_tests.py`
"""
import cgm


def test_generate_graph1():
    graph = cgm.example_graphs.get_cg1()
    assert len(graph.nodes) == 6
