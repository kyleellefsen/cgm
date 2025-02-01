# pylint: disable=missing-function-docstring,invalid-name
import numpy as np
import numpy.testing as npt
import cgm.io

def test_chain_graph_conversion():
    """Test conversion of a simple chain graph A -> B -> C"""
    cg = cgm.example_graphs.get_chain_graph(3)
    graph_tuple = cgm.io.GraphTuple.from_cg(cg)

    # Test basic structure
    assert len(graph_tuple.nodes) == 3
    assert len(graph_tuple.senders) == 2  # Two edges
    assert len(graph_tuple.receivers) == 2

    # Test edge structure and topological ordering
    # A should be first (no parents), followed by B, then C
    edge_pairs = list(zip(graph_tuple.senders, graph_tuple.receivers))
    assert (0, 1) in edge_pairs  # A->B
    assert (1, 2) in edge_pairs  # B->C

    # Test node states
    npt.assert_array_equal(graph_tuple.node_states, [2, 2, 2])  # Binary nodes

    # Test CPD structure and offsets
    # Prior P(A) has 2 values
    # CPD P(B|A) has 2*2=4 values
    # CPD P(C|B) has 2*2=4 values
    # Total: 10 values
    assert len(graph_tuple.cpd_values) == 10

    # Test retrieving individual CPDs
    cpd_a = graph_tuple.get_node_cpd(0)  # A is first in topological order
    assert cpd_a.shape == (2,)  # Prior on A

    cpd_b = graph_tuple.get_node_cpd(1)  # B is second
    assert cpd_b.shape == (2, 2)  # B|A

    cpd_c = graph_tuple.get_node_cpd(2)  # C is last
    assert cpd_c.shape == (2, 2)  # C|B

def test_fork_graph_conversion():
    """Test conversion of a fork graph A -> (B,C)"""
    cg = cgm.example_graphs.get_fork_graph()
    graph_tuple = cgm.io.GraphTuple.from_cg(cg)

    # Test basic structure
    assert len(graph_tuple.nodes) == 3
    assert len(graph_tuple.senders) == 2
    assert len(graph_tuple.receivers) == 2

    # Test edge structure and topological ordering
    # A should be first (root), B and C follow (order between B and C doesn't matter)
    edge_pairs = list(zip(graph_tuple.senders, graph_tuple.receivers))
    assert (0, 1) in edge_pairs  # A->B
    assert (0, 2) in edge_pairs  # A->C

    # Test CPD shapes by retrieving them
    cpd_shapes = []
    for i in range(3):
        cpd = graph_tuple.get_node_cpd(i)
        cpd_shapes.append(cpd.shape)

    # A should be first in topological order (no parents)
    assert cpd_shapes[0] == (2,)    # P(A)
    # B and C can be in either order (both have A as parent)
    assert {cpd_shapes[1], cpd_shapes[2]} == {(2, 2), (2, 2)}  # P(B|A) and P(C|A)

def test_collider_graph_conversion():
    """Test conversion of a collider graph (A,B) -> C"""
    cg = cgm.example_graphs.get_collider_graph()
    graph_tuple = cgm.io.GraphTuple.from_cg(cg)

    # Test basic structure
    assert len(graph_tuple.nodes) == 3
    assert len(graph_tuple.senders) == 2
    assert len(graph_tuple.receivers) == 2

    # Test edge structure and topological ordering
    # A and B should come before C in topological order
    edge_pairs = list(zip(graph_tuple.senders, graph_tuple.receivers))
    assert (0, 2) in edge_pairs  # A->C
    assert (1, 2) in edge_pairs  # B->C

    # Test CPD offsets are monotonically increasing
    offsets = graph_tuple.cpd_offsets
    assert len(offsets) == 4  # n_nodes + 1
    assert np.all(np.diff(offsets) >= 0)

    # Check CPD for node C (should be last in topological order)
    cpd_c = graph_tuple.get_node_cpd(2)
    assert cpd_c.shape == (2, 2, 2)  # P(C|A,B)

    # Verify parent indices for C
    parents_c = graph_tuple.cpd_parents[2]
    assert 0 in parents_c  # A is parent
    assert 1 in parents_c  # B is parent

def test_cpd_value_preservation():
    """Test that CPD values are preserved in conversion"""
    cg = cgm.example_graphs.get_chain_graph(2)  # Simple A->B

    # Get original CPD values
    orig_cpd_b = cg.nodes[1].cpd.values

    # Convert to graph tuple and get values back
    graph_tuple = cgm.io.GraphTuple.from_cg(cg)
    cpd_b = graph_tuple.get_node_cpd(1)

    # Values should match exactly
    npt.assert_array_almost_equal(orig_cpd_b, cpd_b)

def test_disconnected_nodes():
    """Test handling of disconnected nodes"""
    cg = cgm.example_graphs.get_disconnected_graph()
    graph_tuple = cgm.io.GraphTuple.from_cg(cg)

    # Should have nodes but no edges
    assert len(graph_tuple.nodes) > 0
    assert len(graph_tuple.senders) == 0
    assert len(graph_tuple.receivers) == 0

    # All nodes should have prior CPDs
    for i in range(len(graph_tuple.nodes)):
        cpd = graph_tuple.get_node_cpd(i)
        assert len(cpd.shape) == 1  # Just (num_states,)

def test_variable_num_parents():
    """Test handling of nodes with different numbers of parents"""
    # Create custom graph: A -> B -> C, D -> C
    cg = cgm.CG()
    A = cg.node('A', 2)
    B = cg.node('B', 2)
    C = cg.node('C', 2)
    D = cg.node('D', 2)

    cg.P(A)  # Prior
    cg.P(B | A)  # One parent
    cg.P(D)  # Prior
    cg.P(C | [B, D])  # Two parents

    graph_tuple = cgm.io.GraphTuple.from_cg(cg)

    # Check parent arrays are properly padded
    assert graph_tuple.cpd_parents.shape[1] == 2  # max_parents should be 2

    # Check shapes match expectations
    cpd_b = graph_tuple.get_node_cpd(1)
    assert cpd_b.shape == (2, 2)  # B|A

    cpd_c = graph_tuple.get_node_cpd(2)
    assert cpd_c.shape == (2, 2, 2)  # C|B,D


def test_cpd_dimension_ordering():
    """Test that CPD dimension ordering is preserved through serialization.
    
    This test is critical because CPDs must maintain their exact dimension ordering
    to be valid. The CPD values array's dimensions must correspond to the order of
    variables in the CPD's scope. This test verifies that serialization preserves
    this ordering by using a CPD with:
    1. A specific scope order [B, C, A] different from the natural parent order
    2. Values that encode clear patterns along each dimension
    3. Checking both overall array equality and specific position access
    """
    cg = cgm.CG()

    # Create nodes with different numbers of states
    A = cg.node('A', 3)  # 3 states
    B = cg.node('B', 2)  # 2 states - changed to match value array shape
    C = cg.node('C', 4)  # 4 states

    # Create values array with shape matching B,C,A ordering
    # Shape is (B states, C states, A states) = (2, 4, 3)
    values = np.zeros((2, 4, 3))
    for c in range(4):  # C states
        for a in range(3):  # A states
            # Make P(B=0) start at 0.2 and increase predictably
            values[0,c,a] = 0.2 + 0.1*c + 0.1*a  # P(B=0)
            values[1,c,a] = 1 - values[0,c,a]    # P(B=1)

    # Create CPD with this specific scope ordering
    cpd = cgm.CPD(scope=[B, C, A], values=values, child=B)

    # print(cpd.table)
    #                    𝑃(B | C, A)  |    B⁰    B¹
    #                    ─────────────────────────────
    #                    A⁰, C⁰       |  0.200  0.800
    #                    A⁰, C¹       |  0.300  0.700
    #                    A⁰, C²       |  0.400  0.600
    #                    A⁰, C³       |  0.500  0.500
    #                    A¹, C⁰       |  0.300  0.700
    #                    A¹, C¹       |  0.400  0.600
    #                    A¹, C²       |  0.500  0.500
    #                    A¹, C³       |  0.600  0.400
    #                    A², C⁰       |  0.400  0.600
    #                    A², C¹       |  0.500  0.500
    #                    A², C²       |  0.600  0.400
    #                    A², C³       |  0.700  0.300

    # Convert to graph tuple
    graph_tuple = cgm.io.GraphTuple.from_cg(cg)

    # Get back the CPD
    b_idx = [i for i, n in enumerate(sorted(cg.nodes)) if n.name == 'B'][0]
    recovered_cpd = graph_tuple.get_node_cpd(b_idx)

    # The recovered values should maintain the same dimension ordering
    np.testing.assert_array_equal(values, recovered_cpd)

    # Test specific position access
    assert values[1,2,1] == recovered_cpd[1,2,1]


def test_large_cpd_preservation():
    """Test preservation of CPDs with many parents and states"""
    cg = cgm.CG()

    # Create nodes with different numbers of states
    A = cg.node('A', 3)  # 3 states
    B = cg.node('B', 4)  # 4 states
    C = cg.node('C', 2)  # 2 states
    D = cg.node('D', 3)  # 3 states

    # Create specific CPD values for D|A,B,C with shape (3, 3, 4, 2)
    cpd_values = np.zeros((3, 3, 4, 2))

    # Fill with a clear pattern
    for a in range(3):  # A states
        for b in range(4):  # B states
            for c in range(2):  # C states
                # Base probabilities for D that increase with parent states
                d_probs = np.array([
                    0.2 + 0.1*a,           # D=0: increases with A
                    0.3 + 0.1*b,           # D=1: increases with B
                    0.5 + 0.1*c            # D=2: increases with C
                ])
                # Normalize to ensure valid probabilities
                d_probs = d_probs / np.sum(d_probs)
                cpd_values[:, a, b, c] = d_probs

    # Create the CPD with specific scope ordering
    cpd = cgm.CPD(scope=[D, A, B, C], values=cpd_values, child=D)
    print("\nOriginal CPD:")
    print(cpd.table)

    # Convert to graph tuple
    graph_tuple = cgm.io.GraphTuple.from_cg(cg)

    # Get back the CPD and verify values are preserved exactly
    d_idx = [i for i, n in enumerate(sorted(cg.nodes)) if n.name == 'D'][0]
    recovered_cpd = graph_tuple.get_node_cpd(d_idx)
    np.testing.assert_array_almost_equal(cpd_values, recovered_cpd)