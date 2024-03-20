"""
Inside the main directory, run with:
`python -m pytest tests/core_tests.py`
"""
import numpy as np
import numpy.testing as npt
import cgm

def test_generate_graph1():
    graph = cgm.example_graphs.get_cg1()
    assert len(graph.nodes) == 6

def test_CG_Node():
    A = cgm.CG_Node('A', 2)
    assert A.name == 'A'
    assert A.num_states == 2

def test_CPD_2nodes():
    num_states_A = 2
    num_states_B = 3
    A = cgm.CG_Node('A', num_states_A)
    B = cgm.CG_Node('B', num_states_B)
    phi1 = cgm.CPD(A, [B])
    assert phi1.child == A
    assert phi1.parents == [B]
    assert A.parents == {B}
    margin = phi1.marginalize([phi1.child]).values
    npt.assert_allclose(margin, np.ones_like(margin))
    assert phi1.scope == [A, B]
    assert phi1.values.shape == (num_states_A, num_states_B)

def test_CPD_3nodes():
    num_states_A = 2
    num_states_B = 3
    num_states_C = 4
    A = cgm.CG_Node('A', num_states_A)
    B = cgm.CG_Node('B', num_states_B)
    C = cgm.CG_Node('C', num_states_C)
    phi1 = cgm.CPD(A, [B, C])
    assert phi1.child == A
    assert phi1.parents == [B, C]
    assert A.parents == {B, C}
    margin = phi1.marginalize([phi1.child]).values
    npt.assert_allclose(margin, np.ones_like(margin))
    assert phi1.scope == [A, B, C]
    assert phi1.values.shape == (num_states_A, num_states_B, num_states_C)

def test_CPD_condition():
    num_states_A = 2
    num_states_B = 3
    num_states_C = 4
    A = cgm.CG_Node('A', num_states_A)
    B = cgm.CG_Node('B', num_states_B)
    C = cgm.CG_Node('C', num_states_C)
    values_A1 = np.array([[.3, .8, .2, .7],
                          [.7, .2, .8, .1],
                          [.1, .5, .3, .9]])
    values = np.array([values_A1, 1 - values_A1])
    phi1 = cgm.CPD(A, [B, C], values=values)
    npt.assert_allclose(phi1.values, values)
    selected_B_index = 1
    parent_states = {B: selected_B_index}
    phi1_cond = phi1.condition(parent_states)
    assert phi1_cond.child == A
    assert phi1_cond.parents == [C]
    assert phi1_cond.values.shape == (num_states_A, num_states_C)
    expected_values = np.array([values_A1[selected_B_index, :],
                                1 - values_A1[selected_B_index, :]])
    npt.assert_allclose(phi1_cond.values, expected_values)