"""
Inside the main directory, run with:
`python -m pytest tests/core_tests.py`
"""
import pytest
import numpy as np
import numpy.testing as npt
import cgm

def test_generate_graph1():
    graph = cgm.example_graphs.get_cg1()
    assert len(graph.nodes) == 6

def test_factor():
    num_states_a = 2
    num_states_b = 3
    a = cgm.Variable('a', num_states_a)
    b = cgm.Variable('b', num_states_b)
    phi1 = cgm.Factor[cgm.Variable]([a, b], 
                                    np.ones((num_states_a, num_states_b)))
    
    assert phi1.scope == [a, b]
    assert phi1.values.shape == (num_states_a, num_states_b)
    margin = phi1.marginalize([a]).values
    npt.assert_allclose(margin, num_states_a * np.ones((num_states_b,)))
    margin = phi1.marginalize([b]).values
    npt.assert_allclose(margin, num_states_b * np.ones((num_states_a,)))

def test_factor_unsorted_scope():
    num_states_a = 2
    num_states_b = 3
    a = cgm.Variable('a', num_states_a)
    b = cgm.Variable('b', num_states_b)
    phi1 = cgm.Factor[cgm.Variable]([b, a],
                                    np.ones((num_states_b, num_states_a)))
    assert phi1.scope == [b, a]
    assert phi1.values.shape == (num_states_b, num_states_a)
    margin = phi1.marginalize([a]).values
    npt.assert_allclose(margin, num_states_a * np.ones((num_states_b,)))
    margin = phi1.marginalize([b]).values
    npt.assert_allclose(margin, num_states_b * np.ones((num_states_a,)))

def test_factor_incorrect_input_dims():
    """This test should raise an error."""
    num_states_a = 2
    num_states_b = 3
    a = cgm.Variable('a', num_states_a)
    b = cgm.Variable('b', num_states_b)
    with pytest.raises(cgm.ScopeShapeMismatchError):
        cgm.Factor[cgm.Variable]([a, b],
                                 np.ones((num_states_b, num_states_a)))
    phi1 = cgm.Factor[cgm.Variable]([a, b],
                                 np.ones((num_states_a, num_states_b)))
    scope_shape = np.array([s.num_states for s in phi1.scope])
    npt.assert_equal(scope_shape, np.array([num_states_a, num_states_b]))

def test_check_input_with_non_unique_names():
    """Test _check_input with non-unique variable names to ensure it raises 
    NonUniqueVariableNamesError."""
    num_states = 2
    var_name = 'a_duplicate'
    a1 = cgm.Variable(var_name, num_states)
    a2 = cgm.Variable(var_name, num_states)
    with pytest.raises(cgm.NonUniqueVariableNamesError) as exc_info:
        cgm.Factor[cgm.Variable]([a1, a2])
        assert var_name in str(exc_info.value)

def test_CG_Node():
    a = cgm.CG_Node('a', 2)
    assert a.name == 'a'
    assert a.num_states == 2

def test_CPD_2nodes():
    num_states_a = 2
    num_states_b = 3
    a = cgm.CG_Node('a', num_states_a)
    b = cgm.CG_Node('b', num_states_b)
    phi1 = cgm.CPD(a, [b])
    assert phi1.child == a
    assert phi1.parents == [b]
    assert a.parents == {b}
    margin = phi1.marginalize([phi1.child]).values
    npt.assert_allclose(margin, np.ones_like(margin))
    assert phi1.scope == [a, b]
    assert phi1.values.shape == (num_states_a, num_states_b)

def test_CPD_3nodes():
    num_states_a = 2
    num_states_b = 3
    num_states_c = 4
    a = cgm.CG_Node('a', num_states_a)
    b = cgm.CG_Node('b', num_states_b)
    c = cgm.CG_Node('c', num_states_c)
    phi1 = cgm.CPD(a, [b, c])
    assert phi1.child == a
    assert phi1.parents == [b, c]
    assert a.parents == {b, c}
    margin = phi1.marginalize([phi1.child]).values
    npt.assert_allclose(margin, np.ones_like(margin))
    assert phi1.scope == [a, b, c]
    assert phi1.values.shape == (num_states_a, num_states_b, num_states_c)

def test_CPD_condition():
    num_states_a = 2
    num_states_b = 3
    num_states_c = 4
    a = cgm.CG_Node('a', num_states_a)
    b = cgm.CG_Node('b', num_states_b)
    c = cgm.CG_Node('c', num_states_c)
    values_a1 = np.array([[.3, .8, .2, .7],
                          [.7, .2, .8, .1],
                          [.1, .5, .3, .9]])
    values = np.array([values_a1, 1 - values_a1])
    phi1 = cgm.CPD(a, [b, c], values=values)
    npt.assert_allclose(phi1.values, values)
    selected_b_index = 1
    parent_states = {b: selected_b_index}
    phi1_cond = phi1.condition(parent_states)
    assert phi1_cond.child == a
    assert phi1_cond.parents == [c]
    assert phi1_cond.values.shape == (num_states_a, num_states_c)
    expected_values = np.array([values_a1[selected_b_index, :],
                                1 - values_a1[selected_b_index, :]])
    npt.assert_allclose(phi1_cond.values, expected_values)

def test_factor_condition():
    num_states_a = 2
    num_states_b = 3
    num_states_c = 4
    a = cgm.Variable('a', num_states_a)
    b = cgm.Variable('b', num_states_b)
    c = cgm.Variable('c', num_states_c)
    phi1 = cgm.Factor[cgm.Variable]([a, b, c], 5 * np.ones((num_states_a, 
                                                            num_states_b, 
                                                            num_states_c)))
    selected_b_index = 1
    phi1_cond = phi1.condition({b: selected_b_index})
    assert phi1_cond.scope == [a, c]
    assert phi1_cond.values.shape == (num_states_a, num_states_c)
    npt.assert_allclose(phi1_cond.values, 5 * np.ones((num_states_a, 
                                                       num_states_c)))
    
def test_factor_multiplication():
    num_states_a = 2
    num_states_b = 3
    a = cgm.Variable('a', num_states_a)
    b = cgm.Variable('b', num_states_b)
    phi1 = cgm.Factor[cgm.Variable]([a, b], np.ones((num_states_a, num_states_b)))
    phi2 = cgm.Factor[cgm.Variable]([b, a], np.ones((num_states_b, num_states_a)))

    result = phi1 * phi2

    assert result.scope == [a, b]
    assert result.values.shape == (num_states_a, num_states_b)

def test_factor_multiplication_with_different_scopes():
    num_states_a = 2
    num_states_b = 3
    num_states_c = 4
    a = cgm.Variable('a', num_states_a)
    b = cgm.Variable('b', num_states_b)
    c = cgm.Variable('c', num_states_c)
    phi1 = cgm.Factor[cgm.Variable]([a, b], np.ones((num_states_a, num_states_b)))
    phi2 = cgm.Factor[cgm.Variable]([b, c], np.ones((num_states_b, num_states_c)))

    result = phi1 * phi2

    assert result.scope == [a, b, c]
    assert result.values.shape == (num_states_a, num_states_b, num_states_c)
    assert np.allclose(result.values, np.ones((num_states_a, num_states_b, num_states_c)))

def test_factor_multiplication_with_nonintersection_scope():
    num_states_a = 2
    num_states_b = 3
    a = cgm.Variable('a', num_states_a)
    b = cgm.Variable('b', num_states_b)
    phi1 = cgm.Factor[cgm.Variable]([a], np.ones((num_states_a,)))
    phi2 = cgm.Factor[cgm.Variable]([b], np.ones((num_states_b,)))

    result = phi1 * phi2

    assert result.scope == [a, b]
    assert result.values.shape == (num_states_a, num_states_b)
    assert np.allclose(result.values, np.ones((num_states_a, num_states_b)))
    
def test_factor_division_by_scalar():
    num_states_a = 2
    num_states_b = 3
    a = cgm.Variable('a', num_states_a)
    b = cgm.Variable('b', num_states_b)
    phi = cgm.Factor[cgm.Variable]([a, b], np.ones((num_states_a, num_states_b)))

    result = phi / 2

    assert result.scope == [a, b]
    assert result.values.shape == (num_states_a, num_states_b)
    assert np.allclose(result.values, np.divide(np.ones((num_states_a, num_states_b)), 2))

def test_factor_division_by_factor():
    num_states_a = 2
    num_states_b = 3
    num_states_c = 4
    a = cgm.Variable('a', num_states_a)
    b = cgm.Variable('b', num_states_b)
    c = cgm.Variable('c', num_states_c)
    phi1 = cgm.Factor[cgm.Variable]([a, b, c], np.ones((num_states_a, num_states_b, num_states_c)))
    phi2 = cgm.Factor[cgm.Variable]([b, c], np.ones((num_states_b, num_states_c)))

    result = phi1 / phi2

    assert result.scope == [a, b, c]
    assert result.values.shape == (num_states_a, num_states_b, num_states_c)
    assert np.allclose(result.values, np.divide(np.ones((num_states_a, num_states_b, num_states_c)), np.ones((num_states_b, num_states_c))))

def test_factor_argmax():
    num_states_a = 2
    num_states_b = 3
    a = cgm.Variable('a', num_states_a)
    b = cgm.Variable('b', num_states_b)
    phi = cgm.Factor[cgm.Variable]([a, b], np.array([[1, 2, 3], [4, 5, 6]]))

    result = phi.argmax(a)

    assert result.scope == [b]
    assert result.values.shape == (num_states_b,)
    assert np.allclose(result.values, np.array([1, 1, 1]))

    result = phi.argmax(b)

    assert result.scope == [a]
    assert result.values.shape == (num_states_a,)
    assert np.allclose(result.values, np.array([2, 2]))