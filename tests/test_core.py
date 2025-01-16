"""
Inside the main directory, run with:
`python -m pytest tests/test_core.py`
"""
# pylint: disable=missing-function-docstring,invalid-name
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
    assert phi1.scope == (a, b)
    assert phi1.values.shape == (num_states_a, num_states_b)
    margin = phi1.marginalize([a]).values
    npt.assert_allclose(margin, num_states_a * np.ones((num_states_b,)))
    margin = phi1.marginalize([b]).values
    npt.assert_allclose(margin, num_states_b * np.ones((num_states_a,)))

def test_factor_creation_with_integer_value():
    A = cgm.Variable('A', 2)
    B = cgm.Variable('B', 2)
    C = cgm.Variable('C', 2)
    value = 5
    factor = cgm.Factor([A, B, C], value)
    expected_values = np.full((2, 2, 2), value)
    np.testing.assert_array_equal(factor.values, expected_values)

def test_cpd_creation_with_integer_value():
    cg = cgm.CG()
    num_a_states = 6
    A = cgm.CG_Node.from_params('A', num_a_states, cg)
    B = cgm.CG_Node.from_params('B', 3, cg)
    C = cgm.CG_Node.from_params('C', 3, cg)
    value = 5
    factor = cgm.CPD([A, B, C], value)
    expected_values = np.full((num_a_states, 3, 3), 1 / num_a_states)
    np.testing.assert_array_equal(factor.values, expected_values)

def test_factor_property_immutability():
    num_states_a = 2
    num_states_b = 3
    a = cgm.Variable('a', num_states_a)
    b = cgm.Variable('b', num_states_b)
    phi1 = cgm.Factor[cgm.Variable]([a, b],
                                    np.ones((num_states_a, num_states_b)))
    with pytest.raises(AttributeError):
        phi1.scope = [b, a]
    with pytest.raises(AttributeError):
        phi1.scope.pop()
    with pytest.raises(AttributeError):
        phi1.shape = (9, 8)


def test_cpd_property_immutability():
    num_states_a = 2
    num_states_b = 3
    cg = cgm.CG()
    a = cgm.CG_Node.from_params('a', num_states_a, cg)
    b = cgm.CG_Node.from_params('b', num_states_b, cg)
    values = np.array([[.11, .22, .33],
                       [.89, .78, .67]])
    phi1 = cgm.CPD([a, b], values=values)
    with pytest.raises(AttributeError):
        phi1.child = b
    with pytest.raises(AttributeError):
        phi1.parents = {b}
    with pytest.raises(AttributeError):
        phi1.values = np.zeros((num_states_a, num_states_b))
    with pytest.raises(AttributeError):
        phi1.parents.pop()
    with pytest.raises(AttributeError):
        phi1.scope.pop()

def test_factor_unsorted_scope():
    num_states_a = 2
    num_states_b = 3
    a = cgm.Variable('a', num_states_a)
    b = cgm.Variable('b', num_states_b)
    phi1 = cgm.Factor[cgm.Variable]([b, a],
                                    np.ones((num_states_b, num_states_a)))
    assert phi1.scope == (b, a)
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
    cg = cgm.CG()
    a = cgm.CG_Node.from_params('a', 2, cg)
    assert a.name == 'a'
    assert a.num_states == 2

def test_CPD_2nodes():
    cg = cgm.CG()
    num_states_a = 2
    num_states_b = 3
    a = cgm.CG_Node.from_params('a', num_states_a, cg)
    b = cgm.CG_Node.from_params('b', num_states_b, cg)
    phi1 = cgm.CPD([a, b])
    assert phi1.child == a
    assert phi1.parents == frozenset({b})
    assert a.parents == frozenset({b})
    margin = phi1.marginalize([phi1.child]).values
    npt.assert_allclose(margin, np.ones_like(margin))
    assert phi1.scope == (a, b)
    assert phi1.values.shape == (num_states_a, num_states_b)

def test_CPD_3nodes():
    num_states_a = 2
    num_states_b = 3
    num_states_c = 4
    cg = cgm.CG()
    a = cgm.CG_Node.from_params('a', num_states_a, cg)
    b = cgm.CG_Node.from_params('b', num_states_b, cg)
    c = cgm.CG_Node.from_params('c', num_states_c, cg)
    phi1 = cgm.CPD([a,b,c])
    assert phi1.child == a
    assert phi1.parents == frozenset({b, c})
    assert a.parents == frozenset({b, c})
    margin = phi1.marginalize([phi1.child]).values
    npt.assert_allclose(margin, np.ones_like(margin))
    assert phi1.scope == (a, b, c)
    assert phi1.values.shape == (num_states_a, num_states_b, num_states_c)

def test_CPD_condition():
    num_states_a = 2
    num_states_b = 3
    num_states_c = 4
    cg = cgm.CG()
    a = cgm.CG_Node.from_params('a', num_states_a, cg)
    b = cgm.CG_Node.from_params('b', num_states_b, cg)
    c = cgm.CG_Node.from_params('c', num_states_c, cg)
    values_a1 = np.array([[.3, .8, .2, .7],
                          [.7, .2, .8, .1],
                          [.1, .5, .3, .9]])
    values = np.array([values_a1, 1 - values_a1])
    phi1 = cgm.CPD([a, b, c], values=values)
    npt.assert_allclose(phi1.values, values)
    selected_b_index = 1
    parent_states = {b: selected_b_index}
    phi1_cond = phi1.condition(parent_states)
    assert phi1_cond.child == a
    assert phi1_cond.parents == {c}
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
    assert phi1_cond.scope == (a, c)
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
    assert result.scope == (a, b)
    assert result.values.shape == (num_states_a, num_states_b)

def test_factor_arithmetic():
    num_states_a = 2
    num_states_b = 3
    ones = np.ones((num_states_a, num_states_b))
    a = cgm.Variable('a', num_states_a)
    b = cgm.Variable('b', num_states_b)
    phi1 = cgm.Factor[cgm.Variable]([a, b], ones)
    phi2 = cgm.Factor[cgm.Variable]([b, a], 2 * np.ones((num_states_b, num_states_a)))
    npt.assert_allclose((phi1 * phi2).values, 2 * ones)
    npt.assert_allclose((phi1 / phi2).values, (1/2) * ones)
    npt.assert_allclose((phi1 + phi2).values, 3 * ones)
    npt.assert_allclose((phi1 - phi2).values, -1 * ones)
    npt.assert_allclose((5 * phi1).values, 5 * ones)
    npt.assert_allclose((phi1 * 5).values, 5 * ones)
    npt.assert_allclose((3 + phi1).values, 4 * ones)
    npt.assert_allclose((phi1 + 3).values, 4 * ones)
    npt.assert_allclose((phi1 / 5).values, (1/5) * ones)
    npt.assert_allclose((phi1 - 5).values, (-4) * ones)

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
    assert result.scope == (a, b, c)
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
    assert result.scope == (a, b)
    assert result.values.shape == (num_states_a, num_states_b)
    assert np.allclose(result.values, np.ones((num_states_a, num_states_b)))

def test_factor_division_by_scalar():
    num_states_a = 2
    num_states_b = 3
    a = cgm.Variable('a', num_states_a)
    b = cgm.Variable('b', num_states_b)
    phi = cgm.Factor[cgm.Variable]([a, b], np.ones((num_states_a, num_states_b)))
    result = phi / 2
    assert result.scope == (a, b)
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
    assert result.scope == (a, b, c)
    assert result.values.shape == (num_states_a, num_states_b, num_states_c)
    assert np.allclose(result.values,
                       np.divide(np.ones((num_states_a, num_states_b, num_states_c)),
                                 np.ones((num_states_b, num_states_c))))

def test_factor_max():
    num_states_a = 2
    num_states_b = 3
    a = cgm.Variable('a', num_states_a)
    b = cgm.Variable('b', num_states_b)
    phi = cgm.Factor[cgm.Variable]([a, b], np.array([[1, 2, 10], [4, 5, 6]]))
    result = phi.max(a)
    assert result.scope == (b,)
    assert result.values.shape == (num_states_b,)
    assert np.allclose(result.values, np.array([4, 5, 10]))
    result = phi.max(b)
    assert result.scope == (a,)
    assert result.values.shape == (num_states_a,)
    assert np.allclose(result.values, np.array([10, 6]))

def test_factor_argmax():
    num_states_a = 2
    num_states_b = 3
    a = cgm.Variable('a', num_states_a)
    b = cgm.Variable('b', num_states_b)
    phi = cgm.Factor[cgm.Variable]([a, b], np.array([[1, 2, 3], [4, 5, 6]]))
    result = phi.argmax(a)
    assert result.scope == (b,)
    assert result.values.shape == (num_states_b,)
    assert np.allclose(result.values, np.array([1, 1, 1]))
    result = phi.argmax(b)
    assert result.scope == (a,)
    assert result.values.shape == (num_states_a,)
    assert np.allclose(result.values, np.array([2, 2]))

def test_cpd_marginalize_2vars():
    num_states_a = 2
    num_states_b = 3
    cg = cgm.CG()
    a = cgm.CG_Node.from_params('a', num_states_a, cg)
    b = cgm.CG_Node.from_params('b', num_states_b, cg)
    values_a = np.array([[.11, .22, .33],
                         [.89, .78, .67]])
    phi1 = cgm.CPD([a, b], values=values_a)
    values_b = np.array([.1, .5, .4])
    phi2 = cgm.CPD([b], values=values_b)
    marginalized_cpd = phi1.marginalize_cpd(phi2)
    expected_values = values_a @ values_b
    npt.assert_allclose(marginalized_cpd.values, expected_values)

def test_cpd_marginalize_3vars():
    num_states_a = 2
    num_states_b = 3
    num_states_c = 4
    cg = cgm.CG()
    a = cgm.CG_Node.from_params('a', num_states_a, cg)
    b = cgm.CG_Node.from_params('b', num_states_b, cg)
    c = cgm.CG_Node.from_params('c', num_states_c, cg)
    values_a1 = np.array([[.3, .8, .2, .7],
                          [.7, .2, .8, .1],
                          [.1, .5, .3, .9]])
    values_a = np.array([values_a1, 1 - values_a1])
    phi1 = cgm.CPD([a, b, c], values=values_a)

    values_bc = np.array([[.1, .1, .1, .1],
                          [.3, .3, .3, .3],
                          [.6, .6, .6, .6]])
    phi2 = cgm.CPD([b, c], values=values_bc)
    marginalized_cpd = phi1.marginalize_cpd(phi2)

    assert marginalized_cpd.child == a
    assert marginalized_cpd.parents == {c}
    assert marginalized_cpd.values.shape == (num_states_a, num_states_c)

    summand = np.zeros((num_states_a, num_states_c))
    for b_idx in range(num_states_b):
        summand += values_a[:, b_idx, :] * values_bc[b_idx, :]
    npt.assert_allclose(marginalized_cpd.values, summand)

def test_cpd_marginalize():
    num_states_x = 2
    num_states_y = 3
    cg = cgm.CG()
    x = cgm.CG_Node.from_params('X', num_states_x, cg)
    y = cgm.CG_Node.from_params('Y', num_states_y, cg)
    phi1 = cgm.Factor([x, y], np.ones((num_states_x, num_states_y)))
    cpd = cgm.CPD([y, x], np.ones((num_states_y, num_states_x)))
    result = phi1.marginalize_cpd(cpd)
    assert result.scope == (x,)
    assert result.values.shape == (num_states_x,)
    assert np.allclose(result.values, np.ones((num_states_x,)))

def test_cpd_marginalize_two_parents():
    num_states_x = 2
    num_states_y = 3
    num_states_z = 4
    cg = cgm.CG()
    x = cgm.CG_Node.from_params('X', num_states_x, cg)
    y = cgm.CG_Node.from_params('Y', num_states_y, cg)
    z = cgm.CG_Node.from_params('Z', num_states_z, cg)
    phi1 = cgm.Factor([x, y, z], np.ones((num_states_x, num_states_y, num_states_z)))
    cpd = cgm.CPD([y, x, z], np.ones((num_states_y, num_states_x, num_states_z)))
    result = phi1.marginalize_cpd(cpd)
    assert set(result.scope) == set([x, z])
    assert result.values.shape == (num_states_x, num_states_z)
    assert np.allclose(result.values, np.ones((num_states_x, num_states_z)))   

def test_factor_permute_scope_2d():
    num_states_a = 2
    num_states_b = 3
    a = cgm.Variable('a', num_states_a)
    b = cgm.Variable('b', num_states_b)
    values = np.array([[0, 1, 2],
                       [3, 4, 5]])
    factor = cgm.Factor([a, b], values)
    new_scope = (b, a)
    new_factor = factor.permute_scope(new_scope)
    assert new_factor.scope == new_scope
    assert new_factor.values.shape == (num_states_b, num_states_a)
    npt.assert_allclose(new_factor.values, np.array([[0, 3],
                                                     [1, 4],
                                                     [2, 5]]))
def test_factor_permute_scope_is_copy():
    num_states_a = 2
    num_states_b = 3
    a = cgm.Variable('a', num_states_a)
    b = cgm.Variable('b', num_states_b)
    values = np.zeros((num_states_a, num_states_b))
    phi1 = cgm.Factor([a, b], values)
    new_scope = (b, a)
    phi2 = phi1.permute_scope(new_scope)
    phi1.values[0, 0] = 1
    npt.assert_allclose(phi2.values, np.zeros((num_states_b, num_states_a)))

def test_factor_set_scope_is_copy():
    num_states_a = 2
    num_states_b = 3
    a = cgm.Variable('a', num_states_a)
    b = cgm.Variable('b', num_states_b)
    c = cgm.Variable('c', num_states_a)
    d = cgm.Variable('d', num_states_b)
    values = np.zeros((num_states_a, num_states_b))
    phi1 = cgm.Factor([a, b], values)
    new_scope = (c, d)
    phi2 = phi1.set_scope(new_scope)
    phi1.values[0, 0] = 1
    npt.assert_allclose(phi2.values, np.zeros((num_states_a, num_states_b)))

def test_factor_permute_scope_3d():
    num_states_a = 1
    num_states_b = 2
    num_states_c = 3
    a = cgm.Variable('a', num_states_a)
    b = cgm.Variable('b', num_states_b)
    c = cgm.Variable('c', num_states_c)
    values = np.array([[[0, 1, 2],
                        [4, 5, 6]]])
    factor = cgm.Factor([a, b, c], values)
    new_scope = (c, a, b)
    new_factor = factor.permute_scope(new_scope)
    assert new_factor.scope == new_scope
    assert new_factor.values.shape == (num_states_c, num_states_a, num_states_b)
    npt.assert_allclose(new_factor.values, np.array([[[0, 4]],
                                                     [[1, 5]],
                                                     [[2, 6]]]))
def test_cpd_permute_scope():
    num_states_a = 2
    num_states_b = 3
    num_states_c = 4
    cg = cgm.CG()
    a = cgm.CG_Node.from_params('a', num_states_a, cg)
    b = cgm.CG_Node.from_params('b', num_states_b, cg)
    c = cgm.CG_Node.from_params('c', num_states_c, cg)
    values_a1 = np.array([[.3, .8, .2, .7],
                          [.7, .2, .8, .1],
                          [.1, .5, .3, .9]])
    values = np.array([values_a1, 1 - values_a1])
    cpd = cgm.CPD([a, b, c], values)

    new_scope = (c, a, b)
    permuted_cpd = cpd.permute_scope(new_scope)
    assert isinstance(permuted_cpd, cgm.CPD)

    assert permuted_cpd.scope == new_scope
    assert permuted_cpd.child == a
    assert permuted_cpd.parents == {c, b}
    assert np.array_equal(permuted_cpd.values, 
                          np.moveaxis(values, [0, 1, 2], [1, 2, 0]))

    # Test assertion error when new_scope is not a permutation of the original scope
    with pytest.raises(AssertionError):
        invalid_scope = [a, b]
        cpd.permute_scope(invalid_scope)

def test_cpd_set_scope():
    num_states_a = 2
    num_states_b = 3
    cg = cgm.CG()
    a = cgm.CG_Node.from_params('a', num_states_a, cg)
    b = cgm.CG_Node.from_params('b', num_states_b, cg)
    c = cgm.CG_Node.from_params('c', num_states_a, cg)
    d = cgm.CG_Node.from_params('d', num_states_b, cg)
    values = np.array([[0, .6, .9],
                       [1, .4, .1]])
    phi1 = cgm.CPD([a, b], values)
    new_scope = [c, d]
    phi2 = phi1.set_scope(new_scope)
    assert isinstance(phi2, cgm.CPD)
    assert phi2.child == c
    assert phi2.parents == {d}
    npt.assert_allclose(phi2.values, values)

def test_cpd_condition_mutability():
    """Ensures conditioning doesn't change the parents of any node."""
    cg = cgm.example_graphs.get_cg2()
    rain, season, slippery, sprinkler, wet = cg.nodes
    parent_list_before = [n.parents for n in cg.nodes]
    rain.cpd.condition({season: 0})
    parent_list_after = [n.parents for n in cg.nodes]
    for before, after in zip(parent_list_before, parent_list_after):
        assert before == after


def test_variable_equality():
    """Test that Variable equality works correctly"""
    v1 = cgm.Variable('a', 2)
    v2 = cgm.Variable('a', 2)
    v3 = cgm.Variable('a', 3)
    v4 = cgm.Variable('b', 2)
    
    assert v1 == v2  # Same name and states
    assert v1 != v3  # Same name, different states
    assert v1 != v4  # Different name
    assert len({v1, v2}) == 1  # Test hash equality
    assert len({v1, v3}) == 2  # Test hash inequality

def test_cg_node_equality():
    """Test that CG_Node equality works correctly"""
    cg = cgm.CG()
    a1 = cgm.CG_Node.from_params('a', 2, cg)
    a2 = cgm.CG_Node.from_params('a', 2, cg)
    b = cgm.CG_Node.from_params('b', 2, cg)
    
    assert a1 == a2  # Same name, states, no parents
    assert a1 != b   # Different name
    assert len({a1, a2}) == 1  # Test hash equality
    
    # Create CPD to add parent relationship
    phi = cgm.CPD([a1, b])
    a3 = phi.child  # New node with b as parent
    
    assert a1 == a3  # different parents but same node
    assert len({a1, a3}) == 1  # Test hash equality

def test_dag_node_equality():
    """Test that DAG_Node equality works correctly"""
    a = cgm.Variable('a', 2)
    b = cgm.Variable('b', 2)
    dag = cgm.DAG()
    d1 = cgm.DAG_Node[cgm.CG_Node](variable=a, dag=dag)
    d2 = cgm.DAG_Node[cgm.CG_Node](variable=a, dag=dag)
    d3 = cgm.DAG_Node[cgm.CG_Node](variable=b, dag=dag)
    
    assert d1 == d2  # Same variable, no parents
    assert d1 != d3  # Different variable
    assert len({d1, d2}) == 1  # Test hash equality
    assert len({d1, d3}) == 2  # Test hash inequality