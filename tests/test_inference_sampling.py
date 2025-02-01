"""
Inside the main directory, run with:
`python -m pytest tests/test_inference_sampling.py`
"""
# pylint: disable=missing-function-docstring,invalid-name,too-many-locals,logging-fstring-interpolation,f-string-without-interpolation
import logging
import numpy as np
import numpy.testing as npt
import pytest
import cgm
from scipy import stats

logging.basicConfig(level=logging.INFO)
logging.getLogger('asyncio').setLevel(logging.WARNING)


def test_forward_sample():
    logging.debug('Testing forward_sample()')
    cg = cgm.example_graphs.get_cg2()
    # Get nodes by name instead of relying on order
    nodes_by_name = {n.name: n for n in cg.nodes}
    season = nodes_by_name['season']
    
    # Generate samples using the functional interface
    key = np.random.default_rng(30)
    samples, _ = cgm.inference.get_n_samples(cg, key, num_samples=1000)
    marginal = samples.marginalize(list(set(cg.nodes) - {season}))
    
    expected_season = np.array([.25, .25, .25, .25])
    npt.assert_allclose(marginal.normalize().values,
                        expected_season,
                        rtol=.2)
    logging.debug(f'Season count: {marginal.normalize().values}')


def test_forward_sample_no_mutation():
    logging.debug('Testing test_forward_sample_no_mutation()')
    cg = cgm.example_graphs.get_cg2()
    rain, season, slippery, sprinkler, wet = cg.nodes
    parent_list_before = [n.parents for n in cg.nodes]
    
    # Generate samples using the functional interface
    key = np.random.default_rng(30)
    samples, _ = cgm.inference.get_n_samples(cg, key, num_samples=10)
    
    parent_list_after = [n.parents for n in cg.nodes]
    for before, after in zip(parent_list_before, parent_list_after):
        assert before == after


def test_conditioned_sampling():
    """Test that sampling respects conditions from GraphState"""
    logging.debug('Testing conditioned sampling')
    cg = cgm.example_graphs.get_cg2()
    state = cgm.GraphState.create(cg)
    
    # Condition on season=0
    sample = np.full(state.schema.num_vars, -1)
    season_idx = state.schema.var_to_idx['season']
    sample[season_idx] = 0
    conditioned_state = state.condition_on_sample(sample)
    
    # Generate samples with condition
    key = np.random.default_rng(30)
    samples, _ = cgm.inference.get_n_samples(cg, key, num_samples=1000, state=conditioned_state)
    
    # Verify all samples have season=0
    season_marginal = samples.marginalize(list(set(cg.nodes) - {cg.nodes[season_idx]}))
    season_dist = season_marginal.normalize().values
    
    # Should have all probability mass on state 0
    expected = np.zeros(4)
    expected[0] = 1.0
    npt.assert_allclose(season_dist, expected)


def test_random_key_splitting():
    """Test that random key splitting produces different but deterministic results"""
    logging.debug('Testing random key splitting')
    cg = cgm.example_graphs.get_cg2()
    key = np.random.default_rng(30)
    
    # Generate two sets of samples with the same key
    samples1, key = cgm.inference.get_n_samples(cg, key, num_samples=100)
    samples2, _ = cgm.inference.get_n_samples(cg, key, num_samples=100)
    
    # Samples should be different but deterministic
    assert not np.array_equal(samples1.values, samples2.values)
    
    # Repeating with same initial key should give same results
    key = np.random.default_rng(30)
    samples3, _ = cgm.inference.get_n_samples(cg, key, num_samples=100)
    npt.assert_array_equal(samples1.values, samples3.values)


def test_multiple_conditions():
    """Test sampling with multiple conditioned variables"""
    cg = cgm.example_graphs.get_cg2()
    state = cgm.GraphState.create(cg)
    schema = state.schema
    
    # Condition on both season and rain
    sample = np.full(schema.num_vars, -1)
    sample[schema.var_to_idx['season']] = 0
    sample[schema.var_to_idx['rain']] = 1
    conditioned_state = state.condition_on_sample(sample)
    
    # Generate samples
    key = np.random.default_rng(30)
    samples, _ = cgm.inference.get_n_samples(cg, key, num_samples=1000, state=conditioned_state)
    
    # Verify both conditions are respected
    for var_name, expected_value in [('season', 0), ('rain', 1)]:
        var_idx = schema.var_to_idx[var_name]
        marginal = samples.marginalize(list(set(cg.nodes) - {cg.nodes[var_idx]}))
        dist = marginal.normalize().values
        expected = np.zeros(cg.nodes[var_idx].num_states)
        expected[expected_value] = 1.0
        npt.assert_allclose(dist, expected)


def test_statistical_properties():
    """Test statistical properties of the sampler"""
    # Use a simple chain A -> B -> C for statistical tests
    cg = cgm.example_graphs.get_chain_graph(3)
    key = np.random.default_rng(30)
    
    # Generate a large number of samples for statistical tests
    num_samples = 10000
    samples, _ = cgm.inference.get_n_samples(cg, key, num_samples=num_samples)
    
    # Test uniformity of root node (A) using max deviation from expected
    a_marginal = samples.marginalize([cg.nodes[1], cg.nodes[2]]).normalize().values
    expected_prob = 0.5  # Binary nodes, should be uniform
    max_deviation = np.max(np.abs(a_marginal - expected_prob))
    assert max_deviation < 0.1, "Root node distribution should be roughly uniform"
    
    # Test conditional independence
    # In a chain A -> B -> C, A and C should be independent given B
    def get_conditional_dist(a_val: int, b_val: int) -> np.ndarray:
        """Get P(C|A=a,B=b) from samples"""
        mask = np.logical_and(
            samples.values[:, 0] == a_val,
            samples.values[:, 1] == b_val
        )
        if not np.any(mask):
            return np.zeros(2)
        return np.bincount(samples.values[mask, 2], minlength=2) / np.sum(mask)
    
    # For each value of B, P(C|A,B) should be the same for all values of A
    for b in range(2):
        dist_a0 = get_conditional_dist(0, b)
        dist_a1 = get_conditional_dist(1, b)
        if np.all(dist_a0 > 0) and np.all(dist_a1 > 0):  # Only test if we have enough samples
            npt.assert_allclose(dist_a0, dist_a1, rtol=0.1)