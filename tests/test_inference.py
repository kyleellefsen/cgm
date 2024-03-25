"""
Inside 'tests' director, run with:
`python -m unittest inference_tests`
"""
import logging
import numpy as np
import numpy.testing as npt
import cgm
logging.basicConfig(level=logging.INFO)

def test_elimination1():
    logging.debug("Running test_elimination1()")
    cg = cgm.example_graphs.get_cg1()
    A, B, C, D, E, F = cg.nodes
    elimination_order = [F, E, D, C, B, A]
    dve = cgm.inference.variable_elimination.DagVariableElimination(cg)
    factors = dve.variable_elimination(elimination_order)
    npt.assert_almost_equal(factors[0].values, 1, decimal=5)
    
    
def test_elimination2():
    logging.debug(f"Running test_elimination2()")
    cg = cgm.example_graphs.get_cg2()
    rain, season, slippery, sprinkler, wet = cg.nodes
    elimination_order = [season, rain, sprinkler, wet]
    dve = cgm.inference.variable_elimination.DagVariableElimination(cg)
    slippery_factor = dve.variable_elimination(elimination_order)[0] 
    logging.debug(f"Remaining factor: {slippery_factor}")
    logging.debug(f"Probabilities of slippery: {slippery_factor.values}")
    npt.assert_almost_equal(slippery_factor.values.sum(), 1, decimal=5)


def generate_clustergraph1():
    """ From Coursera PGM Course 2, Week 2, Video: Belief Propagation 
    Algorithm, Example Cluster Graph (12 minute mark)."""
    rng = np.random.default_rng(30)
    rngs = [np.random.default_rng(seed) for seed in rng.integers(low=0, high=np.iinfo(np.int64).max, size=7)]
    A = cgm.CG_Node('A', 3)
    B = cgm.CG_Node('B', 3)
    C = cgm.CG_Node('C', 3)
    D = cgm.CG_Node('D', 3)
    E = cgm.CG_Node('E', 3)
    F = cgm.CG_Node('F', 3)
    phi1 = cgm.Factor([A, B, C], rng=rngs[0])
    phi2 = cgm.Factor([B, C], rng=rngs[1])
    phi3 = cgm.Factor([B, D], rng=rngs[2])
    phi4 = cgm.Factor([D, E], rng=rngs[3])
    phi5 = cgm.Factor([B, E], rng=rngs[4])
    phi6 = cgm.Factor([B, D], rng=rngs[5])
    phi7 = cgm.Factor([B, D, F], rng=rngs[6])
    psi1 = cgm.inference.Cluster([phi1])
    psi2 = cgm.inference.Cluster([phi2, phi3, phi6])
    psi3 = cgm.inference.Cluster([phi7])
    psi4 = cgm.inference.Cluster([phi5])
    psi5 = cgm.inference.Cluster([phi4])
    cgm.inference.ClusterEdge([psi1, psi2], [C])
    cgm.inference.ClusterEdge([psi1, psi4], [B])
    cgm.inference.ClusterEdge([psi2, psi4], [B])
    cgm.inference.ClusterEdge([psi2, psi5], [D])
    cgm.inference.ClusterEdge([psi3, psi4], [B])
    cgm.inference.ClusterEdge([psi3, psi5], [D])
    cgm.inference.ClusterEdge([psi4, psi5], [E])
    return cgm.inference.ClusterGraph([psi1, psi2, psi3, psi4, psi5])

def generate_clustergraph_chain():
    """ From Coursera PGM Course 2, Week 2, Video: Clique Tree Algorithm 
    - Correctness, Message Passing In Trees (2 minute mark)."""
    rng = np.random.default_rng(30)
    rngs = [np.random.default_rng(seed) for seed in rng.integers(low=0, high=np.iinfo(np.int64).max, size=5)]
    A = cgm.CG_Node('A', 3)
    B = cgm.CG_Node('B', 3)
    C = cgm.CG_Node('C', 3)
    D = cgm.CG_Node('D', 3)
    E = cgm.CG_Node('E', 3)
    phi1 = cgm.CPD([A, B], rng=rngs[0])
    phi2 = cgm.CPD([B, C], rng=rngs[1])
    phi3 = cgm.CPD([D, C], rng=rngs[2])
    phi4 = cgm.CPD([D, E], rng=rngs[3])
    phi5 = cgm.CPD([E], rng=rngs[4])
    psi1 = cgm.inference.Cluster([phi1])
    psi2 = cgm.inference.Cluster([phi2])
    psi3 = cgm.inference.Cluster([phi3])
    psi4 = cgm.inference.Cluster([phi4])
    psi5 = cgm.inference.Cluster([phi5])
    cgm.inference.ClusterEdge([psi1, psi2], [B])
    cgm.inference.ClusterEdge([psi2, psi3], [C])
    cgm.inference.ClusterEdge([psi3, psi4], [D])
    cgm.inference.ClusterEdge([psi4, psi5], [E])
    g = cgm.inference.ClusterGraph([psi1, psi2, psi3, psi4, psi5])
    return g

def test_clustergraph1():
    logging.debug('Creating Variables and Clusters in clusterGraphTest()')
    g = generate_clustergraph1()
    g.propagate_beliefs_round_robin(3)

def test_forward_backward_on_chain():
    logging.debug('Creating Variables and Clusters in test_cliquetree()')
    g = generate_clustergraph_chain()
    psi1 = g.nodes[0]
    A = g.get_variable('A')
    B = g.get_variable('B')
    logging.debug(f'\n{psi1.get_belief().values}')
    g.forward_backward_algorithm()
    p_of_A_estimate = psi1.get_belief().marginalize([B]).normalize().values
    logging.debug(f"Estimated distribution over A: \n{p_of_A_estimate}")
    joint_factor = cgm.Factor.get_null()
    for f in g.factors:
        joint_factor = joint_factor * f
    marginal_factor = joint_factor.marginalize(g.variables - {A})
    p_of_A_true = marginal_factor.values
    logging.debug(f"True distribution over A: \n{p_of_A_true}")
    np.testing.assert_array_almost_equal(p_of_A_true, p_of_A_estimate, decimal=2)

def test_roundrobin_on_chain():
    logging.debug('Creating Variables and Clusters in test_cliquetree()')
    g = generate_clustergraph_chain()
    psi1 = g.nodes[0]
    A = g.get_variable('A')
    B = g.get_variable('B')
    logging.debug(f'\n{psi1.get_belief().values}')
    g.propagate_beliefs_round_robin(5)
    p_of_A_estimate = psi1.get_belief().marginalize([B]).normalize().values
    logging.debug(f"Estimated distribution over A: \n{p_of_A_estimate}")
    joint_factor = cgm.Factor.get_null()
    for f in g.factors:
        joint_factor = joint_factor * f
    marginal_factor = joint_factor.marginalize(g.variables - {A})
    p_of_A_true = marginal_factor.values
    logging.debug(f"True distribution over A: \n{p_of_A_true}")
    np.testing.assert_array_almost_equal(p_of_A_true, p_of_A_estimate, decimal=2)

def test_forward_sample():
    logging.debug('Testing forward_sample()')
    cg = cgm.example_graphs.get_cg2()
    rain, season, slippery, sprinkler, wet = cg.nodes
    sampler = cgm.inference.forward_sampling.ForwardSampler(cg, 30)
    num_samples = 1000
    samples = sampler.get_n_samples(num_samples)
    marginal = sampler.get_sampled_marginal({season})
    expected_season = np.array([.25, .25, .25, .25])
    npt.assert_allclose(marginal.normalize().values,
                        expected_season,
                        rtol=.2)
    logging.debug(f'Season count: {marginal.normalize().values}')
