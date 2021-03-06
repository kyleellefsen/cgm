"""
Inside 'tests' director, run with:
`python -m unittest inference_tests`
"""
import unittest
import logging
logging.basicConfig(level=logging.INFO)

from context import cgm 
from cgm import *
import numpy as np

class DagVariableEliminationTest(unittest.TestCase):

    def test_elimination1(self):
        logging.debug(f"Running test_elimination1()")
        cg = get_graph1()
        A, B, C, D, E, F = cg.nodes
        elimination_order = [F, E, D, C, B, A]
        dve = DagVariableElimination(cg)
        factors = dve.variable_elimination(elimination_order)
        self.assertAlmostEqual(factors[0].getValues(), 1)
    
    def test_elimination2(self):
        logging.debug(f"Running test_elimination2()")
        cg = get_graph2()
        rain, season, slippery, sprinkler, wet = cg.nodes
        elimination_order = [season, rain, sprinkler, wet]
        dve = DagVariableElimination(cg)
        slippery_factor = dve.variable_elimination(elimination_order)[0] 
        logging.debug(f"Remaining factor: {slippery_factor}")
        logging.debug(f"Probabilities of slippery: {slippery_factor.getValues()}")
        self.assertAlmostEqual(slippery_factor.getValues().sum(), 1)

class ClusterGraphTest(unittest.TestCase):

    def generate_clustergraph1(self):
        """ From Coursera PGM Course 2, Week 2, Video: Belief Propagation 
        Algorithm, Example Cluster Graph (12 minute mark)."""
        np.random.seed(30)
        A = Variable('A', 3)
        B = Variable('B', 3)
        C = Variable('C', 3)
        D = Variable('D', 3)
        E = Variable('E', 3)
        F = Variable('F', 3)
        phi1 = Factor([A, B, C])
        phi2 = Factor([B, C])
        phi3 = Factor([B, D])
        phi4 = Factor([D, E])
        phi5 = Factor([B, E])
        phi6 = Factor([B, D])
        phi7 = Factor([B, D, F])
        psi1 = Cluster([phi1])
        psi2 = Cluster([phi2, phi3, phi6])
        psi3 = Cluster([phi7])
        psi4 = Cluster([phi5])
        psi5 = Cluster([phi4])
        ClusterEdge([psi1, psi2], [C])
        ClusterEdge([psi1, psi4], [B])
        ClusterEdge([psi2, psi4], [B])
        ClusterEdge([psi2, psi5], [D])
        ClusterEdge([psi3, psi4], [B])
        ClusterEdge([psi3, psi5], [D])
        ClusterEdge([psi4, psi5], [E])
        return ClusterGraph([psi1, psi2, psi3, psi4, psi5])

    def generate_clustergraph_chain(self):
        """ From Coursera PGM Course 2, Week 2, Video: Clique Tree Algorithm 
        - Correctness, Message Passing In Trees (2 minute mark)."""
        np.random.seed(30)
        A = BNode('A', 3)
        B = BNode('B', 3)
        C = BNode('C', 3)
        D = BNode('D', 3)
        E = BNode('E', 3) 
        phi1 = CPD(A, [B])
        phi2 = CPD(B, [C])
        phi3 = CPD(C, [D])
        phi4 = CPD(D, [E])
        phi5 = CPD(E, [])
        psi1 = Cluster([phi1])
        psi2 = Cluster([phi2])
        psi3 = Cluster([phi3])
        psi4 = Cluster([phi4])
        psi5 = Cluster([phi5])
        ClusterEdge([psi1, psi2], [B])
        ClusterEdge([psi2, psi3], [C])
        ClusterEdge([psi3, psi4], [D])
        ClusterEdge([psi4, psi5], [E])
        g = ClusterGraph([psi1, psi2, psi3, psi4, psi5])
        return g

    def test_clustergraph1(self):
        logging.debug('Creating Variables and Clusters in clusterGraphTest()')
        g = self.generate_clustergraph1()
        g.propagate_beliefs_round_robin(3)

    def test_forward_backward_on_chain(self):
        logging.debug('Creating Variables and Clusters in test_cliquetree()')
        g = self.generate_clustergraph_chain()
        psi1 = g.nodes[0]
        A = g.get_variable('A')
        B = g.get_variable('B')
        logging.debug(f'\n{psi1.get_belief().getValues()}')
        g.forward_backward_algorithm()
        p_of_A_estimate = psi1.get_belief().marginalize([B]).normalize().getValues()
        logging.debug(f"Estimated distribution over A: \n{p_of_A_estimate}")
        joint_factor = Factor.getNull()
        for f in g.factors:
            joint_factor = joint_factor * f
        marginal_factor = joint_factor.marginalize(g.variables - {A})
        p_of_A_true = marginal_factor.getValues()
        logging.debug(f"True distribution over A: \n{p_of_A_true}")
        np.testing.assert_array_almost_equal(p_of_A_true, p_of_A_estimate, decimal=2)
        
    def test_roundrobin_on_chain(self):
        logging.debug('Creating Variables and Clusters in test_cliquetree()')
        g = self.generate_clustergraph_chain()
        psi1 = g.nodes[0]
        A = g.get_variable('A')
        B = g.get_variable('B')
        logging.debug(f'\n{psi1.get_belief().getValues()}')
        g.propagate_beliefs_round_robin(5)
        p_of_A_estimate = psi1.get_belief().marginalize([B]).normalize().getValues()
        logging.debug(f"Estimated distribution over A: \n{p_of_A_estimate}")
        joint_factor = Factor.getNull()
        for f in g.factors:
            joint_factor = joint_factor * f
        marginal_factor = joint_factor.marginalize(g.variables - {A})
        p_of_A_true = marginal_factor.getValues()
        logging.debug(f"True distribution over A: \n{p_of_A_true}")
        np.testing.assert_array_almost_equal(p_of_A_true, p_of_A_estimate, decimal=2)

class ForwardSamplingTest(unittest.TestCase):

    def test_forward_sample(self):
        logging.debug('Testing forward_sample()')
        cg = get_graph2()
        rain, season, slippery, sprinkler, wet = cg.nodes
        sampler = ForwardSampler(cg)
        samples = sampler.getNSamples(1000)
        marginal = sampler.getSampledMarginal({season})
        logging.debug(f'Season count: {marginal.getValues()}')

        












if __name__ == '__main__':
    unittest.main()