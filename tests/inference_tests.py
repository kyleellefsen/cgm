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
        dag = get_dag1()
        A, B, C, D, E, F = dag.nodes
        elimination_order = [F, E, D, C, B, A]
        dve = DagVariableElimination(dag)
        factors = dve.variable_elimination(elimination_order)
        self.assertAlmostEqual(factors[0].factor, 1)
    
    def test_elimination2(self):
        logging.debug(f"Running test_elimination2()")
        dag = get_dag2()
        rain, season, slippery, sprinkler, wet = dag.nodes
        elimination_order = [season, rain, sprinkler, wet]
        dve = DagVariableElimination(dag)
        slippery_factor = dve.variable_elimination(elimination_order)[0] 
        logging.debug(f"Remaining factor: {slippery_factor}")
        logging.debug(f"Probabilities of slippery: {slippery_factor.factor}")
        self.assertAlmostEqual(slippery_factor.factor.sum(), 1)

class ClusterGraphTest(unittest.TestCase):

    def test_clustergraph(self):
        """ From Coursera PGM Course 2, Week 2, Video: Belief Propagation 
        Algorithm, Example Cluster Graph (12 minute mark)."""
        logging.info('Creating Variables and Clusters in clusterGraphTest()')
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

        g = ClusterGraph([psi1, psi2, psi3, psi4, psi5])
        g.propagate_beliefs_round_robin(10)





if __name__ == '__main__':
    unittest.main()