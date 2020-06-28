"""
Inside 'tests' director, run with:
`python -m unittest core_tests`
"""
import unittest

from context import cgm
from cgm import DAG_Node, CPD, DAG
import numpy as np

class TestDAGCreation(unittest.TestCase):

    def test_generate_graph1(self):
        np.random.seed(30)    
        # Define all nodes
        A = DAG_Node('A', 3)
        B = DAG_Node('B', 3)
        C = DAG_Node('C', 3)
        D = DAG_Node('D', 3)
        # Specify all parents of nodes
        CPD(B, [A])
        CPD(C, [B])
        CPD(D, [A, B])
        nodes = [A, B, C, D]
        # Create graph
        graph = DAG(nodes)
        self.assertEqual(len(graph.nodes), 4)
        return graph

    
    


if __name__ == '__main__':
    unittest.main()

