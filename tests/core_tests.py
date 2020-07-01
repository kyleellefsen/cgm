"""
Inside 'tests' director, run with:
`python -m unittest core_tests`
"""
import unittest

from context import cgm
from cgm import DAG_Node, CPD, DAG
from cgm import get_cg1, get_cg2
import numpy as np

class TestDAGCreation(unittest.TestCase):

    def test_generate_graph1(self):
        graph = get_cg1()
        self.assertEqual(len(graph.nodes), 6)
        return graph

if __name__ == '__main__':
    unittest.main()

