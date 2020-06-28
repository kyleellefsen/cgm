"""
Inside 'tests' director, run with:
`python -m unittest inference_tests`
"""
import unittest
import logging
logging.basicConfig(level=logging.DEBUG)

from context import cgm 
from cgm import *
import numpy as np

class TestInference(unittest.TestCase):

    def test_elimination(self):
        print('Running test_elimination()')
        graph = get_graph1()
        A, B, C, D, E, F = graph.nodes
        elimination_order = [F, E, D, C, B, A]
        factors = eliminate_nodes(elimination_order, graph)
        self.assertAlmostEqual(factors[0].factor, 1)


if __name__ == '__main__':
    unittest.main()