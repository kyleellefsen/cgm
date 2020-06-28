import numpy as np
from .core import Variable, Factor, DAG_Node, CPD, DAG


def get_graph1():
    np.random.seed(30)    
    # Define all nodes
    A = DAG_Node('A', 3)
    B = DAG_Node('B', 3)
    C = DAG_Node('C', 3)
    D = DAG_Node('D', 3)
    E = DAG_Node('E', 3)
    F = DAG_Node('F', 3)
    # Specify all parents of nodes
    CPD(B, [A])
    CPD(C, [A])
    CPD(D, [B, C])
    CPD(F, [C])
    CPD(E, [D])
    graph = DAG([A, B, C, D, E, F])
    return graph