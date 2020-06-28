import numpy as np
from .core import Variable, Factor, DAG_Node, CPD, DAG


def get_dag1():
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

def get_dag2():
    np.random.seed(30)
    # Define all nodes
    season    = DAG_Node('season', 4)
    rain      = DAG_Node('rain', 2)
    sprinkler = DAG_Node('sprinkler', 2)
    wet       = DAG_Node('wet', 2)
    slippery  = DAG_Node('slippery', 2)

    # Specify all parents of nodes
    CPD(season, [], np.array([.25, .25, .25, .25]))
    CPD(rain, [season])
    CPD(wet, [rain, sprinkler])
    CPD(slippery, [wet])
    graph = DAG([season, rain, sprinkler, wet, slippery])
    return graph