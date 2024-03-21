import numpy as np
from .core import CG_Node, CPD, CG


def get_cg1():
    np.random.seed(30)    
    # Define all nodes
    A = CG_Node('A', 3)
    B = CG_Node('B', 3)
    C = CG_Node('C', 3)
    D = CG_Node('D', 3)
    E = CG_Node('E', 3)
    F = CG_Node('F', 3)
    # Specify all parents of nodes
    CPD(B, [A])
    CPD(C, [A])
    CPD(D, [B, C])
    CPD(F, [C])
    CPD(E, [D])
    graph = CG([A, B, C, D, E, F])
    return graph

def get_cg2():
    np.random.seed(30)
    # Define all nodes
    season    = CG_Node('season', 4)
    rain      = CG_Node('rain', 2)
    sprinkler = CG_Node('sprinkler', 2)
    wet       = CG_Node('wet', 2)
    slippery  = CG_Node('slippery', 2)

    # Specify all parents of nodes
    CPD(season, [], np.array([.25, .25, .25, .25]))
    CPD(rain, [season])
    CPD(wet, [rain, sprinkler])
    CPD(slippery, [wet])
    graph = CG([season, rain, sprinkler, wet, slippery])
    return graph