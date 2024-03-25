import numpy as np
import cgm

def get_cg1():
    np.random.seed(30)
    # Define all nodes
    A = cgm.CG_Node('A', 3)
    B = cgm.CG_Node('B', 3)
    C = cgm.CG_Node('C', 3)
    D = cgm.CG_Node('D', 3)
    E = cgm.CG_Node('E', 3)
    F = cgm.CG_Node('F', 3)
    # Specify all parents of nodes
    cgm.CPD([B, A])
    cgm.CPD([C, A])
    cgm.CPD([D, B, C])
    cgm.CPD([C, F])
    cgm.CPD([D, E])
    graph = cgm.CG([A, B, C, D, E, F])
    return graph

def get_cg2():
    np.random.seed(30)
    # Define all nodes
    season    = cgm.CG_Node('season', 4)
    rain      = cgm.CG_Node('rain', 2)
    sprinkler = cgm.CG_Node('sprinkler', 2)
    wet       = cgm.CG_Node('wet', 2)
    slippery  = cgm.CG_Node('slippery', 2)

    # Specify all parents of nodes
    cgm.CPD([season], np.array([.25, .25, .25, .25]))
    cgm.CPD([rain, season])
    cgm.CPD([wet, rain, sprinkler])
    cgm.CPD([slippery, wet])
    graph = cgm.CG([season, rain, sprinkler, wet, slippery])
    return graph