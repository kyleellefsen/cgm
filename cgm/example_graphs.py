import numpy as np
import cgm

def get_cg1():
    np.random.seed(30)
    cg = cgm.CG()
    mkNode = lambda name, n: cgm.CG_Node.from_params(name, n, cg)
    A = mkNode('A', 3)
    B = mkNode('B', 3)
    C = mkNode('C', 3)
    D = mkNode('D', 3)
    E = mkNode('E', 3)
    F = mkNode('F', 3)

    # Define CPDs with all parents at once for each node
    cgm.CPD([B, A])        # B depends on A
    cgm.CPD([C, A])        # C depends on A
    cgm.CPD([D, B, C, E])  # D depends on B, C, and E (combined into one CPD)
    cgm.CPD([F, C])        # F depends on C

    return cg

def get_cg2():
    np.random.seed(30)
    cg = cgm.CG()
    mkNode = lambda name, n: cgm.CG_Node.from_params(name, n, cg)
    # Define all nodes
    season    = mkNode('season', 4)
    rain      = mkNode('rain', 2)
    sprinkler = mkNode('sprinkler', 2)
    wet       = mkNode('wet', 2)
    slippery  = mkNode('slippery', 2)

    # Specify all parents of nodes
    cgm.CPD([season], np.array([.25, .25, .25, .25]))
    cgm.CPD([rain, season])
    cgm.CPD([wet, rain, sprinkler])
    cgm.CPD([slippery, wet])
    return cg