import numpy as np
import cgm

def get_cg1():
    np.random.seed(30)
    g = cgm.CG()
    A = g.node('A', 3)
    B = g.node('B', 3)
    C = g.node('C', 3)
    D = g.node('D', 3)
    E = g.node('E', 3)
    F = g.node('F', 3)

    # Define CPDs with all parents at once for each node
    g.P(B | A)  # B depends on A
    g.P(C | A)  # C depends on A
    g.P(D | [B, C, E])  # D depends on B, C, and E
    g.P(F | C)  # F depends on C

    return g

def get_cg2():
    np.random.seed(30)
    g = cgm.CG()
    # Define all nodes
    season    = g.node('season', 4)
    rain      = g.node('rain', 2)
    sprinkler = g.node('sprinkler', 2)
    wet       = g.node('wet', 2)
    slippery  = g.node('slippery', 2)

    # Specify all parents of nodes
    g.P(season, values=np.array([.25, .25, .25, .25]))
    g.P(rain | season)
    g.P(wet | [rain, sprinkler])
    g.P(slippery | wet)
    return g

def get_chain_graph(n: int):
    """Creates a chain graph A → B → C → ... of length n with binary nodes."""
    np.random.seed(30)
    g = cgm.CG()
    
    # Create n binary nodes
    nodes = [g.node(chr(65 + i), 2) for i in range(n)]
    
    # Create chain structure with priors and CPDs
    g.P(nodes[0])  # Prior for first node
    for i in range(n-1):
        g.P(nodes[i+1] | nodes[i])  # Each node depends on previous
    
    return g

def get_fork_graph():
    """Creates a fork graph A → (B,C) with binary nodes."""
    np.random.seed(30)
    g = cgm.CG()
    
    A = g.node('A', 2)
    B = g.node('B', 2)
    C = g.node('C', 2)
    
    g.P(A)
    g.P(B | A)
    g.P(C | A)
    
    return g

def get_collider_graph():
    """Creates a collider graph (A,B) → C with binary nodes."""
    np.random.seed(30)
    g = cgm.CG()
    
    A = g.node('A', 2)
    B = g.node('B', 2)
    C = g.node('C', 2)
    
    g.P(A)
    g.P(B)
    g.P(C | [A, B])
    
    return g

def get_disconnected_graph():
    """Creates a graph with disconnected binary nodes A, B, C."""
    np.random.seed(30)
    g = cgm.CG()
    
    A = g.node('A', 2)
    B = g.node('B', 2)
    C = g.node('C', 2)
    
    g.P(A)
    g.P(B)
    g.P(C)
    
    return g