import numpy as np
from typing import List
import logging
from ..core import Variable, Factor, CG_Node, CPD, CG


class ForwardSampler:
    def __init__(self, cg: CG):
        self.cg = cg
        self.samples = None
        self.current_sample = None

    def forward_sample(self):
        self.current_sample = {n: None for n in self.cg.nodes}
        for n in self.cg.nodes:
            self._sample_node(n)
        return self.current_sample
    
    def _sample_node(self, node: CG_Node):
        for p in node.parents:
            if self.current_sample[p] is None:
                self._sample_node(p)
        parent_states = {n: self.current_sample[n] for n in node.parents}
        sample = node.cpd.sample(parent_states)[0]
        self.current_sample[node] = sample

    def getNSamples(self, N):
        scope = sorted(self.cg.nodes)
        nDims = tuple(n.nStates for n in self.cg.nodes)
        self.samples = Factor(scope, np.zeros(nDims, dtype=np.int))
        for _ in range(N):
            sample = self.forward_sample()
            self.samples.factor[tuple(sample[v] for v in self.samples.scope)] += 1
        return self.samples

    def getSampledMarginal(self, nodes: set):
        return self.samples.marginalize(list(set(self.cg.nodes) - nodes))
