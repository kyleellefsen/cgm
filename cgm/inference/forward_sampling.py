"""This module provides a class for performing forward sampling on a graphical 
model.
"""
import numpy as np
import cgm

SampleDict = dict[cgm.CG_Node, int]
PartialSampleDict = dict[cgm.CG_Node, int|None]



class ForwardSampler:
    """A class for performing forward sampling on a graphical model.

    Attributes:
        cg (cgm.CG): The graphical model to perform forward sampling on.
        nodes (list[cgm.CG_Node]): The list of nodes in the graphical model.
        samples (cgm.Factor[cgm.CG_Node]): The factor representing the samples.
        rng (np.random.Generator): The random number generator.

    Example Usage:

        cg = cgm.example_graphs.get_cg1()
        sampler = cgm.ForwardSampler(cg, seed=30)
        samples = sampler.get_n_samples(100)
        print(samples.values)
    


    """

    def __init__(self, cg: cgm.CG, seed: int = 30):
        """Initializes a ForwardSampler object.

        Args:
            cg (cgm.CG): The graphical model to perform forward sampling on.
            seed (int): The seed for the random number generator.
        """
        self.cg: cgm.CG = cg
        self.nodes: list[cgm.CG_Node] = sorted(cg.nodes)
        self.samples: cgm.Factor[cgm.CG_Node] = cgm.Factor[cgm.CG_Node].get_null()
        self.rng = np.random.default_rng(seed)

    def get_n_samples(self, num_samples: int):
        """
        Generates n samples from the graphical model.

        Args:
            num_samples (int): The number of samples to generate.

        Returns:
            cgm.Factor[cgm.CG_Node]: The factor representing the generated samples.
        """
        scope: list[cgm.CG_Node] = sorted(self.cg.nodes)
        num_dims = tuple(n.num_states for n in self.cg.nodes)
        self.samples = cgm.Factor[cgm.CG_Node](scope, np.zeros(num_dims, dtype=int))
        for _ in range(num_samples):
            sample, self.rng = self._forward_sample(self.rng)
            idx: tuple[int, ...] = tuple(sample[v] for v in self.samples.scope)
            self.samples.increment_at_index(idx, 1)
        return self.samples

    def get_sampled_marginal(self, nodes: set):
        """
        Computes the marginal count of the sampled nodes.

        Args:
            nodes (set): The set of nodes to compute the marginal count for.

        Returns:
            cgm.Factor[cgm.CG_Node]: The factor representing the computed marginal count.
        """
        return self.samples.marginalize(list(set(self.cg.nodes) - nodes))

    def _forward_sample(self, rng: np.random.Generator) -> tuple[SampleDict, np.random.Generator]:
        """
        Performs forward sampling on the graphical model.

        Args:
            rng (np.random.Generator): The random number generator.

        Returns:
            SampleDict: The generated sample.
        """
        current_sample: PartialSampleDict = {n: None for n in self.cg.nodes}
        for n in self.cg.nodes:
            current_sample, rng = self._sample_node(n, current_sample, rng)
        populated_sample: SampleDict = {k: v for k, v in current_sample.items() if v is not None}
        return populated_sample, rng

    def _sample_node(self,
                     node: cgm.CG_Node,
                     current_sample: PartialSampleDict,
                     rng: np.random.Generator) -> tuple[PartialSampleDict, np.random.Generator]:
        """
        Samples a node in the graphical model.

        Args:
            node (cgm.CG_Node): The node to sample.
            rng (np.random.Generator): The random number generator.

        Returns:
            np.random.Generator: The updated random number generator.
        """
        for p in node.parents:
            if current_sample[p] is None:
                current_sample, rng = self._sample_node(p, current_sample, rng)
        populated_sample = {k: v for k, v in current_sample.items() if v is not None}
        parent_states = {n: populated_sample[n] for n in node.parents}
        conditioned_node = node.cpd.condition(parent_states)
        sample, rng = conditioned_node.sample(1, rng)
        current_sample[node] = sample.item()
        return current_sample, rng
