"""
Inside the main directory, run with:
`python -m pytest tests/test_inference_sampling.py`
"""
# pylint: disable=missing-function-docstring,invalid-name,too-many-locals,logging-fstring-interpolation,f-string-without-interpolation
import logging
import numpy as np
import numpy.testing as npt
import cgm
logging.basicConfig(level=logging.INFO)
logging.getLogger('asyncio').setLevel(logging.WARNING)


def test_forward_sample():
    logging.debug('Testing forward_sample()')
    cg = cgm.example_graphs.get_cg2()
    rain, season, slippery, sprinkler, wet = cg.nodes
    sampler = cgm.inference.ForwardSampler(cg, 30)
    num_samples = 1000
    samples = sampler.get_n_samples(num_samples)
    marginal = sampler.get_sampled_marginal({season})
    expected_season = np.array([.25, .25, .25, .25])
    npt.assert_allclose(marginal.normalize().values,
                        expected_season,
                        rtol=.2)
    logging.debug(f'Season count: {marginal.normalize().values}')


def test_forward_sample_no_mutation():
    logging.debug('Testing test_forward_sample_no_mutation()')
    cg = cgm.example_graphs.get_cg2()
    rain, season, slippery, sprinkler, wet = cg.nodes
    parent_list_before = [n.parents for n in cg.nodes]
    sampler = cgm.inference.ForwardSampler(cg, 30)
    num_samples = 10
    samples = sampler.get_n_samples(num_samples)
    parent_list_after = [n.parents for n in cg.nodes]
    for before, after in zip(parent_list_before, parent_list_after):
        assert before == after