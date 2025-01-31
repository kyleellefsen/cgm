"""Variational inference for CGM"""

from functools import reduce

import numpy as np
import cgm
import cgm.viz


# Define a two node cgm, z for the latent variable and x for the observed variable
# This follows the example from Figure 2.1 of "Active Inference" by Friston et al.

def show_variable_names(table):
    """Replace symbolic values in probability table with human-readable strings."""
    value_translation_dict = {
        "Z⁰": "frog      ",
        "Z¹": "apple     ",
        "X⁰": "doesn't jump",
        "X¹": "jump       "
    }
    return reduce(lambda s, kv: s.replace(*kv), value_translation_dict.items(), str(table))

def compute_exact_posterior(g: cgm.CG):
    """Compute the exact posterior P(Z|X) using Bayes' rule."""
    X, Z = g.nodes
    p_theta, p_Z = X.cpd, Z.cpd
    joint = p_theta * p_Z  # P(Z, X)
    evidence = joint.marginalize([Z])  # i.e. the marginal P(X)
    posterior_factor = joint / evidence
    posterior_factor = posterior_factor.permute_scope([Z, X])
    posterior = g.P(Z | X, posterior_factor.values, virtual=True)
    return posterior


def create_graph():
    g = cgm.CG()  # the causal graph
    Z = g.node('Z', 2) # Z⁰ for frog, Z¹ for apple
    X = g.node('X', 2)  # X⁰ for doesn't jump, X¹ for jump

    prior = g.P(Z, np.array([0.1, 0.9])) # .1 for frog, .9 for apple
    likelihood = g.P(X | Z, np.array([[0.19, 0.81], [0.99, 0.01]]).T)
    cgm.viz.show(g)
    return g

def compute_log_evidence_and_kl_divergence_and_elbo(
        x_i: int,
        g: cgm.CG,
        q_phi: cgm.CPD,
        rng: np.random.Generator,
        num_samples=1000
    ) -> tuple[float, float, float, np.random.Generator]:
    """Compute ELBO for a given observation x_obs."""
    X, Z = g.nodes
    p_theta, p_Z = X.cpd, Z.cpd
    if p_theta is None or p_Z is None:
        raise ValueError("p_theta or p_Z is None")
    elbo = 0
    q_phi_x = q_phi.condition({X: x_i})
    z_samples, rng = q_phi_x.sample(num_samples, rng)
    likelihoods = np.array([p_theta.condition({Z: z_samples[k]}).values[x_i] for k in range(num_samples)])
    priors = np.array([p_Z.values[z_samples[k]] for k in range(num_samples)])
    q_phi_values = np.array([q_phi_x.values[z_samples[k]] for k in range(num_samples)])
    evidence = np.mean(np.exp(np.log(likelihoods) + np.log(priors) - np.log(q_phi_values)))
    log_evidence = np.log(evidence)
    # ELBO is E[log[...]] whereas log_evidence is log[E[...]]. ELBO \leq log_evidence
    elbo = np.mean(np.log(likelihoods) + np.log(priors) - np.log(q_phi_values))


    p_theta_z = compute_exact_posterior(g).condition({X:x_i})
    q_phi_x = q_phi.condition({X:x_i})
    kl_q_p = np.sum(q_phi_x.values * np.log(q_phi_x.values / p_theta_z.values))

    # log_evidence = kl_q_p + elbo

    return log_evidence, kl_q_p, elbo, rng


def main():
    g = create_graph()
    X, Z = g.nodes
    p_theta = X.cpd  # The likelihood. The generative model. 𝑃_theta(X | Z)
    p_Z = Z.cpd  # The prior. 𝑃_Z(Z)

    # Define q_phi(Z|X) - our variational approximation to the posterior
    # Initially, we'll make it uniform to show how it improves
    q_phi_values = np.array([[0.5, 0.5], [0.5, 0.5]])  # [X⁰, X¹] rows, [Z⁰, Z¹] columns
    q_phi = g.P(Z | X, q_phi_values, virtual=True)

    # Example: Observe X=1 (jump)
    x_i = 1
    rng = np.random.default_rng(31)
    print("\nObservation: X=jump")
    num_samples = 10000
    log_evidence, kl_q_p, elbo, rng = compute_log_evidence_and_kl_divergence_and_elbo(x_i, g, q_phi, rng, num_samples)
    # print(f"ELBO: {elbo:.3f}")
    # print(f"KL Divergence: {kl_q_p:.3f}")
    # print(f"Log Evidence: {log_evidence:.3f}")
    # print(f"KL Divergence + ELBO: {kl_q_p + elbo:.3f}")
    print(f"Surprise: {-log_evidence:.3f}")


    # print(f"Variational free energy: {-elbo:.3f}")

    # Let's improve q_phi through gradient descent
    learning_rate = 0.1
    num_iterations = 100

    # Optimization loop
    for iteration in range(num_iterations):
        # Compute gradient (difference between current q and exact posterior)
        p_theta_z = compute_exact_posterior(g).condition({X: x_i})
        current_q = q_phi.condition({X: x_i}).values
        gradient = current_q - p_theta_z.values

        # Update q_phi with gradient descent
        q_phi_values[x_i] -= learning_rate * gradient

        # Project to probability simplex
        q_phi_values[x_i] = np.clip(q_phi_values[x_i], 0, 1)
        q_phi_values[x_i] /= q_phi_values[x_i].sum()

        # Update CPD and recompute metrics
        q_phi = g.P(Z | X, q_phi_values, virtual=True)
        log_evidence, kl_q_p, elbo, rng = compute_log_evidence_and_kl_divergence_and_elbo(
            x_i, g, q_phi, rng, num_samples
        )

        print(f"Iter {iteration}: KL={kl_q_p:.4f}, ELBO={elbo:.4f}")

        # Only print the relevant X=1 row
        print("Exact posterior for X=1:")
        print(compute_exact_posterior(g).condition({X: x_i}).table)
        print("Variational approximation for X=1:")
        print(q_phi.condition({X: x_i}).table)

        print(compute_exact_posterior(g).table)
        print(q_phi.table)


# if __name__ == "__main__":
#     graph = create_graph()
#     compute_exact_posterior(graph)
    


