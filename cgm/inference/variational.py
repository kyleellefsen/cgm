"""Variational inference for CGM"""

from functools import reduce
from dataclasses import dataclass, field
import time
import numpy as np
import cgm
import cgm.viz

@dataclass
class Distribution:
    """Distribution state for visualization"""
    values: list[float]
    parents: list[str]
    type: str = "categorical"

@dataclass
class Message:
    """Message passing state for visualization"""
    from_node: str
    to_node: str
    type: str  # 'gradient' or 'belief'
    values: list[float]

@dataclass
class VisualizationNode:
    """Node state for visualization"""
    id: str
    name: str
    type: str  # 'observed' or 'latent'
    numStates: int
    distributions: dict[str, Distribution]

@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics"""
    iteration_time: float = 0.0
    num_numerical_warnings: int = 0
    max_gradient_norm: float = 0.0
    min_probability: float = 1.0

@dataclass
class OptimizationStep:
    """Single optimization step state"""
    step: int
    elbo: float
    kl: float
    performance: PerformanceMetrics

@dataclass
class OptimizationState:
    """Optimization state for visualization"""
    step: int = 0
    elbo: float = 0
    klDivergence: float = 0
    learningRate: float = 0.1
    history: list[OptimizationStep] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)

@dataclass
class SystemState:
    """Overall system state for visualization"""
    nodes: list[VisualizationNode]
    optimization: OptimizationState


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

    # Compute ELBO using importance sampling
    q_phi_x = q_phi.condition({X: x_i})
    z_samples, rng = q_phi_x.sample(num_samples, rng)

    # Compute components with numerical stability
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
    # Initialize visualization server
    cgm.viz.start_server()

    # Create graph
    g = create_graph()
    X, Z = g.nodes
    p_theta = X.cpd  # The likelihood. The generative model. 𝑃_theta(X | Z)
    p_Z = Z.cpd  # The prior. 𝑃_Z(Z)

    # Define q_phi(Z|X) - our variational approximation to the posterior
    # Initially, we'll make it uniform to show how it improves
    q_phi_values = np.array([[0.5, 0.5], [0.5, 0.5]])  # [X⁰, X¹] rows, [Z⁰, Z¹] columns
    q_phi = g.P(Z | X, q_phi_values, virtual=True)

    # Create visualization state with all nodes including q_phi
    graph_state = cgm.viz.GraphState(
        nodes=[
            {"id": Z.name, "states": Z.num_states, "cpd": p_Z.table.html()},
            {"id": X.name, "states": X.num_states, "cpd": p_X_given_Z.table.html()},
            {"id": "q_phi", "states": 2, "cpd": q_phi.table.html()}
        ],
        links=[
            {"source": Z.name, "target": X.name},
            {"source": X.name, "target": "q_phi"}
        ]
    )
    cgm.viz._current_graph = graph_state

    # Show initial graph
    cgm.viz.show(g, open_browser=True)

    # Optimization loop
    num_iterations = 100
    learning_rate = 0.01  # Reduced learning rate for stability

    for iteration in range(num_iterations):
        # Compute exact posterior for comparison
        p_theta = p_X_given_Z
        p_Z = p_Z
        if p_theta is not None and p_Z is not None:
            # Compute joint distribution P(X,Z)
            joint_values = p_theta.values * p_Z.values.reshape(-1, 1)
            # Add small epsilon to avoid log(0)
            joint_values = np.clip(joint_values, 1e-10, 1.0)
            
            # Compute evidence P(X)
            evidence = joint_values.sum(axis=0)
            evidence = np.clip(evidence, 1e-10, 1.0)
            
            # Compute true posterior P(Z|X)
            posterior = joint_values / evidence
            posterior = np.clip(posterior, 1e-10, 1.0)

            # KL divergence between q_phi and true posterior
            kl_div = 0.0
            for x in range(2):  # For each value of X
                q = np.clip(q_phi.values[:, x], 1e-10, 1.0)  # q(Z|X=x)
                p = np.clip(posterior[:, x], 1e-10, 1.0)      # p(Z|X=x)
                kl_div += 0.5 * np.sum(q * np.log(q / p))  # Weight each X value equally

            # ELBO = E_q[log p(x,z)] - E_q[log q(z)]
            q_values = np.clip(q_phi.values, 1e-10, 1.0)
            elbo = (
                np.sum(q_values * np.log(joint_values)) -
                np.sum(q_values * np.log(q_values))
            )

            print(f"Iteration {iteration}:")
            print(f"  KL(q||p) = {kl_div:.4f}")
            print(f"  ELBO = {elbo:.4f}")
            print(f"  q(Z|X=jump) = [{q_phi.values[0,0]:.3f}, {q_phi.values[1,0]:.3f}]")
            print(f"  q(Z|X=doesn't jump) = [{q_phi.values[0,1]:.3f}, {q_phi.values[1,1]:.3f}]")

            # Gradient descent on q_phi
            # Here we're using the exact gradient, but in practice we'd use samples
            grad = np.zeros_like(q_phi.values)
            for x in range(2):  # For each value of X
                for z in range(2):  # For each value of Z
                    # grad = d/dq [ q*log(p) - q*log(q) ]
                    # Add small epsilon to avoid log(0)
                    q_val = np.clip(q_phi.values[z,x], 1e-10, 1.0)
                    joint_val = np.clip(joint_values[z,x], 1e-10, 1.0)
                    grad[z,x] = np.log(joint_val) - np.log(q_val) - 1

            # Update q_phi with gradient ascent
            new_values = q_phi.values + learning_rate * grad
            # Clip values to ensure they stay positive
            new_values = np.clip(new_values, 1e-10, 1.0)
            # Normalize to ensure valid probabilities
            new_values = new_values / new_values.sum(axis=0, keepdims=True)
            q_phi = g.P(Z | X, values=new_values, virtual=True)

            # Update visualization state with new q_phi
            graph_state = cgm.viz.GraphState(
                nodes=[
                    {"id": Z.name, "states": Z.num_states, "cpd": p_Z.table.html()},
                    {"id": X.name, "states": X.num_states, "cpd": p_X_given_Z.table.html()},
                    {"id": "q_phi", "states": 2, "cpd": q_phi.table.html()}
                ],
                links=[
                    {"source": Z.name, "target": X.name},
                    {"source": X.name, "target": "q_phi"}
                ]
            )
            cgm.viz._current_graph = graph_state

            # Show updated graph (without opening new browser window)
            cgm.viz.show(g, open_browser=False)

            # Add small delay to allow visualization to update
            time.sleep(0.1)

    # After optimization loop
    cgm.viz.stop_server()

# if __name__ == "__main__":
#     main()
    


