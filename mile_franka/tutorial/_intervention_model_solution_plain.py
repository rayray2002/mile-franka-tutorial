"""Reference answer for the intervention model exercise.

This is MILE's actual probit model: a human intervenes when the robot's planned
actions diverge from what the human expects it to do.  The gap is measured by
comparing each policy action sample's log-probability against the expectation of
the log-probability under the mental model's samples, then passing the signed gap
through a Normal CDF.
"""
import torch
import torch.distributions as D

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.sac.policies import SACPolicy

from mile.computational_model import monte_carlo_samples, sum_independent_dims

# Default MILE tuning parameters (Franka-Stack tasks)
_DEFAULT_COST = 2
_DEFAULT_CDF_SCALE = 2.0


def compute_intervention_prob(state, mental_model, policy,
                               cost=_DEFAULT_COST, cdf_scale=_DEFAULT_CDF_SCALE):
    """MILE probit model: p(ν=1|s) ≈ E_{a~π}[Φ(log π(a|s) − E_{â~π̂}[log π(â|s)] − cost)]

    1. Sample 1000 actions from the mental model π̂ (what the human expects the robot to do).
    2. Compute the robot policy's log-probability of those samples → a baseline E[log π(â|s)].
    3. Sample 1000 actions from the robot policy π.
    4. For each sample, compute the signed gap: log π(a|s) − baseline − cost.
    5. Pass the gap through Φ (Normal CDF) to get a per-sample intervention probability.
    6. Average across samples → scalar per state.

    Interpretation: if the policy consistently assigns high log-prob to its own actions
    but low log-prob to what the mental model expects, the gap is large and p(ν=1) → 1.
    When policy ≈ mental model, the gap is near zero, so p(ν=1) is low.
    """
    # Step 1+2: mental-model baseline
    mental_model_samples = monte_carlo_samples(state=state, policy=mental_model, num_samples=1000)
    if isinstance(policy, SACPolicy):
        mu, log_std, _ = policy.actor.get_action_dist_params(state)
        policy_dist = D.Normal(mu, log_std.exp())
    elif isinstance(policy, ActorCriticPolicy):
        dist_obj = policy.get_distribution(state)
        policy_dist = dist_obj.distribution
    else:
        raise ValueError("policy must be SACPolicy or ActorCriticPolicy")
    mental_model_expectation = torch.mean(
        sum_independent_dims(policy_dist.log_prob(mental_model_samples)), dim=0)

    # Step 3+4+5: policy samples → gap → CDF
    policy_samples = monte_carlo_samples(state=state, policy=policy, num_samples=1000)
    cdf_dist = D.Normal(torch.tensor([0.0], device=state.device),
                        torch.tensor([cdf_scale], device=state.device))
    gaps = sum_independent_dims(policy_dist.log_prob(policy_samples)) - mental_model_expectation - cost
    per_sample_prob = cdf_dist.cdf(gaps)

    # Step 6: average
    return torch.mean(per_sample_prob, dim=0)
