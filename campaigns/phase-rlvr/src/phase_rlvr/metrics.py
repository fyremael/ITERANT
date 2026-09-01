from __future__ import annotations

import math


def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator used in code-generation evaluation.

    n is the number of sampled responses and c the number judged correct.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= c <= n:
        raise ValueError("c must lie in [0, n]")
    if not 1 <= k <= n:
        raise ValueError("k must lie in [1, n]")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def length_normalized_policy_loss(
    token_logprob_terms: list[float], advantage: float, alpha: float
) -> float:
    """Reference scalar for the PHASE-RLVR alpha normalization contract.

    Production integration should apply this normalization inside the policy loss,
    not materialize Python lists.
    """
    if not token_logprob_terms:
        raise ValueError("token_logprob_terms cannot be empty")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    length = len(token_logprob_terms)
    return -advantage * sum(token_logprob_terms) / (length**alpha)
