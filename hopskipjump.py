import torch
torch.backends.mkldnn.enabled = False
torch.backends.nnpack.enabled = False
from typing import Callable, Optional, Sequence, Tuple


def hopskipjump_attack(
    predict_fn: Callable[[torch.Tensor], int],
    x_orig: torch.Tensor,
    y_orig: int,
    n_steps: int = 30,
    n_grad_samples: int = 30,
    n_init_trials: int = 100,
    init_noise_scale: float = 3.0,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    feature_indices: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, int]:
    """
    HopSkipJump Attack (Chen et al., 2020) — decision-based black-box attack.

    Requires only hard-label query access to the classifier (predict_fn).
    Starting from a randomly found misclassified point, alternates between:
      1. Binary search toward the decision boundary from the clean sample.
      2. Random-direction gradient estimation at the boundary.
      3. A geometric-decay step along the estimated gradient toward the
         clean input, keeping the perturbed point on the adversarial side.

    Args:
        predict_fn:        callable (x: 1-D tensor) -> predicted class int.
                           Called once per query; the caller is responsible
                           for model locking inside this function.
        x_orig:            clean input tensor, shape (n_features,).
        y_orig:            true class label of x_orig.
        n_steps:           boundary-walk iterations.
        n_grad_samples:    random directions per gradient estimate.
        n_init_trials:     max noise samples to find initial adversarial point.
        init_noise_scale:  std of Gaussian noise used during initialisation.
        clip_min/clip_max: optional per-feature clipping for physical
                           plausibility (e.g. clip_min=0.0 for pressures).
        feature_indices:   optional subset of feature indices the attack is
                           allowed to perturb. All other coordinates of
                           x_orig are left untouched throughout the attack
                           (random init noise, gradient-probe directions and
                           steps are zeroed outside this subset). If None,
                           all features are perturbable.

    Returns:
        (x_adv, n_queries): adversarial tensor and total query count.
        x_adv == x_orig (cloned) if no adversarial point was found.
    """
    n_queries = 0

    def _clip(x: torch.Tensor) -> torch.Tensor:
        if clip_min is not None or clip_max is not None:
            return torch.clamp(x, min=clip_min, max=clip_max)
        return x

    if feature_indices is not None:
        mask = torch.zeros_like(x_orig)
        mask[list(feature_indices)] = 1.0
    else:
        mask = None

    def _masked_randn() -> torch.Tensor:
        noise = torch.randn_like(x_orig)
        if mask is not None:
            noise = noise * mask
        return noise

    # ── Phase 1: find initial adversarial starting point ──────────────────
    x_adv = None
    for _ in range(n_init_trials):
        x_candidate = _clip(x_orig + _masked_randn() * init_noise_scale)
        n_queries += 1
        if predict_fn(x_candidate) != y_orig:
            x_adv = x_candidate.clone()
            break

    if x_adv is None:
        return x_orig.clone(), n_queries

    # ── Phase 2: boundary walk ─────────────────────────────────────────────
    for step in range(n_steps):

        # Binary search between x_orig (correctly classified)
        # and x_adv (adversarially classified)
        x_lo, x_hi = x_orig.clone(), x_adv.clone()
        for _ in range(20):
            x_mid = _clip((x_lo + x_hi) * 0.5)
            n_queries += 1
            if predict_fn(x_mid) == y_orig:
                x_lo = x_mid
            else:
                x_hi = x_mid
                x_adv = x_mid

        # Gradient estimation at the boundary via random binary queries
        dist = float(torch.norm(x_adv - x_orig))
        if dist < 1e-9:
            break
        # Step size for random perturbation scales with distance and
        # dimensionality so probes stay near the boundary.
        n_dims = float(mask.sum().item()) if mask is not None else float(x_orig.numel())
        delta = dist / (n_dims ** 0.5)

        grad_est = torch.zeros_like(x_orig)
        for _ in range(n_grad_samples):
            u = _masked_randn()
            u = u / (u.norm() + 1e-12)
            x_probe = _clip(x_adv + delta * u)
            n_queries += 1
            # u points "away from clean": if probe is still adversarial,
            # the boundary is in that direction; else it flipped back.
            grad_est += u if predict_fn(x_probe) != y_orig else -u
        grad_est /= n_grad_samples

        # Geometric-decay step: move x_adv toward x_orig along grad_est.
        # Smaller steps as we get closer (step+1 denominator).
        step_size = dist / float((step + 1) ** 0.5)
        x_new = _clip(x_adv - step_size * grad_est)
        n_queries += 1
        if predict_fn(x_new) != y_orig:
            x_adv = x_new

    return x_adv, n_queries
