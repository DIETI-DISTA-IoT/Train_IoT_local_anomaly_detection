import torch
torch.backends.mkldnn.enabled = True
torch.backends.nnpack.enabled = False
import time
import logging
from typing import Callable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


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
    predict_batch_fn: Optional[Callable[[torch.Tensor], Sequence[int]]] = None,
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
        predict_batch_fn:  optional callable (X: 2-D tensor of shape
                           (k, n_features)) -> sequence of k predicted class
                           ints. When provided, the gradient-estimation phase
                           evaluates all n_grad_samples probes in a single
                           batched forward pass instead of n_grad_samples
                           sequential batch-1 queries. The set of probes, their
                           predictions and the resulting gradient estimate are
                           mathematically identical to the sequential path
                           (the random directions are drawn in the same order);
                           only the number of forward passes and model-lock
                           acquisitions is reduced. Falls back to per-query
                           evaluation when None.

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
    for trial in range(n_init_trials):
        x_candidate = _clip(x_orig + _masked_randn() * init_noise_scale)
        n_queries += 1
        if predict_fn(x_candidate) != y_orig:
            x_adv = x_candidate.clone()
            logger.debug(f"HSJA init: adversarial start found after {trial + 1} trial(s).")
            break

    if x_adv is None:
        logger.debug(
            f"HSJA init: no adversarial start found in {n_init_trials} trials; "
            f"returning clean sample (queries={n_queries})."
        )
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
            logger.debug(f"HSJA step {step + 1}/{n_steps}: converged (dist<1e-9), stopping early.")
            break
        # Step size for random perturbation scales with distance and
        # dimensionality so probes stay near the boundary.
        n_dims = float(mask.sum().item()) if mask is not None else float(x_orig.numel())
        delta = dist / (n_dims ** 0.5)

        if predict_batch_fn is not None:
            # Batched path: draw all directions at once. torch.randn fills
            # row-major, so this yields the exact same values (in the same
            # order) as n_grad_samples sequential torch.randn_like(x_orig)
            # draws, keeping the estimate identical to the sequential path.
            U = torch.randn((n_grad_samples,) + tuple(x_orig.shape))
            if mask is not None:
                U = U * mask
            U = U / (U.flatten(1).norm(dim=1).view(-1, *([1] * (U.ndim - 1))) + 1e-12)
            X_probe = _clip(x_adv.unsqueeze(0) + delta * U)
            probe_preds = predict_batch_fn(X_probe)
            n_queries += n_grad_samples
            signs = torch.tensor(
                [1.0 if int(p) != y_orig else -1.0 for p in probe_preds],
                dtype=U.dtype,
            )
            grad_est = (signs.view(-1, *([1] * (U.ndim - 1))) * U).sum(0) / n_grad_samples
        else:
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

        # Per-step progress (DEBUG): boundary distance should trend downward.
        logger.debug(
            f"HSJA step {step + 1}/{n_steps}: boundary_dist={dist:.4f}, "
            f"cumulative_queries={n_queries}."
        )

    return x_adv, n_queries


def hopskipjump_attack_batch(
    predict_batch_fn: Callable[[torch.Tensor], Sequence[int]],
    X_orig: torch.Tensor,
    y_orig: torch.Tensor,
    n_steps: int = 30,
    n_grad_samples: int = 30,
    n_init_trials: int = 100,
    init_noise_scale: float = 3.0,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    feature_indices: Optional[Sequence[int]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorised HopSkipJump — attacks a whole batch of samples in lockstep.

    Runs the identical per-sample HSJA algorithm as ``hopskipjump_attack`` but
    advances every sample together, so each phase issues a single batched
    forward pass over all participating samples (and, for gradient estimation,
    over all of their probes at once) instead of one batch-1 query at a time.
    This collapses the ~15-20k batch-1 forwards of a sequential round into a
    few hundred batched forwards — the regime where CPU CNN inference, which is
    pathologically underutilised at batch size 1, reclaims its throughput.

    The per-sample math is unchanged: init draws the same Gaussian candidates
    and takes the first misclassified one; the boundary walk does the same
    20-iteration binary search, the same random-direction gradient sign
    estimate, and the same geometric-decay step. Per-sample query counts mirror
    the sequential algorithm (init = trials up to first hit, or n_init_trials
    if none; each boundary step = 20 + n_grad_samples + 1). Because directions
    are now drawn for the whole batch at once, the global RNG stream differs
    from running the samples one by one, so individual adversarial points are
    not bit-identical to the sequential path — the attack is statistically, not
    deterministically, equivalent.

    Args:
        predict_batch_fn: callable (X: (k, d) tensor) -> sequence of k class
                          ints. Caller owns model locking inside it.
        X_orig:           clean inputs, shape (N, d).
        y_orig:           true labels, shape (N,) (int/long).
        should_stop:      optional callable polled once per boundary step; if it
                          returns True the walk halts early (cooperative
                          shutdown) and the best adversarial points so far are
                          returned.
        (remaining args as in hopskipjump_attack.)

    Returns:
        (X_adv, n_queries): X_adv shape (N, d); n_queries long tensor shape
        (N,) with per-sample logical query counts. Rows whose initialisation
        failed are returned unchanged (== X_orig) with n_queries == n_init_trials.
    """
    t0 = time.time()
    N, d = X_orig.shape
    device = X_orig.device
    y_orig = y_orig.to(device=device, dtype=torch.long)

    def _clip(x: torch.Tensor) -> torch.Tensor:
        if clip_min is not None or clip_max is not None:
            return torch.clamp(x, min=clip_min, max=clip_max)
        return x

    if feature_indices is not None:
        mask = torch.zeros(d, dtype=X_orig.dtype, device=device)
        mask[list(feature_indices)] = 1.0
        n_dims = float(mask.sum().item())
    else:
        mask = None
        n_dims = float(d)

    def _predict(X: torch.Tensor) -> torch.Tensor:
        # predict_batch_fn returns plain Python ints (no inference-mode tensor
        # escapes the caller's critical section); wrap back into a long tensor.
        return torch.as_tensor(predict_batch_fn(X), dtype=torch.long, device=device)

    n_queries = torch.zeros(N, dtype=torch.long, device=device)
    X_adv = X_orig.clone()

    # ── Phase 1: initialisation, batched across samples AND trials ──────────
    # Draw all (N, n_init_trials) Gaussian candidates up front and evaluate
    # them in a single forward pass, then pick — per sample — the FIRST
    # misclassified trial, exactly as the sequential early-break would.
    T = n_init_trials
    noise = torch.randn(N, T, d, device=device)
    if mask is not None:
        noise = noise * mask
    candidates = _clip(X_orig.unsqueeze(1) + noise * init_noise_scale)        # (N,T,d)
    init_preds = _predict(candidates.reshape(N * T, d)).reshape(N, T)         # (N,T)
    mis = init_preds != y_orig.unsqueeze(1)                                   # (N,T)

    trial_ids = torch.arange(T, device=device).unsqueeze(0).expand(N, T)
    # First misclassified trial index per sample; == T when none misclassified.
    first_idx = torch.where(mis, trial_ids, torch.full_like(trial_ids, T)).min(dim=1).values
    solved = first_idx < T
    first_idx_c = first_idx.clamp(max=T - 1)
    # Logical query count mirrors the sequential early-break.
    n_queries += torch.where(solved, first_idx + 1, torch.full_like(first_idx, T))

    sel = candidates[torch.arange(N, device=device), first_idx_c]            # (N,d)
    X_adv = torch.where(solved.unsqueeze(1), sel, X_adv)

    n_solved = int(solved.sum().item())
    logger.info(
        f"HSJA batch: init found an adversarial start for {n_solved}/{N} samples "
        f"({N - n_solved} robust) in {time.time() - t0:.1f}s."
    )

    # ── Phase 2: boundary walk, vectorised over the live (solved) samples ───
    live = solved.clone()
    for step in range(n_steps):
        if should_stop is not None and should_stop():
            logger.warning(
                f"HSJA batch: stop requested at step {step + 1}/{n_steps}; "
                f"returning best-so-far for {int(live.sum().item())} live samples."
            )
            break

        idx = live.nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            break
        xo, yo, xadv = X_orig[idx], y_orig[idx], X_adv[idx]                   # (m,d),(m,),(m,d)

        # Binary search toward the boundary (20 iters), batched over samples.
        x_lo, x_hi = xo.clone(), xadv.clone()
        for _ in range(20):
            x_mid = _clip((x_lo + x_hi) * 0.5)
            same = (_predict(x_mid) == yo).unsqueeze(1)                       # (m,1)
            x_lo = torch.where(same, x_mid, x_lo)
            x_hi = torch.where(same, x_hi, x_mid)
            xadv = torch.where(same, xadv, x_mid)
        n_queries[idx] += 20
        X_adv[idx] = xadv

        dist = torch.norm(xadv - xo, dim=1)                                  # (m,)
        conv = dist < 1e-9
        if conv.any():
            live[idx[conv]] = False                                         # freeze converged
        cont = ~conv
        if not cont.any():
            logger.info(
                f"HSJA batch: step {step + 1}/{n_steps} — 0 live / {N} "
                f"(all remaining converged), elapsed={time.time() - t0:.1f}s."
            )
            continue

        cidx = idx[cont]
        xo2, yo2, xadv2, dist2 = xo[cont], yo[cont], xadv[cont], dist[cont]
        m = cidx.numel()

        # Gradient-sign estimate, batched over samples AND probes: (m, G, d).
        delta = dist2 / (n_dims ** 0.5)                                      # (m,)
        U = torch.randn(m, n_grad_samples, d, device=device)
        if mask is not None:
            U = U * mask
        U = U / (U.norm(dim=2, keepdim=True) + 1e-12)
        probes = _clip(xadv2.unsqueeze(1) + delta.view(m, 1, 1) * U)         # (m,G,d)
        probe_preds = _predict(probes.reshape(m * n_grad_samples, d)).reshape(m, n_grad_samples)
        n_queries[cidx] += n_grad_samples
        signs = (probe_preds != yo2.unsqueeze(1)).to(U.dtype) * 2.0 - 1.0    # (m,G) in {+1,-1}
        grad = (signs.unsqueeze(2) * U).sum(1) / n_grad_samples             # (m,d)

        # Geometric-decay step toward the clean inputs, kept on the adv side.
        step_size = dist2 / float((step + 1) ** 0.5)                         # (m,)
        x_new = _clip(xadv2 - step_size.unsqueeze(1) * grad)                 # (m,d)
        accept = (_predict(x_new) != yo2).unsqueeze(1)                       # (m,1)
        n_queries[cidx] += 1
        X_adv[cidx] = torch.where(accept, x_new, xadv2)

        logger.info(
            f"HSJA batch: step {step + 1}/{n_steps} — {int(live.sum().item())} live / {N}, "
            f"mean_boundary_dist={float(dist2.mean()):.4f}, elapsed={time.time() - t0:.1f}s."
        )

    return X_adv, n_queries
