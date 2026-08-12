"""Flow-matching timestep sampling and loss for Anima.

Conventions confirmed against diffusers' own Anima pipeline (modular_pipelines/anima/), which is
the authority, and cross-checked against diffusion-pipe:

    x_t    = (1 - t) * x0 + t * noise
    target = noise - x0                     (velocity)
    t in [0, 1]                             (NOT raw 0-1000; the pipeline divides by
                                             scheduler.config.num_train_timesteps)

Timestep distribution matters more than usual here. Anima's normalised latents land at std ~0.69
rather than 1.0 -- that is a property of the data against the VAE's global constants, identical in
both trainers, and must NOT be "corrected" (the constants are what the model was pretrained with).
But it does mean the effective SNR at a given t is lower than a unit-variance schedule assumes,
and `shift` pushes t *higher* still. Worth remembering when tuning: on illustration data, a shift
below the 3.0 inference default is a reasonable thing to try.
"""

import math
from dataclasses import dataclass

import torch


def time_shift(mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
    """Flux-style shift. Ported from diffusion-pipe (models/cosmos_predict2.py:22)."""
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
    """Linear interpolation of mu against sequence length (cosmos_predict2.py:26)."""
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def apply_static_shift(t: torch.Tensor, shift: float) -> torch.Tensor:
    """t' = (t*shift) / (1 + (shift-1)*t).

    shift > 1 biases toward larger t, i.e. noisier samples. This is the same map
    FlowMatchEulerDiscreteScheduler applies at inference with shift=3.0.
    """
    return (t * shift) / (1 + (shift - 1) * t)


def apply_flux_shift(t: torch.Tensor, latent_h: int, latent_w: int) -> torch.Tensor:
    """Resolution-dependent shift: larger images get pushed to higher noise.

    mu is interpolated against the *token* count (latent_h//2 * latent_w//2, i.e. after the
    2x2 patch embed), matching diffusion-pipe (models/anima.py:1002).
    """
    mu = get_lin_function(y1=0.5, y2=1.15)((latent_h // 2) * (latent_w // 2))
    return time_shift(mu, 1.0, t)


@dataclass
class FlowConfig:
    timestep_sample_method: str = "logit_normal"  # "logit_normal" | "uniform"
    sigmoid_scale: float = 1.0
    shift: float | None = None       # static shift; mutually exclusive with flux_shift
    flux_shift: bool = False         # resolution-dependent shift
    use_ot: bool = False             # cosine optimal-transport noise pairing
    # How a curriculum phase's `t_range` restricts the draw. Inert without a curriculum.
    phase_mapping: str = "rescale"   # "rescale" | "truncate" -- see sample_timesteps

    # High-frequency token loss (see `hf_loss`). 0.0 = off, and off is bit-identical to the
    # feature not existing -- the term is gated in Python, allocates nothing, and draws no RNG.
    hf_scale: float = 0.0            # lambda
    hf_exponent: float = 1.0         # gamma; >1 concentrates on the highest-detail tokens

    def __post_init__(self):
        if self.shift is not None and self.flux_shift:
            raise ValueError("set either `shift` or `flux_shift`, not both")
        # shift <= 0 is not "shift off" -- the map (t*shift)/(1+(shift-1)*t) sends every t to 0 at
        # shift=0 and goes negative below it. Either way the batch trains at t=0, where the input is
        # the clean latent and the velocity target is unpredictable noise: loss parks at 1.0 and the
        # run learns nothing while looking perfectly healthy. Omit the key for no shift.
        if self.shift is not None and self.shift <= 0.0:
            raise ValueError(
                f"flow.shift must be > 0, got {self.shift}. shift={self.shift} does not mean "
                f"'no shift' -- it collapses every timestep to 0, which pins the loss at 1.0 and "
                f"trains on nothing. Remove the `shift` key entirely for no shift, or use 1.0 for "
                f"the identity map."
            )
        if self.timestep_sample_method not in ("logit_normal", "uniform"):
            raise ValueError(f"unknown timestep_sample_method: {self.timestep_sample_method}")
        # `sigmoid_scale` is the width of the logistic squash applied to a normal draw. There is no
        # sigmoid in the uniform path, so the value would be read and discarded -- rejected rather
        # than ignored, because a config that says "widen the timestep distribution" and silently
        # does not is indistinguishable from one that works.
        if self.timestep_sample_method == "uniform" and self.sigmoid_scale != 1.0:
            raise ValueError(
                f"flow.sigmoid_scale={self.sigmoid_scale} only applies to "
                f"timestep_sample_method='logit_normal'; the uniform sampler has no sigmoid to "
                f"scale. Set timestep_sample_method='logit_normal' or drop sigmoid_scale."
            )
        if self.sigmoid_scale <= 0.0:
            raise ValueError(f"flow.sigmoid_scale must be > 0, got {self.sigmoid_scale}")
        if self.phase_mapping not in ("rescale", "truncate"):
            raise ValueError(
                f"flow.phase_mapping must be 'rescale' or 'truncate', got {self.phase_mapping!r}")
        # OT needs an assignment solver. Without scipy the pairing silently degrades to the
        # identity permutation -- i.e. `use_ot = true` trains exactly as if it were false, for the
        # whole run, with nothing to show for it. Checked once here rather than swallowed per step.
        if self.use_ot:
            import importlib.util

            if importlib.util.find_spec("scipy") is None:
                raise ValueError(
                    "flow.use_ot = true needs scipy for linear_sum_assignment. Install it with "
                    "`uv sync --extra ot`, or set use_ot = false. (Without it the noise pairing "
                    "would fall back to the identity permutation and do nothing.)"
                )
        if self.hf_scale < 0.0:
            raise ValueError(f"flow.hf_scale must be >= 0, got {self.hf_scale}")
        if self.hf_exponent <= 0.0:
            raise ValueError(f"flow.hf_exponent must be > 0, got {self.hf_exponent}")


def _draw(cfg: FlowConfig, n: int, latent_h: int, latent_w: int, device) -> torch.Tensor:
    """One unrestricted draw of t in [0,1], distribution and shift applied."""
    if cfg.timestep_sample_method == "logit_normal":
        t = torch.distributions.normal.Normal(0, 1).sample((n,)).to(device)
        t = torch.sigmoid(t * cfg.sigmoid_scale)
    else:
        t = torch.distributions.uniform.Uniform(0, 1).sample((n,)).to(device)

    # Shift is applied *before* any range mapping, matching the reference implementation. The
    # other order is not equivalent -- shift is nonlinear, so shifting a rescaled t bends the
    # distribution differently inside the phase than rescaling a shifted one.
    if cfg.shift is not None:
        t = apply_static_shift(t, cfg.shift)
    elif cfg.flux_shift:
        t = apply_flux_shift(t, latent_h, latent_w)
    return t


def sample_timesteps(
    cfg: FlowConfig,
    batch_size: int,
    latent_h: int,
    latent_w: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
    quantile: float | None = None,
    t_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Draw t in [0,1], shaped (B,).

    `quantile` replaces sampling with a fixed quantile of the distribution -- used for
    reproducible validation loss, so eval noise doesn't jitter between runs.

    `t_range` restricts the draw to a sub-range, which is how a curriculum phase asks for
    "low-noise only". Two inequivalent ways to honour it, selected by `cfg.phase_mapping`:

    * **rescale** (default, and what TrainTrain does): affinely map the whole distribution onto
      the range. Shape is preserved exactly -- every quantile is `span x` the full one -- so a
      phase is the same curve, squeezed. Measured for logit_normal(1.3) on [0, 0.6]: median
      0.499 -> 0.299.
    * **truncate**: keep the pretraining density and discard draws outside the range. The
      surviving mass is skewed toward the top of the range: same setup gives median 0.344.

    Neither is obviously right, which is why both exist. Note that truncation and rejection
    sampling are the *same* distribution, not two options -- rejection is just how truncation is
    implemented here.
    """
    lo, hi = (0.0, 1.0) if t_range is None else t_range

    if quantile is not None:
        # A deterministic probe, not a sample. Placed by linear interpolation in both mappings:
        # "the q-th quantile of a truncated distribution" would need the CDF of whatever
        # distribution is configured, and an eval probe only has to be *consistent*, not
        # distribution-faithful. Documented rather than silently differing from training.
        base = torch.full((batch_size,), quantile, device=device)
        if cfg.timestep_sample_method == "logit_normal":
            base = torch.sigmoid(torch.distributions.normal.Normal(0, 1).icdf(base)
                                 * cfg.sigmoid_scale)
        if cfg.shift is not None:
            base = apply_static_shift(base, cfg.shift)
        elif cfg.flux_shift:
            base = apply_flux_shift(base, latent_h, latent_w)
        return (base * (hi - lo) + lo).to(dtype)

    if t_range is None or cfg.phase_mapping == "rescale":
        return (_draw(cfg, batch_size, latent_h, latent_w, device) * (hi - lo) + lo).to(dtype)

    # Truncate by rejection. Draws are scalars, so rejected ones cost nothing measurable; the
    # bound exists so a range with negligible probability fails loudly instead of hanging.
    out = torch.empty(0, device=device)
    for _ in range(_TRUNCATE_MAX_ROUNDS):
        c = _draw(cfg, batch_size * 4, latent_h, latent_w, device)
        out = torch.cat([out, c[(c >= lo) & (c <= hi)]])
        if out.numel() >= batch_size:
            return out[:batch_size].to(dtype)
    raise RuntimeError(
        f"flow.phase_mapping='truncate' could not fill a batch from t_range [{lo}, {hi}]: that "
        f"range holds almost no probability under {cfg.timestep_sample_method} "
        f"(sigmoid_scale={cfg.sigmoid_scale}, shift={cfg.shift}). Widen the phase's t_range, "
        f"change the distribution, or use phase_mapping='rescale', which always fills."
    )


_TRUNCATE_MAX_ROUNDS = 64


def cosine_optimal_transport(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Reorder `y` so each row pairs with its most cosine-similar row in `x`.

    Straighter flow trajectories -> faster convergence. Falls back to the identity permutation if
    no assignment solver is available, rather than silently changing the objective.
    """
    n = x.shape[0]
    if n < 2:
        return torch.arange(n, device=x.device)

    xn = torch.nn.functional.normalize(x.flatten(1).float(), dim=1)
    yn = torch.nn.functional.normalize(y.flatten(1).float(), dim=1)
    cost = -(xn @ yn.T)  # maximise similarity == minimise negative similarity

    try:
        from scipy.optimize import linear_sum_assignment

        _, col = linear_sum_assignment(cost.cpu().numpy())
        return torch.as_tensor(col, device=x.device)
    except ImportError:
        return torch.arange(n, device=x.device)


def prepare_flow_batch(
    latents: torch.Tensor,
    cfg: FlowConfig,
    generator: torch.Generator | None = None,
    quantile: float | None = None,
    stats: dict | None = None,
    t_range: tuple[float, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """latents (B,C,T,H,W) -> (noisy_latents, timesteps, target).

    target is the flow velocity `noise - x0`.

    `stats`, if given, is filled with diagnostics -- currently `ot_moved`, the fraction of rows the
    optimal-transport pairing actually reordered. That number is the only way to tell OT apart from
    a no-op: it is 0.0 at batch size 1 by construction, and would also be 0.0 if the solver were
    silently falling back to the identity permutation.
    """
    latents = latents.float()
    b, _, _, h, w = latents.shape

    t = sample_timesteps(
        cfg, b, h, w, device=latents.device, dtype=torch.float32,
        generator=generator, quantile=quantile, t_range=t_range,
    )

    noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype, generator=generator)

    if cfg.use_ot and b > 1:
        with torch.no_grad():
            perm = cosine_optimal_transport(latents, noise)
            if stats is not None:
                identity = torch.arange(b, device=perm.device)
                stats["ot_moved"] = (perm != identity).float().mean().item()
            noise = noise[perm]
    elif stats is not None and cfg.use_ot:
        stats["ot_moved"] = 0.0        # batch of 1: nothing to permute

    t_expanded = t.view(-1, 1, 1, 1, 1)
    noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
    target = noise - latents

    return noisy_latents, t, target


def flow_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Plain MSE against the velocity target.

    No per-timestep loss weighting: for rectified flow the uniform weighting is already the
    correct one, and the timestep *distribution* (logit-normal + shift) is what biases training
    toward particular noise levels. Adding weighting on top would double-count.
    """
    loss = (model_pred.float() - target.float()) ** 2

    if mask is not None:
        while mask.ndim < loss.ndim:
            mask = mask.unsqueeze(1)
        loss = loss * mask
        return loss.sum() / mask.expand_as(loss).sum().clamp(min=1)

    return loss.mean()


# --------------------------------------------------------------------------------------------
# High-frequency token loss
#
# An auxiliary term that concentrates effort on the tokens carrying fine detail. Plain velocity
# MSE spreads its gradient uniformly over tokens, so flat regions -- which are most of an
# illustration -- dominate the signal by area, and the edges and texture that decide whether an
# image looks sharp contribute in proportion to how little of the frame they occupy. This term
# reweights the *same* error by each token's local Laplacian energy, measured on the clean target.
#
# Three properties are load-bearing and are not to be "improved":
#   * weights come from `clean` only, never from the prediction (they would otherwise be a term
#     the model can minimise by flattening its own output);
#   * the eps appears in numerator *and* denominator, so a flat latent degenerates to plain
#     per-token MSE instead of 0/0;
#   * nothing here draws from the RNG, so hf_scale=0 vs hf_scale>0 leaves the timestep and
#     caption-dropout streams bit-identical and same-seed runs stay comparable.
#
# Spec: high-frequency-token-loss.md.
# --------------------------------------------------------------------------------------------


def _laplacian_energy(x: torch.Tensor) -> torch.Tensor:
    """Squared 4-neighbour Laplacian, (B,C,H,W) -> (B,C,H,W).

    Replication padding is not a detail: with zero padding a *constant* input produces a spurious
    response on the boundary ring, which destroys the "flat latent degenerates to plain MSE"
    property. With replicate, a constant input gives exactly 0.
    """
    padded = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="replicate")
    lap = (
        4.0 * x
        - padded[:, :, :-2, 1:-1] - padded[:, :, 2:, 1:-1]
        - padded[:, :, 1:-1, :-2] - padded[:, :, 1:-1, 2:]
    )
    return lap * lap


def _tokenize(x: torch.Tensor, patch: int) -> torch.Tensor:
    """(B,C,H,W) -> (B,N,C*patch*patch). Feature order within a token is irrelevant; only means
    are ever taken over the last axis."""
    b, c, h, w = x.shape
    cols = torch.nn.functional.unfold(x, kernel_size=patch, stride=patch)
    return cols.transpose(1, 2).reshape(b, (h // patch) * (w // patch), c * patch * patch)


def hf_token_weights(
    clean: torch.Tensor, patch: int, exponent: float, eps: float = 1e-6
) -> torch.Tensor:
    """Per-token detail weights from the clean target, (B,C,H,W) -> (B,N), per-sample mean 1.

    The mean-1 rescale keeps the term's scale comparable to a plain x0-MSE for any `exponent`, so
    `hf_scale` means the same thing as `hf_exponent` is tuned.
    """
    detail = _tokenize(_laplacian_energy(clean), patch).mean(dim=-1)
    raw = ((detail + eps) / (detail.mean(dim=-1, keepdim=True) + eps)) ** exponent
    return raw / raw.mean(dim=-1, keepdim=True)


def hf_loss(
    model_pred: torch.Tensor,
    noisy: torch.Tensor,
    clean: torch.Tensor,
    timesteps: torch.Tensor,
    patch: int,
    exponent: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Detail-weighted MSE on the *predicted clean estimate*, for 5D (B,C,T,H,W) latents with T=1.

    Tweedie for this parameterisation: with `x_t = (1-t)*x0 + t*noise` and `v = noise - x0`,
    `x_t - t*v` collapses exactly to `x0`, so `x0_hat = noisy - t*pred`.

    Deliberately on the x0 side rather than the velocity side: `||x0_hat - x0||^2 = t^2 * ||v -
    v_target||^2` elementwise, and content weighting is only meaningful on the clean estimate --
    at low t the velocity target is noise-dominated and "which token has detail" says nothing
    about it. The t^2 factor makes the term weak at small t; that is accepted, not compensated,
    and no timestep gate is applied.
    """
    b, c, t_dim, h, w = clean.shape
    if t_dim != 1:
        raise ValueError(f"hf loss expects a single latent frame, got T={t_dim}")
    if h % patch or w % patch:
        raise ValueError(f"latent {h}x{w} is not divisible by the patch size {patch}")

    clean = clean.reshape(b, c, h, w).float()
    x0_pred = (noisy.float() - timesteps.float().view(-1, 1, 1, 1, 1) * model_pred.float())
    x0_pred = x0_pred.reshape(b, c, h, w)

    # `clean` is data and carries no grad, so the weights are already constant; detach anyway so a
    # future change that puts the target in the graph cannot silently start training the weights.
    with torch.no_grad():
        w_tok = hf_token_weights(clean, patch, exponent)

    per_token = _tokenize(x0_pred - clean, patch).square().mean(dim=-1)

    if mask is None:
        return (w_tok * per_token).mean(dim=-1).mean()

    # Pool the mask to token resolution the same way the loss is tokenized, so a partially
    # feathered token contributes in proportion to how much of it is supervised. Normalised by
    # the mask's own weight, not by token count: dividing by the full token count would shrink
    # the loss simply because fewer tokens are supervised, making the term's magnitude depend
    # on mask size rather than on error.
    m = mask.reshape(b, 1, h, w).expand(b, c, h, w).float()
    m_tok = _tokenize(m, patch).mean(dim=-1)
    num = (w_tok * per_token * m_tok).sum(dim=-1)
    den = (w_tok * m_tok).sum(dim=-1).clamp_min(1e-8)
    return (num / den).mean()
