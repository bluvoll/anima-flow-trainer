"""Gate for the high-frequency token loss (spec: high-frequency-token-loss.md §8).

    .venv/bin/python anima/parity/test_hf_loss.py

Runs on CPU in a second -- no model, no GPU. The term is cheap arithmetic, so the risk is not
performance but silent wrongness: a weight computed from the prediction instead of the target, a
missing eps that NaNs on a blank latent, or an RNG draw that desynchronises same-seed runs. Every
one of those is a negative control below rather than a comment.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from anima.training.flow import (  # noqa: E402
    FlowConfig,
    _laplacian_energy,
    hf_loss,
    hf_token_weights,
    prepare_flow_batch,
    sample_timesteps,
)

P = 2  # Anima's spatial patch size (transformer config patch_size = [1, 2, 2])


# ------------------------------------------------ independent from-first-principles reference


def ref_hf(x0_pred, clean, patch, exponent, eps=1e-6):
    """§2 of the spec, rewritten from the prose with public ops only. Deliberately not sharing a
    single helper with the implementation -- a shared bug would cancel out."""
    b, c, h, w = clean.shape
    n = (h // patch) * (w // patch)

    pad = F.pad(clean, (1, 1, 1, 1), mode="replicate")
    lap = (4.0 * clean - pad[:, :, :-2, 1:-1] - pad[:, :, 2:, 1:-1]
           - pad[:, :, 1:-1, :-2] - pad[:, :, 1:-1, 2:])
    d = lap * lap

    # Reshape-based tokenization rather than unfold, so the two paths agree only if the token
    # geometry is right and not merely because they call the same kernel.
    def tok(x):
        return (x.reshape(b, c, h // patch, patch, w // patch, patch)
                 .permute(0, 2, 4, 1, 3, 5).reshape(b, n, c * patch * patch))

    detail = tok(d).mean(dim=-1)
    raw = ((detail + eps) / (detail.mean(dim=-1, keepdim=True) + eps)) ** exponent
    weights = raw / raw.mean(dim=-1, keepdim=True)

    per_token = tok(x0_pred - clean).square().mean(dim=-1)
    return (weights * per_token).mean(dim=-1).mean()


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    return bool(ok)


def make_batch(b=3, c=16, h=12, w=16, seed=0):
    """(clean, noisy, pred, t) in the trainer's 5D (B,C,T,H,W) layout."""
    # fp32 throughout: `hf_loss` casts to float internally, matching `flow_loss`'s convention, so
    # a float64 reference could not be compared against it exactly.
    g = torch.Generator().manual_seed(seed)
    clean = torch.randn(b, c, 1, h, w, generator=g)
    noise = torch.randn(b, c, 1, h, w, generator=g)
    t = torch.rand(b, generator=g) * 0.98 + 0.01
    tv = t.view(-1, 1, 1, 1, 1)
    noisy = (1 - tv) * clean + tv * noise
    pred = torch.randn(b, c, 1, h, w, generator=g)
    return clean, noisy, pred, t


def main() -> int:
    r = []

    # 1. Laplacian unit tests. A constant input is exactly 0 when the constant is representable
    #    (4a - a - a - a - a rounds otherwise, at ~1 ulp squared) -- with ZERO padding it is not,
    #    by a factor of 1e13, and that is what would break the flat-latent degeneration below.
    for v in (4.0, 0.25):
        e = _laplacian_energy(torch.full((1, 1, 8, 8), v)).abs().max().item()
        r.append(check(f"laplacian(constant {v}) == 0 exactly", e == 0.0))
    e = _laplacian_energy(torch.full((1, 1, 8, 8), 3.7)).abs().max().item()
    r.append(check("laplacian(constant 3.7) == 0 to rounding", e < 1e-11, f"{e:.2e}"))
    const = torch.full((1, 1, 8, 8), 3.7)
    zero_pad = F.conv2d(F.pad(const, (1, 1, 1, 1)),
                        torch.tensor([[[[0.0, -1, 0], [-1, 4, -1], [0, -1, 0]]]]))
    r.append(check("zero padding would give a boundary ring (negative control)",
                   zero_pad.abs().max().item() > 1.0, f"max {zero_pad.abs().max().item():.2f}"))

    s = 0.5
    checker = torch.tensor([[(-1.0) ** (i + j) for j in range(8)] for i in range(8)]) * s
    interior = _laplacian_energy(checker.view(1, 1, 8, 8))[0, 0, 1:-1, 1:-1]
    r.append(check("laplacian(checkerboard) interior == 64*s^2",
                   torch.allclose(interior, torch.full_like(interior, 64 * s * s)),
                   f"got {interior[0, 0].item():.4f}, want {64 * s * s:.4f}"))

    # 2. Weights are per-sample mean exactly 1, for any exponent.
    clean, noisy, pred, t = make_batch()
    flat = clean.reshape(clean.shape[0], clean.shape[1], *clean.shape[3:])
    for g in (0.25, 1.0, 3.0):
        w = hf_token_weights(flat, P, g)
        r.append(check(f"weights mean 1 (gamma={g})",
                       torch.allclose(w.mean(dim=-1), torch.ones(w.shape[0])),
                       f"max|mean-1| {(w.mean(dim=-1) - 1).abs().max().item():.2e}"))
    # ...and gamma is doing something: higher gamma concentrates mass on fewer tokens.
    spread = [hf_token_weights(flat, P, g).max(dim=-1).values.mean().item() for g in (0.25, 1.0, 3.0)]
    r.append(check("gamma concentrates weight", all(a < b for a, b in zip(spread, spread[1:])),
                   "max weight " + " < ".join(f"{v:.2f}" for v in spread)))

    # 3. Matches the from-first-principles reference across gamma.
    x0_hat = (noisy - t.view(-1, 1, 1, 1, 1) * pred).reshape(flat.shape)
    for g in (0.5, 1.0, 2.0):
        got = hf_loss(pred, noisy, clean, t, P, g)
        want = ref_hf(x0_hat, flat, P, g)
        # Not bit-identical by construction: the reference tokenizes by reshape/permute and the
        # implementation by unfold, so the reductions sum in a different order.
        r.append(check(f"matches spec reference (gamma={g})",
                       torch.allclose(got, want, rtol=1e-6, atol=0),
                       f"rel {((got - want).abs() / want).item():.2e}"))

    # 4. Negative controls -- each must NOT match, or the corresponding property is untested.
    n = flat.shape[0]
    uniform = ((x0_hat - flat).reshape(n, -1) ** 2).mean(dim=-1).mean()
    r.append(check("uniform weights differ (control)",
                   not torch.allclose(hf_loss(pred, noisy, clean, t, P, 1.0), uniform),
                   f"hf {hf_loss(pred, noisy, clean, t, P, 1.0).item():.4f} vs plain {uniform.item():.4f}"))
    from_pred = ref_hf(x0_hat, flat, P, 1.0)
    w_wrong = hf_token_weights(x0_hat, P, 1.0)          # weights from the PREDICTION
    per_tok = ((x0_hat - flat).reshape(n, 16, 6, 2, 8, 2)
               .permute(0, 2, 4, 1, 3, 5).reshape(n, 48, 64)).square().mean(-1)
    r.append(check("weights-from-prediction differ (control)",
                   not torch.allclose((w_wrong * per_tok).mean(-1).mean(), from_pred)))
    vel_err = ((pred - (noisy - flat.unsqueeze(2)) / t.view(-1, 1, 1, 1, 1)) ** 2).mean()
    r.append(check("velocity-domain term differs (control)",
                   not torch.allclose(from_pred, vel_err)))

    # 5. Flat latent: weights degenerate to 1, the term becomes plain per-token x0-MSE, finite.
    #    Without the eps in numerator and denominator this is 0/0 -> NaN.
    zc = torch.zeros(2, 16, 1, 8, 8)
    zp = torch.randn(2, 16, 1, 8, 8, generator=torch.Generator().manual_seed(1))
    zt = torch.full((2,), 0.3)
    zn = zt.view(-1, 1, 1, 1, 1) * torch.randn(2, 16, 1, 8, 8, generator=torch.Generator().manual_seed(2))
    got = hf_loss(zp, zn, zc, zt, P, 1.0)
    want = ((zn - zt.view(-1, 1, 1, 1, 1) * zp) ** 2).mean()
    r.append(check("flat latent == plain x0-MSE, finite",
                   torch.isfinite(got) and torch.allclose(got, want),
                   f"{got.item():.6f} vs {want.item():.6f}"))
    w_flat = hf_token_weights(zc.reshape(2, 16, 8, 8), P, 2.0)
    r.append(check("flat latent weights all exactly 1",
                   torch.equal(w_flat, torch.ones_like(w_flat))))
    no_eps = torch.zeros(1, 4).sum() / torch.zeros(1, 1).sum()
    r.append(check("missing-eps formula NaNs (control)", bool(torch.isnan(no_eps).all())))

    # 6. Single-token grid: N == 1 -> w == 1 exactly.
    one = torch.randn(1, 16, 2, 2, generator=torch.Generator().manual_seed(3))
    w1 = hf_token_weights(one, P, 2.0)
    r.append(check("N=1 grid gives w=1", w1.shape == (1, 1) and torch.equal(w1, torch.ones(1, 1))))

    # 7. Tweedie identity: a perfect velocity prediction gives exactly zero, at any t. This is
    #    what pins the x0_hat = noisy - t*pred algebra to *this* parameterisation.
    perfect_t = torch.tensor([0.01, 0.5, 0.99])
    pc = torch.randn(3, 16, 1, 8, 8, generator=torch.Generator().manual_seed(4))
    pn = torch.randn(3, 16, 1, 8, 8, generator=torch.Generator().manual_seed(5))
    tv = perfect_t.view(-1, 1, 1, 1, 1)
    resid = hf_loss(pn - pc, (1 - tv) * pc + tv * pn, pc, perfect_t, P, 1.5).abs().item()
    r.append(check("exact prediction -> hf == 0 at all t", resid < 1e-12, f"{resid:.2e}"))

    # 8. RNG neutrality: the term must draw nothing, or hf_scale>0 shifts the timestep and
    #    caption-dropout streams and same-seed runs stop being comparable.
    torch.manual_seed(99)
    before = torch.get_rng_state()
    hf_loss(pred, noisy, clean, t, P, 1.0)
    r.append(check("draws no RNG", torch.equal(before, torch.get_rng_state())))

    # 9. The weights carry no gradient; the term is differentiable through the prediction only.
    gp = pred.clone().requires_grad_(True)
    hf_loss(gp, noisy, clean, t, P, 1.0).backward()
    r.append(check("gradient flows to prediction", gp.grad is not None and gp.grad.abs().sum() > 0))

    # 10. Config validation, and the off-gate.
    for bad in (dict(hf_scale=-0.1), dict(hf_exponent=0.0), dict(hf_exponent=-1.0)):
        try:
            FlowConfig(**bad)
            r.append(check(f"reject {bad}", False, "accepted"))
        except ValueError:
            r.append(check(f"reject {bad}", True))
    r.append(check("hf_scale defaults to 0 (off)", FlowConfig().hf_scale == 0.0))

    # 11. The other two FlowConfig knobs that used to fail silently.
    #     sigmoid_scale is a no-op under `uniform` -- there is no sigmoid to scale.
    try:
        FlowConfig(timestep_sample_method="uniform", sigmoid_scale=1.3)
        r.append(check("reject sigmoid_scale under uniform", False, "accepted -> silently ignored"))
    except ValueError:
        r.append(check("reject sigmoid_scale under uniform", True))
    r.append(check("uniform + default sigmoid_scale is fine",
                   FlowConfig(timestep_sample_method="uniform").sigmoid_scale == 1.0))

    scales = {}
    for s in (0.5, 1.0, 2.0):
        torch.manual_seed(0)
        t = sample_timesteps(FlowConfig(sigmoid_scale=s), 20000, 96, 96, torch.device("cpu"))
        scales[s] = (t.quantile(0.1).item(), t.quantile(0.9).item())
    widths = [hi - lo for lo, hi in scales.values()]
    r.append(check("sigmoid_scale widens the logit_normal spread",
                   widths[0] < widths[1] < widths[2],
                   " < ".join(f"{w:.3f}" for w in widths)))

    #     use_ot silently degraded to the identity permutation when scipy was absent.
    import unittest.mock as mock

    with mock.patch("importlib.util.find_spec", return_value=None):
        try:
            FlowConfig(use_ot=True)
            r.append(check("reject use_ot without scipy", False, "accepted -> silent no-op"))
        except ValueError:
            r.append(check("reject use_ot without scipy", True))

    #     ...and it reports how much it actually reordered, which is the only way to tell it apart
    #     from a no-op. Batch 1 must be exactly 0.0; a real batch must move rows.
    moved = {}
    for bs in (1, 8):
        st = {}
        torch.manual_seed(0)
        prepare_flow_batch(torch.randn(bs, 16, 1, 16, 16), FlowConfig(use_ot=True), stats=st)
        moved[bs] = st.get("ot_moved")
    r.append(check("ot_moved is 0.0 at batch 1, >0 at batch 8",
                   moved[1] == 0.0 and moved[8] > 0.0, str(moved)))
    st = {}
    prepare_flow_batch(torch.randn(8, 16, 1, 16, 16), FlowConfig(use_ot=False), stats=st)
    r.append(check("no ot_moved key when use_ot is off", "ot_moved" not in st))

    # 11. Shape guards: a T>1 latent or an odd latent side must fail loudly, not silently
    #     mis-tokenize into the wrong spatial extent.
    for name, args in (
        ("T>1 rejected", (pred.repeat(1, 1, 2, 1, 1), noisy.repeat(1, 1, 2, 1, 1),
                          clean.repeat(1, 1, 2, 1, 1), t)),
        ("odd latent side rejected", (pred[..., :11], noisy[..., :11], clean[..., :11], t)),
    ):
        try:
            hf_loss(*args, P, 1.0)
            r.append(check(name, False, "accepted"))
        except ValueError:
            r.append(check(name, True))

    n_ok = sum(r)
    print(f"\n{n_ok}/{len(r)} passed")
    return 0 if n_ok == len(r) else 1


if __name__ == "__main__":
    sys.exit(main())
