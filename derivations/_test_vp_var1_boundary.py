"""
Phase 4 gate (f) — VP boundary regression test at `var_k = 1`.

Both Mathematica (`posterior_rederivation.nb`) and SymPy
(`01_posterior_rederivation.ipynb`) confirm that under the VP schedule,
the analytic posterior-mean limit as `t → 1⁻` is

    μ_post(t→1) = (√2 · var_k · x_s − μ_k) / (var_k − 1)

which diverges when `var_k = 1` exactly. This is a real mathematical
singularity, not a numerical artifact. The shipped JIT has no guard for
it (caveat C5 in the predecessor `03_go_no_go.md`, now Phase 4 entry
gate (f) in `03_rollout_plan.md`).

This regression test pins the production behaviour around the boundary
so that future changes (clamps, schedule changes, kernel rewrites) cannot
silently change what happens at `var_k = 1`.

Sweep:
    var_k in {0.5, 0.99, 1.00, 1.01, 1.5, 5.0}
    t     in {0.99, 0.999, 0.9999, 0.99999}     (s_src is fixed, t < s_src)
    s_src = 0.5                                  (clean source, noisy target)

Hard assertion:
    For `var_k != 1`: `torch.isfinite(out).all()` must hold.

Soft characterization (printed but not asserted):
    For `var_k = 1`: record `max(|out|)`, fraction non-finite, fraction
    near-zero. This is the regression baseline. If a future run prints
    different numbers, that's a behaviour change that needs human review.

[VERIFY FIRST] from `03_rollout_plan.md` §4.1 (gate f): re-confirm the
analytic divergence formula in Mathematica + SymPy at the rebuild SHA
before treating this test as a complete validation. The plan flags this
as a precondition; this script does not re-derive the formula.

Forward-compatibility: handles both pre-C1 (no `zeta_max` in production
signature) and post-C1 (passes `float('inf')` to disable the clamp, which
is what we want for a *boundary* characterization — the clamp would mask
the true behaviour we are trying to pin).
"""
import hashlib
import importlib.util
import math
import os
import re
import sys
import tempfile

import torch


def _extract_function_source(text, fn_name):
    marker = f'@torch.jit.script\ndef {fn_name}('
    start = text.index(marker)
    m = re.search(r'\n\n\n(?=@torch\.jit\.script|class |def )', text[start:])
    if m is None:
        raise RuntimeError(f'Could not find end of {fn_name}')
    return text[start:start + m.start() + 1] + '\n'


def _load_production_general():
    here = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.normpath(os.path.join(
        here, '..', 'repos', 'piFlow', 'lakonlab', 'models', 'diffusions', 'gmflow.py'))
    if not os.path.exists(src_path):
        raise FileNotFoundError(f'Production source not found: {src_path}')

    with open(src_path) as f:
        text = f.read()

    fn_src = _extract_function_source(text, 'gmflow_posterior_mean_jit_general')
    sha = hashlib.sha256(fn_src.encode()).hexdigest()
    sig = re.search(
        r'def gmflow_posterior_mean_jit_general\(([\s\S]*?)\):', fn_src).group(1)
    has_zeta_max = 'zeta_max' in sig

    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='_piflow_gate_f_extracted.py', delete=False)
    try:
        tmp.write('import torch\n\n')
        tmp.write(fn_src)
        tmp.flush()
        tmp_path = tmp.name
    finally:
        tmp.close()

    spec = importlib.util.spec_from_file_location('piflow_gate_f_extracted', tmp_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['piflow_gate_f_extracted'] = mod
    spec.loader.exec_module(mod)

    return mod.gmflow_posterior_mean_jit_general, has_zeta_max, src_path, sha, fn_src


def _vp_alpha_sigma(t, dtype, eps=1e-30):
    t_t = torch.tensor([[[[t]]]], dtype=dtype)
    sigma = torch.sin(math.pi * t_t / 2).clamp(min=eps)
    alpha = torch.cos(math.pi * t_t / 2)
    return alpha, sigma


def _make_batch(B, K, C, H, W, dtype, var_k, seed):
    g = torch.Generator().manual_seed(seed)
    return dict(
        gm_means=torch.randn(B, K, C, H, W, generator=g, dtype=dtype),
        gm_logweights=torch.randn(B, K, 1, H, W, generator=g, dtype=dtype),
        gm_vars=torch.full((B, 1, 1, 1, 1), float(var_k), dtype=dtype),
        x_t_src=torch.randn(B, C, H, W, generator=g, dtype=dtype),
        x_t=torch.randn(B, C, H, W, generator=g, dtype=dtype),
    )


def _call(jit_fn, has_zeta_max, alpha_s, sigma_s, alpha_t, sigma_t, batch, eps):
    if has_zeta_max:
        return jit_fn(
            alpha_s, sigma_s, alpha_t, sigma_t,
            batch['x_t_src'], batch['x_t'],
            batch['gm_means'], batch['gm_vars'], batch['gm_logweights'],
            eps, float('inf'))
    return jit_fn(
        alpha_s, sigma_s, alpha_t, sigma_t,
        batch['x_t_src'], batch['x_t'],
        batch['gm_means'], batch['gm_vars'], batch['gm_logweights'], eps)


def main():
    print('=' * 78)
    print('GATE (f) — VP boundary regression at var_k = 1')
    print('=' * 78)
    print()

    jit_fn, has_zeta_max, src_path, sha, fn_src = _load_production_general()
    print(f'[setup] source: {src_path}')
    print(f'[setup] gmflow_posterior_mean_jit_general '
          f'({len(fn_src)} bytes) SHA-256: {sha}')
    print(f'[setup] zeta_max in signature: {has_zeta_max}'
          f'{" (passing inf to characterize the unclamped boundary)" if has_zeta_max else ""}')
    print()
    print('Reminder: [VERIFY FIRST] the analytic divergence formula')
    print('  mu_post(t->1) = (sqrt(2)*var_k*x_s - mu_k) / (var_k - 1)')
    print('  in Mathematica + SymPy at the current rebuild SHA before treating')
    print('  this gate as fully validated. See 03_rollout_plan.md §4.1.')
    print()

    dtype = torch.float32
    eps = 1e-6
    B, K, C, H, W = 2, 4, 8, 16, 16
    s_src = 0.5

    var_values = [0.5, 0.99, 1.00, 1.01, 1.5, 5.0]
    t_values = [0.99, 0.999, 0.9999, 0.99999]

    alpha_s, sigma_s = _vp_alpha_sigma(s_src, dtype)

    header = (f'{"var_k":<8s} {"t":<10s} {"sigma_t":<12s} '
              f'{"|out| max":<14s} {"finite_frac":<12s} '
              f'{"|out| mean":<14s} {"verdict":<10s}')
    print(header)
    print('-' * 88)

    hard_pass = True
    var1_records = []

    for var_k in var_values:
        batch = _make_batch(B, K, C, H, W, dtype, var_k, seed=int(var_k * 31) + 7)
        for t in t_values:
            alpha_t, sigma_t = _vp_alpha_sigma(t, dtype)
            out = _call(jit_fn, has_zeta_max,
                        alpha_s, sigma_s, alpha_t, sigma_t, batch, eps)

            finite_mask = torch.isfinite(out)
            finite_frac = finite_mask.float().mean().item()
            if finite_frac == 0.0:
                max_out = float('inf')
                mean_out = float('nan')
            else:
                max_out = out[finite_mask].abs().max().item()
                mean_out = out[finite_mask].abs().mean().item()

            is_var1 = abs(var_k - 1.0) < 1e-12
            if is_var1:
                var1_records.append(dict(
                    t=t, finite_frac=finite_frac,
                    max_out=max_out, mean_out=mean_out))
                verdict = 'BOUNDARY'
            else:
                ok = (finite_frac == 1.0)
                hard_pass &= ok
                verdict = 'PASS' if ok else 'FAIL'

            print(f'{var_k:<8.4f} {t:<10.5f} {sigma_t.item():<12.4e} '
                  f'{max_out:<14.3e} {finite_frac:<12.4f} '
                  f'{mean_out:<14.3e} {verdict:<10s}')

    print()
    print('--- var_k = 1 boundary characterization (regression baseline) ---')
    if not var1_records:
        print('  (no var_k=1 rows produced — sweep configuration error)')
    else:
        for r in var1_records:
            print(f"  t={r['t']:<10.5f}  finite_frac={r['finite_frac']:.4f}  "
                  f"|out| max={r['max_out']:.3e}  |out| mean={r['mean_out']:.3e}")
    print()
    print('Action: if any var_k=1 row above changes between runs at the same SHA,')
    print('        a code path that affects the boundary has shifted silently.')
    print('        Investigate before promoting Phase 4 to a live call site.')

    print()
    if hard_pass:
        print('=== GATE (f) HARD ASSERTIONS PASS ===')
        print('All var_k != 1 cells produced fully-finite output across the sweep.')
        return 0
    print('!!! GATE (f) HARD ASSERTIONS FAIL !!!')
    print('At least one (var_k != 1, t) combination produced non-finite output.')
    print('Phase 4 cannot proceed until this is resolved. Likely root causes:')
    print('  - C1 (zeta_max clamp) interacting badly with VP near t=1')
    print('  - denom underflow at the production eps clamp')
    print('  - gm_vars * zeta + 1 vanishing for var_k near 1')
    return 1


if __name__ == '__main__':
    sys.exit(main())
