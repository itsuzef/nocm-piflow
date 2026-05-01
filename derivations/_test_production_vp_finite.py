"""
Gate V5 — production VP-stability empirical re-check.

Independently re-verifies the empirical claim from the five-reviewer
synthesis (`experiments/2026-04-24-five-reviewers/SYNTHESIS.md` §2,
disagreement B): that the **production** `gmflow_posterior_mean_jit_general`
stays numerically finite at small `t` under the VP schedule, even though
it has no `zeta_max` clamp. Only one reviewer (`general`) measured this
directly; the synthesis explicitly notes the result was not independently
re-verified.

This script is the V5 gate from `derivations/03_rollout_plan.md` §1.

Sweep:
    t       in {1e-3, 1e-5, 1e-10, 1e-20}
    var_k   in {1.0, 5.0, 10.0}
    s_src   in {0.5, 0.9}

Pass condition: every output tensor satisfies `torch.isfinite(out).all()`.

Forward-compatibility note: post-Phase-2 work-item C1, the production
function will gain a `zeta_max` argument. To keep this gate honest (it
verifies the *unclamped* path), if `zeta_max` is present in the production
signature we pass `float('inf')` to disable the clamp. The gate question
is: does the production formula itself stay finite, not whether the clamp
catches it.

Runs against the actual checked-in submodule source via the same
regex/importlib trick as `_test_differential_equivalence.py` and
`_test_wrapper_dispatch.py`. No `lakonlab` package import required.
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
    """Return (jit_fn, signature_has_zeta_max, src_path, sha256, fn_src)."""
    here = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.normpath(os.path.join(
        here, '..', 'repos', 'piFlow', 'lakonlab', 'models', 'diffusions', 'gmflow.py'))
    if not os.path.exists(src_path):
        raise FileNotFoundError(f'Production source not found: {src_path}')

    with open(src_path) as f:
        text = f.read()

    fn_src = _extract_function_source(text, 'gmflow_posterior_mean_jit_general')
    sha = hashlib.sha256(fn_src.encode()).hexdigest()

    sig_line_match = re.search(
        r'def gmflow_posterior_mean_jit_general\(([\s\S]*?)\):', fn_src)
    if sig_line_match is None:
        raise RuntimeError('Could not parse signature of gmflow_posterior_mean_jit_general')
    sig_text = sig_line_match.group(1)
    has_zeta_max = 'zeta_max' in sig_text

    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='_piflow_v5_extracted.py', delete=False)
    try:
        tmp.write('import torch\n\n')
        tmp.write(fn_src)
        tmp.flush()
        tmp_path = tmp.name
    finally:
        tmp.close()

    spec = importlib.util.spec_from_file_location('piflow_v5_extracted', tmp_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['piflow_v5_extracted'] = mod
    spec.loader.exec_module(mod)

    return mod.gmflow_posterior_mean_jit_general, has_zeta_max, src_path, sha, fn_src


def _vp_alpha_sigma(t, dtype, eps=1e-30):
    """VP/trig schedule: alpha = cos(pi*t/2), sigma = sin(pi*t/2)."""
    t_t = torch.tensor([[[[t]]]], dtype=dtype)
    sigma = torch.sin(math.pi * t_t / 2).clamp(min=eps)
    alpha = torch.cos(math.pi * t_t / 2)
    return alpha, sigma


def _make_batch(B, K, C, H, W, dtype, var_k, seed):
    """gm_vars is shape (B, 1, 1, 1, 1) — shared across components, per the
    Mathematica precondition. We override its scalar value per the sweep."""
    g = torch.Generator().manual_seed(seed)
    return dict(
        gm_means=torch.randn(B, K, C, H, W, generator=g, dtype=dtype),
        gm_logweights=torch.randn(B, K, 1, H, W, generator=g, dtype=dtype),
        gm_vars=torch.full((B, 1, 1, 1, 1), float(var_k), dtype=dtype),
        x_t_src=torch.randn(B, C, H, W, generator=g, dtype=dtype),
        x_t=torch.randn(B, C, H, W, generator=g, dtype=dtype),
    )


def _call_general(jit_fn, has_zeta_max, *,
                  alpha_t_src, sigma_t_src, alpha_t, sigma_t,
                  x_t_src, x_t, gm_means, gm_vars, gm_logweights, eps):
    """Invoke the production JIT, disabling the clamp if it exists."""
    if has_zeta_max:
        return jit_fn(
            alpha_t_src, sigma_t_src, alpha_t, sigma_t, x_t_src, x_t,
            gm_means, gm_vars, gm_logweights, eps, float('inf'))
    return jit_fn(
        alpha_t_src, sigma_t_src, alpha_t, sigma_t, x_t_src, x_t,
        gm_means, gm_vars, gm_logweights, eps)


def main():
    print('=' * 78)
    print('GATE V5 — production VP-finite check (unclamped path)')
    print('=' * 78)
    print()

    jit_fn, has_zeta_max, src_path, sha, fn_src = _load_production_general()
    print(f'[setup] source: {src_path}')
    print(f'[setup] gmflow_posterior_mean_jit_general '
          f'({len(fn_src)} bytes) SHA-256: {sha}')
    print(f'[setup] zeta_max in signature: {has_zeta_max}'
          f'{" (passing inf to disable clamp)" if has_zeta_max else ""}')
    print()

    dtype = torch.float32
    eps = 1e-6
    B, K, C, H, W = 2, 4, 16, 16, 16

    t_values = [1e-3, 1e-5, 1e-10, 1e-20]
    var_values = [1.0, 5.0, 10.0]
    s_values = [0.5, 0.9]

    header = (f'{"s_src":<8s} {"var_k":<8s} {"t":<10s} '
              f'{"sigma_t":<12s} {"|out| max":<14s} {"|out| mean":<14s} '
              f'{"finite?":<8s} {"status":<6s}')
    print(header)
    print('-' * 78)

    all_ok = True
    for s in s_values:
        alpha_s, sigma_s = _vp_alpha_sigma(s, dtype)
        for var_k in var_values:
            batch = _make_batch(B, K, C, H, W, dtype, var_k, seed=int(var_k * 17 + s * 31))
            for t in t_values:
                alpha_t, sigma_t = _vp_alpha_sigma(t, dtype)
                out = _call_general(
                    jit_fn, has_zeta_max,
                    alpha_t_src=alpha_s, sigma_t_src=sigma_s,
                    alpha_t=alpha_t, sigma_t=sigma_t,
                    x_t_src=batch['x_t_src'], x_t=batch['x_t'],
                    gm_means=batch['gm_means'], gm_vars=batch['gm_vars'],
                    gm_logweights=batch['gm_logweights'], eps=eps)
                finite = bool(torch.isfinite(out).all().item())
                max_out = out.abs().max().item()
                mean_out = out.abs().mean().item() if finite else float('nan')
                ok = finite
                all_ok &= ok
                print(f'{s:<8.2f} {var_k:<8.1f} {t:<10.0e} '
                      f'{sigma_t.item():<12.4e} '
                      f'{max_out:<14.3e} {mean_out:<14.3e} '
                      f'{str(finite):<8s} {"PASS" if ok else "FAIL"}')

    print()
    if all_ok:
        print('=== V5 PASS ===')
        print('Production gmflow_posterior_mean_jit_general stays finite across the')
        print('full sweep of (s, var_k, t) on the unclamped path. The empirical claim')
        print('from the five-reviewer synthesis is independently re-verified.')
        print()
        print('Note: this does not mean the unclamped path is *robust* — finite output')
        print('here arises from inf/inf cancellation in (out_means / denom). The C1')
        print('clamp should still ship for hygiene (see 03_rollout_plan.md §2.2 C1).')
        return 0
    print('!!! V5 FAIL — at least one (s, var_k, t) produced non-finite output !!!')
    print('Action: caveat C1 escalates from blocking-Phase-4 to blocking-Phase-2.')
    print('The zeta_max clamp must ship in the same PR as the wrapper hygiene work.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
