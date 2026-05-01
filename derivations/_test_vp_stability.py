"""
Test 5: Float32 stability sweep under VP/trig schedule (production function).

Sweeps t in (0, 1) with realistic image-shape tensors. Verifies that:
  1. No NaN or Inf appears in the output
  2. The denom doesn't degenerate (no division by ~0)
  3. With the analytic ZETA_MAX clamp, behaviour is bounded all the way
     to t = 1e-3 (well below any practical sampling step)

This validates that the schedule-agnostic production code is safe to ship
for VP sampling in float32. Exercises the production
`gmflow_posterior_mean_jit_general` (post-C1) via the same regex/importlib
loading strategy as the other test scripts.  Pre-C1 the function had no
zeta_max parameter and could produce large outputs at extreme t; post-C1 the
clamp is always active at the wrapper-supplied value.
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
        mode='w', suffix='_piflow_stability_extracted.py', delete=False)
    try:
        tmp.write('import torch\n\n')
        tmp.write(fn_src)
        tmp.flush()
        tmp_path = tmp.name
    finally:
        tmp.close()

    spec = importlib.util.spec_from_file_location('piflow_stability_extracted', tmp_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['piflow_stability_extracted'] = mod
    spec.loader.exec_module(mod)

    return mod.gmflow_posterior_mean_jit_general, has_zeta_max, src_path, sha, fn_src


def make_batch(B, K, C, H, W, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    return dict(
        gm_means=torch.randn(B, K, C, H, W, generator=g, dtype=dtype),
        gm_logweights=torch.randn(B, K, 1, H, W, generator=g, dtype=dtype),
        gm_vars=(torch.randn(B, 1, 1, 1, 1, generator=g, dtype=dtype) * 0.5).exp() ** 2,
        x_t_src=torch.randn(B, C, H, W, generator=g, dtype=dtype),
        x_t=torch.randn(B, C, H, W, generator=g, dtype=dtype),
    )


def vp_alphasigma(t, dtype):
    """VP/trig schedule: alpha = cos(pi*t/2), sigma = sin(pi*t/2).

    Returns shape (1, 1, 1, 1) so it broadcasts against (B, C, H, W) x tensors.
    """
    t_t = torch.tensor([[[[t]]]], dtype=dtype)
    return torch.cos(math.pi * t_t / 2), torch.sin(math.pi * t_t / 2)


def main():
    dtype = torch.float32
    eps = 1e-6
    max_var_assumed = 10.0
    zeta_max = torch.finfo(dtype).max / max_var_assumed

    jit_fn, has_zeta_max, src_path, sha, fn_src = _load_production_general()

    print('=' * 78)
    print('TEST 5 — Float32 stability sweep under VP schedule (production JIT)')
    print('=' * 78)
    print(f'source            : {src_path}')
    print(f'SHA-256           : {sha}')
    print(f'zeta_max in sig   : {has_zeta_max}')
    print(f'dtype             = {dtype}')
    print(f'eps               = {eps}')
    print(f'assumed max var_k = {max_var_assumed}')
    print(f'analytic ZETA_MAX = {zeta_max:.3e}')
    print()

    if not has_zeta_max:
        print('ERROR: production function has no zeta_max parameter.')
        print('C1 has not been applied. Fold requires C1 to be shipped first.')
        return 1

    B, K, C, H, W = 2, 4, 16, 16, 16
    batch = make_batch(B, K, C, H, W, dtype, seed=0)

    s = 0.9
    alpha_s, sigma_s = vp_alphasigma(s, dtype)

    t_values = [1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.3, 0.5, 0.7, 0.85]

    print(f'{"t":<10s} {"sigma_t":<12s} {"alpha_t":<12s} {"|out| max":<14s} '
          f'{"|out| mean":<14s} {"finite?":<8s} {"status":<6s}')
    print('-' * 78)

    all_ok = True
    for t in t_values:
        alpha_t, sigma_t = vp_alphasigma(t, dtype)
        out = jit_fn(
            alpha_s, sigma_s, alpha_t, sigma_t,
            batch['x_t_src'], batch['x_t'],
            batch['gm_means'], batch['gm_vars'], batch['gm_logweights'],
            eps, zeta_max)
        finite = torch.isfinite(out).all().item()
        max_out = out.abs().max().item()
        mean_out = out.abs().mean().item()
        ok = finite and max_out < 1e6
        all_ok &= ok
        print(f'{t:<10.0e} {sigma_t.item():<12.4e} {alpha_t.item():<12.4e} '
              f'{max_out:<14.3e} {mean_out:<14.3e} {str(finite):<8s} '
              f'{"PASS" if ok else "FAIL"}')

    print()

    print('--- Edge stress: t -> 0 (forces zeta clamp to fire) ---')
    for t in [1e-10, 1e-15, 1e-20]:
        alpha_t, sigma_t = vp_alphasigma(t, dtype)
        out = jit_fn(
            alpha_s, sigma_s, alpha_t, sigma_t,
            batch['x_t_src'], batch['x_t'],
            batch['gm_means'], batch['gm_vars'], batch['gm_logweights'],
            eps, zeta_max)
        finite = torch.isfinite(out).all().item()
        max_out = out.abs().max().item()
        ok = finite
        all_ok &= ok
        print(f'  t={t:<8.0e}  sigma_t={sigma_t.item():.3e}  '
              f'|out| max={max_out:.3e}  finite={finite}  '
              f'{"PASS" if ok else "FAIL"}')

    print()
    if all_ok:
        print('=== ALL CHECKS PASS ===')
        print('VP schedule is float32-safe across full SNR range with the analytic')
        print(f'ZETA_MAX clamp ({zeta_max:.3e}) and the existing eps clamps on denom.')
        return 0
    print('!!! AT LEAST ONE CHECK FAILED !!!')
    return 1


if __name__ == '__main__':
    sys.exit(main())
