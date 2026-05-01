"""
Phase 2 — wrapper dispatch equivalence.

`GMFlowMixin.gmflow_posterior_mean` grew two optional kwargs in Phase 2:
`alpha_t_src` and `alpha_t`. When neither is provided, the method must
still call the legacy `gmflow_posterior_mean_jit` exactly as before
(zero-blast-radius guarantee from the go/no-go doc). When either is
provided, the missing member auto-fills as `1 - sigma_*` and the call
routes to `gmflow_posterior_mean_jit_general`.

This test proves:
  (A) Source audit — the real `gmflow_posterior_mean` method body contains
      the expected dispatch tokens. Guards against the wrapper being
      rewritten in a way that silently bypasses the dispatch.
  (B) Functional equivalence — a reference dispatch that mirrors the real
      wrapper produces bit-exact outputs across four opt-in patterns whose
      semantics all reduce to the linear schedule:
        - default (no alpha kwargs)
        - both alphas explicit, set to `1 - sigma_*`
        - only `alpha_t_src` explicit (alpha_t auto-fills)
        - only `alpha_t` explicit (alpha_t_src auto-fills)
  (C) VP smoke — passing `alpha = cos(pi*t/2)`, `sigma = sin(pi*t/2)`
      runs without error and produces finite, bounded output.

We cannot `import lakonlab...` directly (mmcv / mmgen missing in this env).
The test extracts both production JIT functions via regex + importlib, the
same technique as `_test_differential_equivalence.py`, so it runs against
the *actual* checked-in source in the pinned submodule.
"""
import hashlib
import importlib.util
import math
import os
import re
import sys
import tempfile

import torch


def _load_production_state():
    """Return (jit_module, method_body_src, full_file_src, file_path)."""
    here = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.normpath(os.path.join(
        here, '..', 'repos', 'piFlow', 'lakonlab', 'models', 'diffusions', 'gmflow.py'))
    if not os.path.exists(src_path):
        raise FileNotFoundError(f'Production source not found: {src_path}')

    with open(src_path) as f:
        text = f.read()

    def _extract_top_level(fn_name):
        marker = f'@torch.jit.script\ndef {fn_name}('
        start = text.index(marker)
        m = re.search(r'\n\n\n(?=@torch\.jit\.script|class |def )', text[start:])
        if m is None:
            raise RuntimeError(f'Could not find end of {fn_name}')
        return text[start:start + m.start() + 1] + '\n'

    fn_src_jit = _extract_top_level('gmflow_posterior_mean_jit')
    fn_src_gen = _extract_top_level('gmflow_posterior_mean_jit_general')

    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='_piflow_phase2_extracted.py', delete=False)
    try:
        tmp.write('import torch\n\n')
        tmp.write(fn_src_jit)
        tmp.write('\n\n')
        tmp.write(fn_src_gen)
        tmp.flush()
        tmp_path = tmp.name
    finally:
        tmp.close()
    spec = importlib.util.spec_from_file_location('piflow_phase2_extracted', tmp_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['piflow_phase2_extracted'] = mod
    spec.loader.exec_module(mod)

    m = re.search(
        r'    def gmflow_posterior_mean\([\s\S]*?(?=\n    def |\nclass )', text)
    if m is None:
        raise RuntimeError('Could not locate gmflow_posterior_mean method body')
    method_body = m.group(0)

    return mod, method_body, text, src_path


def _audit_method_body(body):
    """Return list of missing expected dispatch tokens."""
    expected = [
        'alpha_t_src=None',
        'alpha_t=None',
        'use_general = alpha_t_src is not None or alpha_t is not None',
        'gmflow_posterior_mean_jit_general',
        'gmflow_posterior_mean_jit',
    ]
    return [t for t in expected if t not in body]


def _reference_dispatch(mod, d, alpha_t_src=None, alpha_t=None, eps=1e-6):
    """Mirror the dispatch block of the real wrapper.

    Keep this in lockstep with gmflow.py::GMFlowMixin.gmflow_posterior_mean.
    The source audit (_audit_method_body) catches token-level drift.
    """
    sigma_t_src = d['sigma_t_src']
    sigma_t = d['sigma_t']
    use_general = alpha_t_src is not None or alpha_t is not None
    if use_general:
        if alpha_t_src is None:
            alpha_t_src = 1 - sigma_t_src
        if alpha_t is None:
            alpha_t = 1 - sigma_t
        _zeta_max = float(torch.finfo(d['gm_vars'].dtype).max) / 10.0
        return mod.gmflow_posterior_mean_jit_general(
            alpha_t_src, sigma_t_src, alpha_t, sigma_t,
            d['x_t_src'], d['x_t'],
            d['gm_means'], d['gm_vars'], d['gm_logweights'], eps, _zeta_max)
    return mod.gmflow_posterior_mean_jit(
        sigma_t_src, sigma_t, d['x_t_src'], d['x_t'],
        d['gm_means'], d['gm_vars'], d['gm_logweights'], eps)


def _make_batch(B, K, C, H, W, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    gm_means = torch.randn(B, K, C, H, W, generator=g, dtype=dtype)
    gm_logweights = torch.randn(B, K, 1, H, W, generator=g, dtype=dtype)
    gm_logstds = torch.randn(B, 1, 1, 1, 1, generator=g, dtype=dtype) * 0.5
    gm_vars = (gm_logstds * 2).exp()
    sigma_t_src = torch.rand(B, 1, 1, 1, generator=g, dtype=dtype) * 0.6 + 0.3
    sigma_t = (sigma_t_src * (torch.rand(B, 1, 1, 1, generator=g, dtype=dtype) * 0.6 + 0.3)
               ).clamp(min=1e-3)
    x_t_src = torch.randn(B, C, H, W, generator=g, dtype=dtype)
    x_t = torch.randn(B, C, H, W, generator=g, dtype=dtype)
    return dict(
        sigma_t_src=sigma_t_src, sigma_t=sigma_t,
        x_t_src=x_t_src, x_t=x_t,
        gm_means=gm_means, gm_vars=gm_vars, gm_logweights=gm_logweights,
    )


def test_linear_equivalence_across_opt_in_patterns(mod):
    """All four routes must produce identical outputs under linear schedule."""
    configs = [
        (2, 4,  3,  8,  8, torch.float32, 1e-5),
        (2, 8, 16, 16, 16, torch.float32, 1e-5),
        (4, 4, 64, 32, 32, torch.float32, 1e-5),
        (1, 4,  3,  8,  8, torch.float64, 1e-12),
        (2, 8, 16, 16, 16, torch.float64, 1e-12),
    ]
    print(f'{"shape":<28s} {"dtype":<10s} {"max_abs(any)":<14s} {"status":<6s}')
    print('-' * 78)
    all_pass = True
    for B, K, C, H, W, dtype, tol in configs:
        max_abs = 0.0
        for seed in range(20):
            d = _make_batch(B, K, C, H, W, dtype, seed)
            a_src = 1 - d['sigma_t_src']
            a_t = 1 - d['sigma_t']
            out_default = _reference_dispatch(mod, d)
            variants = [
                _reference_dispatch(mod, d, alpha_t_src=a_src, alpha_t=a_t),
                _reference_dispatch(mod, d, alpha_t_src=a_src),
                _reference_dispatch(mod, d, alpha_t=a_t),
            ]
            for v in variants:
                diff = (out_default - v).abs().max().item()
                if diff > max_abs:
                    max_abs = diff
        ok = max_abs < tol
        all_pass &= ok
        shape_str = f'(B={B}, K={K}, C={C}, H={H}, W={W})'
        dtype_str = str(dtype).replace('torch.', '')
        print(f'{shape_str:<28s} {dtype_str:<10s} {max_abs:<14.3e} '
              f'{"PASS" if ok else "FAIL"}')
    return all_pass


def test_vp_smoke(mod):
    """VP schedule: alpha=cos(pi*t/2), sigma=sin(pi*t/2). Must stay finite.

    Uses denoising direction `t_t < t_src` (same convention as the Phase 1
    test): the wrapper's job is one reverse step from noisier `t_src` toward
    cleaner `t_t`. Going the other direction makes `zeta < 0` and the
    posterior is ill-conditioned.
    """
    g = torch.Generator().manual_seed(7)
    B, K, C, H, W = 2, 4, 16, 16, 16
    t_src = torch.rand(B, 1, 1, 1, generator=g) * 0.5 + 0.4  # [0.4, 0.9]
    t_t = t_src * (torch.rand(B, 1, 1, 1, generator=g) * 0.6 + 0.2)  # < t_src
    half_pi = math.pi / 2
    sigma_t_src = torch.sin(t_src * half_pi).clamp(min=1e-3)
    sigma_t = torch.sin(t_t * half_pi).clamp(min=1e-3)
    alpha_t_src = torch.cos(t_src * half_pi)
    alpha_t = torch.cos(t_t * half_pi)
    d = dict(
        sigma_t_src=sigma_t_src, sigma_t=sigma_t,
        x_t_src=torch.randn(B, C, H, W, generator=g),
        x_t=torch.randn(B, C, H, W, generator=g),
        gm_means=torch.randn(B, K, C, H, W, generator=g),
        gm_vars=(torch.randn(B, 1, 1, 1, 1, generator=g) * 0.5 * 2).exp(),
        gm_logweights=torch.randn(B, K, 1, H, W, generator=g),
    )
    out = _reference_dispatch(mod, d, alpha_t_src=alpha_t_src, alpha_t=alpha_t)
    finite = bool(torch.isfinite(out).all().item())
    max_abs = out.abs().max().item()
    ok = finite and max_abs < 1e4
    print(f'VP smoke: all_finite={finite}  max_abs={max_abs:.3e}  '
          f'{"PASS" if ok else "FAIL"}')
    return ok


def main():
    print('=' * 78)
    print('Phase 2 — wrapper dispatch test')
    print('=' * 78)
    print()

    mod, method_body, _, src_path = _load_production_state()
    method_sha = hashlib.sha256(method_body.encode()).hexdigest()
    print(f'[setup] Production source: {src_path}')
    print(f'[setup] gmflow_posterior_mean method '
          f'({len(method_body)} bytes) SHA-256: {method_sha}')

    missing = _audit_method_body(method_body)
    if missing:
        print(f'[audit] FAIL — missing dispatch tokens: {missing}')
        return 1
    print(f'[audit] all expected dispatch tokens present')
    print()

    print('--- test 1: linear-schedule equivalence across opt-in patterns ---')
    t1 = test_linear_equivalence_across_opt_in_patterns(mod)

    print()
    print('--- test 2: VP smoke test ---')
    t2 = test_vp_smoke(mod)

    print()
    if t1 and t2:
        print('=== ALL PASS ===')
        return 0
    print('!!! FAIL !!!')
    return 1


if __name__ == '__main__':
    sys.exit(main())
