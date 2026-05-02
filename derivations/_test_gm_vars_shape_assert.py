"""
Phase 2 work-item C2 — gm_vars shape assertion.

The whole correctness argument for `gmflow_posterior_mean_jit*` rests on
`gm_vars` being shape `(bs, *, 1, 1, 1, 1)` — i.e., variance shared across
the K mixture components and across channels (`gm_dim` and `channel_dim`).
The Mathematica completing-the-square proof (see `posterior_rederivation.nb`,
`02_mathematica_crosscheck.md`) only cancels the `vk·ν²/(2·dk)` term under
softmax when `var_k` is k-independent. A future architecture emitting
per-component variances would silently break the formula with no test failing.

C2 (see `derivations/03_rollout_plan.md` §2.2) adds:
    torch._assert(gm_vars.size(gm_dim) == 1, "...")
    torch._assert(gm_vars.size(channel_dim) == 1, "...")
inside both `gmflow_posterior_mean_jit` and `gmflow_posterior_mean_jit_general`.

This script verifies that:
  (1) Bad-shape gm_vars (per-component) raises a runtime error.
  (2) Bad-shape gm_vars (per-channel) raises a runtime error.
  (3) The legitimate shape `(B, 1, 1, 1, 1)` does NOT raise.
  (4) Both production JIT functions enforce the assertion (legacy and general).

Pre-C2 state: this script is **expected to fail** at tests 1–2 (the
assertion does not exist yet). The failure message points back at C2. The
test is committed before C2 ships so the work-item is closed by making this
script pass, not by writing a new test alongside the implementation.

Loading strategy mirrors `_test_differential_equivalence.py`: regex-extract
the JIT functions from the production source, no `lakonlab` import.
"""
import hashlib
import importlib.util
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


def _load_production_jits():
    here = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.normpath(os.path.join(
        here, '..', 'repos', 'piFlow', 'lakonlab', 'models', 'diffusions', 'gmflow.py'))
    if not os.path.exists(src_path):
        raise FileNotFoundError(f'Production source not found: {src_path}')

    with open(src_path) as f:
        text = f.read()

    fn_src_jit = _extract_function_source(text, 'gmflow_posterior_mean_jit')
    fn_src_gen = _extract_function_source(text, 'gmflow_posterior_mean_jit_general')

    sha_jit = hashlib.sha256(fn_src_jit.encode()).hexdigest()
    sha_gen = hashlib.sha256(fn_src_gen.encode()).hexdigest()

    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='_piflow_c2_extracted.py', delete=False)
    try:
        tmp.write('import torch\n\n')
        tmp.write(fn_src_jit)
        tmp.write('\n\n')
        tmp.write(fn_src_gen)
        tmp.flush()
        tmp_path = tmp.name
    finally:
        tmp.close()

    spec = importlib.util.spec_from_file_location('piflow_c2_extracted', tmp_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['piflow_c2_extracted'] = mod
    spec.loader.exec_module(mod)

    sig_gen = re.search(
        r'def gmflow_posterior_mean_jit_general\(([\s\S]*?)\):', fn_src_gen).group(1)
    has_zeta_max = 'zeta_max' in sig_gen

    return (
        mod.gmflow_posterior_mean_jit,
        mod.gmflow_posterior_mean_jit_general,
        has_zeta_max, src_path, sha_jit, sha_gen, fn_src_jit, fn_src_gen,
    )


def _make_batch(B, K, C, H, W, dtype, gm_vars_shape, seed):
    """`gm_vars_shape` is the *target* shape — caller picks legit or per-component."""
    g = torch.Generator().manual_seed(seed)
    return dict(
        gm_means=torch.randn(B, K, C, H, W, generator=g, dtype=dtype),
        gm_logweights=torch.randn(B, K, 1, H, W, generator=g, dtype=dtype),
        gm_vars=(torch.randn(*gm_vars_shape, generator=g, dtype=dtype) * 0.1).exp() ** 2,
        sigma_t_src=torch.tensor([[[[0.6]]]], dtype=dtype).expand(B, 1, 1, 1).contiguous(),
        sigma_t=torch.tensor([[[[0.3]]]], dtype=dtype).expand(B, 1, 1, 1).contiguous(),
        x_t_src=torch.randn(B, C, H, W, generator=g, dtype=dtype),
        x_t=torch.randn(B, C, H, W, generator=g, dtype=dtype),
    )


def _call_legacy(jit_legacy, d, eps=1e-6):
    return jit_legacy(
        d['sigma_t_src'], d['sigma_t'], d['x_t_src'], d['x_t'],
        d['gm_means'], d['gm_vars'], d['gm_logweights'], eps)


def _call_general(jit_general, d, has_zeta_max, eps=1e-6):
    alpha_t_src = 1 - d['sigma_t_src']
    alpha_t = 1 - d['sigma_t']
    if has_zeta_max:
        return jit_general(
            alpha_t_src, d['sigma_t_src'], alpha_t, d['sigma_t'],
            d['x_t_src'], d['x_t'],
            d['gm_means'], d['gm_vars'], d['gm_logweights'],
            eps, float('inf'))
    return jit_general(
        alpha_t_src, d['sigma_t_src'], alpha_t, d['sigma_t'],
        d['x_t_src'], d['x_t'],
        d['gm_means'], d['gm_vars'], d['gm_logweights'], eps)


def _expect_raises(label, callable_fn):
    """Run callable_fn and verify it raises with our explicit C2 assert message."""
    try:
        callable_fn()
    except (RuntimeError, AssertionError, torch.jit.Error) as e:
        msg = str(e)
        # Require the explicit 'gm_vars' string from our torch._assert message.
        # Accepting 'size'/'shape'/'assert' alone would allow incidental
        # TorchScript broadcast errors to produce a false PASS.
        ok = 'gm_vars' in msg
        if ok:
            print(f'  [{label}] PASS — raised: {type(e).__name__}: {msg[:100]}')
            return True
        print(f'  [{label}] FAIL — raised but message does not contain "gm_vars" '
              f'(got {type(e).__name__}: {msg[:200]})')
        return False
    print(f'  [{label}] FAIL — no exception raised. C2 assertion is missing or '
          f'admits the malformed input.')
    return False


def _expect_no_raise(label, callable_fn):
    try:
        out = callable_fn()
    except Exception as e:
        print(f'  [{label}] FAIL — unexpected exception on legitimate shape: '
              f'{type(e).__name__}: {str(e)[:200]}')
        return False
    finite = bool(torch.isfinite(out).all().item())
    if finite:
        print(f'  [{label}] PASS — legitimate shape accepted, output finite')
        return True
    print(f'  [{label}] FAIL — legitimate shape accepted but output non-finite')
    return False


def main():
    print('=' * 78)
    print('C2 — gm_vars shape assertion test')
    print('=' * 78)
    print()

    (jit_legacy, jit_general, has_zeta_max, src_path,
     sha_jit, sha_gen, fn_src_jit, fn_src_gen) = _load_production_jits()
    print(f'[setup] source: {src_path}')
    print(f'[setup] gmflow_posterior_mean_jit         '
          f'({len(fn_src_jit)} bytes) SHA-256: {sha_jit}')
    print(f'[setup] gmflow_posterior_mean_jit_general '
          f'({len(fn_src_gen)} bytes) SHA-256: {sha_gen}')
    print()

    B, K, C, H, W = 2, 4, 8, 8, 8
    dtype = torch.float32

    legitimate_shape = (B, 1, 1, 1, 1)
    per_component_shape = (B, K, 1, 1, 1)
    per_channel_shape = (B, 1, C, 1, 1)

    legit_batch = _make_batch(B, K, C, H, W, dtype, legitimate_shape, seed=0)
    bad_kshape_batch = _make_batch(B, K, C, H, W, dtype, per_component_shape, seed=1)
    bad_cshape_batch = _make_batch(B, K, C, H, W, dtype, per_channel_shape, seed=2)

    print('--- happy path (legitimate shape (B,1,1,1,1) must NOT raise) ---')
    h1 = _expect_no_raise(
        'legacy   / legit',
        lambda: _call_legacy(jit_legacy, legit_batch))
    h2 = _expect_no_raise(
        'general  / legit',
        lambda: _call_general(jit_general, legit_batch, has_zeta_max))

    print()
    print('--- malformed: per-component gm_vars (size at gm_dim != 1) ---')
    b1 = _expect_raises(
        'legacy   / per-K   ',
        lambda: _call_legacy(jit_legacy, bad_kshape_batch))
    b2 = _expect_raises(
        'general  / per-K   ',
        lambda: _call_general(jit_general, bad_kshape_batch, has_zeta_max))

    print()
    print('--- malformed: per-channel gm_vars (size at channel_dim != 1) ---')
    b3 = _expect_raises(
        'legacy   / per-C   ',
        lambda: _call_legacy(jit_legacy, bad_cshape_batch))
    b4 = _expect_raises(
        'general  / per-C   ',
        lambda: _call_general(jit_general, bad_cshape_batch, has_zeta_max))

    print()
    all_pass = h1 and h2 and b1 and b2 and b3 and b4
    if all_pass:
        print('=== C2 PASS ===')
        print('Both production JITs reject malformed gm_vars shapes and accept the')
        print("legitimate (B,1,1,1,1) shape. The Mathematica precondition (gm_vars"
              " shared across K) is now enforced at runtime.")
        return 0
    print('!!! C2 FAIL !!!')
    print('Likely cause (pre-C2): the runtime assertion has not been added to the')
    print('production JITs yet. See 03_rollout_plan.md §2.2 work-item C2.')
    print('Add `torch._assert(gm_vars.size(gm_dim) == 1, ...)` and the matching')
    print('channel-dim assert inside both `gmflow_posterior_mean_jit` and')
    print('`gmflow_posterior_mean_jit_general` in')
    print('`repos/piFlow/lakonlab/models/diffusions/gmflow.py`.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
