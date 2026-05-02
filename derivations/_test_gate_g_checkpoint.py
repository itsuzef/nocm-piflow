"""
Gate (g) — Real-checkpoint linear-schedule equivalence harness.

PURPOSE
-------
Proves that the Phase 2 hygiene bundle (C1–C4) produces bit-exact identical
outputs to the pre-Phase-2 baseline on the *linear schedule* path — the path
that every existing call site uses. The synthetic-tensor tests (V3, V4) cannot
rule out divergence from extreme means or near-degenerate GM weights produced
by a real trained model; this script does.

RUN PENDING
-----------
This script cannot run locally — it requires:
  1. mmcv + mmgen + lakonlab installed (not available in this env)
  2. Model checkpoint (download or local copy):
       huggingface://Lakonik/pi-Flow-ImageNet/gmdit_k32_imagenet_piid_1step/
       diffusion_pytorch_model.safetensors
  3. GPU (bfloat16 denoising model)

Run it on the research server / CI machine, in the piFlow repo root:
  # Step 1 — record baseline at the pre-Phase-2 SHA
  git -C repos/piFlow checkout 6190862
  python derivations/_test_gate_g_checkpoint.py --record \\
      --checkpoint <path_to_safetensors_or_hf_id> \\
      --output derivations/_gate_g_baseline.pt

  # Step 2 — verify at the post-Phase-2 SHA
  git -C repos/piFlow checkout a22f5e1
  python derivations/_test_gate_g_checkpoint.py --verify \\
      --checkpoint <path_to_safetensors_or_hf_id> \\
      --baseline derivations/_gate_g_baseline.pt

PASS condition: max abs diff = 0.0 across all (batch × spatial) elements for
each of the N_STEPS gmflow_posterior_mean calls in the fixed-seed run.

DESIGN NOTES
------------
- Does NOT use the mmcv test runner or ImageNet dataloader — only the denoising
  model + gmflow_posterior_mean. This keeps the script portable across runner
  versions and avoids the 50k-image eval overhead.
- Uses a fixed synthetic latent input (torch.randn with seed) shaped to match
  the real model's latent dimensions (B=4, C=4, H=32, W=32). The denoising
  model is real (loaded from checkpoint); only the initial x_T is synthetic.
  This exercises the full GM output distribution of the trained model.
- Fixes class labels (class 0, the goldfish, repeated) so conditioning is
  deterministic.
- Runs N_STEPS = 4 denoising steps (not 128 — fast, sufficient for catching
  any non-identity behaviour on the linear path; the zeta_max clamp only fires
  at extreme alpha/sigma ratios not reached in normal sampling).
- Records the full sequence of gmflow_posterior_mean outputs (one per substep
  after the first), not just final x_T, so a divergence can be pinpointed to
  the exact step.

See derivations/03_rollout_plan.md §4.1 gate (g) and §2.5.
"""

import argparse
import sys
import os

# ---------------------------------------------------------------------------
# Environment check — fail early with a clear message before any imports
# ---------------------------------------------------------------------------
_MISSING = []
for _pkg in ('mmcv', 'mmgen', 'torch', 'safetensors'):
    try:
        __import__(_pkg)
    except ImportError:
        _MISSING.append(_pkg)
if _MISSING:
    print('ERROR: gate (g) harness requires packages not available in this env:')
    for p in _MISSING:
        print(f'  missing: {p}')
    print()
    print('Run this script on the research server where mmcv/mmgen/lakonlab are installed.')
    print('See the module docstring for exact commands.')
    sys.exit(2)

import torch
import math

# After env check, do real imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'repos', 'piFlow'))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 20260501            # fixed seed for reproducibility
N_STEPS = 4                # denoising steps (fast; linear-path test only)
N_SUBSTEPS = 8             # substeps per step (triggers gmflow_posterior_mean)
BATCH_SIZE = 4             # small batch — enough to exercise real GM weights
LATENT_C, LATENT_H, LATENT_W = 4, 32, 32
CLASS_LABEL = 0            # goldfish (class 0 in ImageNet)
NUM_TIMESTEPS = 1000
CHECKPOINT_HF = (
    'Lakonik/pi-Flow-ImageNet/'
    'gmdit_k32_imagenet_piid_1step/diffusion_pytorch_model.safetensors')


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_model(checkpoint):
    """Load GMDiT denoising model + PiFlowImitation wrapper.

    `checkpoint` is either:
      - a local path to a .safetensors or .pth file
      - a HuggingFace model id (e.g. 'huggingface://Lakonik/...')
    """
    from mmcv import Config
    from mmgen.models.builder import build_module

    cfg_path = os.path.join(
        os.path.dirname(__file__), '..', 'repos', 'piFlow',
        'configs', 'piflow_imagenet', 'gmdit_k32_imagenet_piid_1step_test.py')
    cfg = Config.fromfile(cfg_path)

    # Override pretrained with the caller-supplied checkpoint path
    cfg.model.diffusion.denoising.pretrained = checkpoint
    cfg.model.diffusion_use_ema = False   # skip EMA swap for speed

    model = build_module(cfg.model)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


# ---------------------------------------------------------------------------
# Sampling harness — patches gmflow_posterior_mean to intercept outputs
# ---------------------------------------------------------------------------
class _InterceptingWrapper:
    """Thin shim that calls the real gmflow_posterior_mean and records results."""

    def __init__(self, diffusion_model):
        self._model = diffusion_model
        self.recorded = []   # list of (step_id, substep_id, output_tensor)
        self._orig = diffusion_model.gmflow_posterior_mean

    def __enter__(self):
        model = self._model
        recorded = self.recorded
        orig = self._orig

        def _patched(*args, **kwargs):
            out = orig(*args, **kwargs)
            recorded.append(out.detach().cpu().clone())
            return out

        model.gmflow_posterior_mean = _patched
        return self

    def __exit__(self, *_):
        self._model.gmflow_posterior_mean = self._orig


def _run_sampling(model):
    """Run a fixed-seed sampling job. Returns list of intercepted outputs."""
    device = next(model.parameters()).device
    dtype = torch.float32

    g = torch.Generator(device=device).manual_seed(SEED)
    x_T = torch.randn(
        BATCH_SIZE, LATENT_C, LATENT_H, LATENT_W,
        generator=g, dtype=dtype, device=device)
    class_labels = torch.full(
        (BATCH_SIZE,), CLASS_LABEL, dtype=torch.long, device=device)

    timesteps = torch.linspace(NUM_TIMESTEPS, 1, N_STEPS, dtype=dtype, device=device)

    with _InterceptingWrapper(model.diffusion) as w:
        with torch.no_grad():
            # Simplified sampling loop — mirrors the substep logic in
            # gmflow.py sample() without the full pipeline overhead.
            x_t = x_T.clone()
            for step_id, t in enumerate(timesteps):
                t_scalar = t.item()
                t_tensor = t.expand(BATCH_SIZE, 1, 1, 1)
                sigma_t = t_tensor / NUM_TIMESTEPS

                # Forward pass — get real GM output from the trained model
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                    enabled=torch.cuda.is_available()):
                    gm_output = model.diffusion.denoising(
                        x_t, t_tensor.squeeze(-1).squeeze(-1).squeeze(-1),
                        class_labels=class_labels)

                x_t_base = x_t.clone()
                t_base = t_tensor.clone()

                for sub_id in range(N_SUBSTEPS):
                    if sub_id == 0:
                        # First substep: plain x0 prediction, no gmflow_posterior_mean
                        model_output = model.diffusion.gm_to_model_output(
                            gm_output, 'mean')
                    else:
                        # Subsequent substeps: gmflow_posterior_mean is called
                        t_sub = t - (t - (timesteps[step_id + 1]
                                         if step_id + 1 < len(timesteps)
                                         else torch.zeros_like(t))) * sub_id / N_SUBSTEPS
                        t_sub_tensor = t_sub.expand(BATCH_SIZE, 1, 1, 1)
                        model_output = model.diffusion.gmflow_posterior_mean(
                            gm_output, x_t, x_t_base,
                            t_sub_tensor, t_base,
                            prediction_type='x0')

                    # Simple Euler step
                    sigma_t_next = (
                        (timesteps[step_id + 1] if step_id + 1 < len(timesteps)
                         else torch.zeros_like(t))
                        / NUM_TIMESTEPS).expand(BATCH_SIZE, 1, 1, 1)
                    x_t = x_t + (sigma_t_next - sigma_t) * (x_t - model_output) / sigma_t.clamp(min=1e-6)

    return w.recorded  # list of cpu tensors, one per gmflow_posterior_mean call


# ---------------------------------------------------------------------------
# Record / verify modes
# ---------------------------------------------------------------------------
def cmd_record(args):
    print('=' * 78)
    print('GATE (g) — RECORD mode')
    print('=' * 78)
    print(f'checkpoint : {args.checkpoint}')
    print(f'output     : {args.output}')
    print(f'seed       : {SEED}   steps: {N_STEPS}   substeps: {N_SUBSTEPS}')
    print()

    model = _load_model(args.checkpoint)
    submodule_sha = _get_submodule_sha()
    print(f'submodule SHA: {submodule_sha}')
    print('Running sampling...')
    outputs = _run_sampling(model)

    baseline = dict(
        submodule_sha=submodule_sha,
        seed=SEED,
        n_steps=N_STEPS,
        n_substeps=N_SUBSTEPS,
        batch_size=BATCH_SIZE,
        outputs=outputs,
    )
    torch.save(baseline, args.output)
    n_calls = len(outputs)
    print(f'Recorded {n_calls} gmflow_posterior_mean calls.')
    print(f'Baseline saved to: {args.output}')
    print()
    print('Next: checkout post-Phase-2 SHA (a22f5e1) and run --verify.')


def cmd_verify(args):
    print('=' * 78)
    print('GATE (g) — VERIFY mode')
    print('=' * 78)
    print(f'checkpoint : {args.checkpoint}')
    print(f'baseline   : {args.baseline}')
    print()

    baseline = torch.load(args.baseline, weights_only=False)
    baseline_sha = baseline['submodule_sha']
    baseline_outputs = baseline['outputs']
    current_sha = _get_submodule_sha()

    print(f'baseline SHA : {baseline_sha}')
    print(f'current SHA  : {current_sha}')
    print(f'baseline calls recorded: {len(baseline_outputs)}')
    print()

    if baseline['seed'] != SEED:
        print(f'ERROR: baseline seed {baseline["seed"]} != harness SEED {SEED}')
        return 1
    if baseline['n_steps'] != N_STEPS or baseline['n_substeps'] != N_SUBSTEPS:
        print('ERROR: step/substep count mismatch — re-record baseline.')
        return 1

    model = _load_model(args.checkpoint)
    print('Running sampling...')
    current_outputs = _run_sampling(model)

    if len(current_outputs) != len(baseline_outputs):
        print(f'ERROR: call count mismatch '
              f'(baseline={len(baseline_outputs)}, current={len(current_outputs)})')
        return 1

    print(f'{"call":<6s} {"shape":<30s} {"max_abs_diff":<16s} {"status":<6s}')
    print('-' * 62)
    all_pass = True
    for i, (b, c) in enumerate(zip(baseline_outputs, current_outputs)):
        diff = (b.float() - c.float()).abs().max().item()
        ok = diff == 0.0
        all_pass &= ok
        print(f'{i:<6d} {str(tuple(b.shape)):<30s} {diff:<16.3e} '
              f'{"PASS" if ok else "FAIL"}')

    print()
    if all_pass:
        print('=== GATE (g) PASS ===')
        print('Post-Phase-2 SHA produces bit-exact identical gmflow_posterior_mean')
        print('outputs to the pre-Phase-2 baseline on the linear schedule path.')
        print()
        print('Action: check the last §2.5 exit criterion in 03_rollout_plan.md')
        print('and update §0 with this SHA pair and run date.')
        return 0

    print('!!! GATE (g) FAIL !!!')
    print('At least one gmflow_posterior_mean call produced a non-zero diff.')
    print('Phase 4 cannot proceed until the source of divergence is found.')
    print('Likely causes:')
    print('  - C1 zeta_max clamp firing on real model weights (check max |zeta|)')
    print('  - C2 assertion failing silently (check gm_vars shape in checkpoint)')
    print('  - Unexpected change in default-path dispatch (re-audit V3/V4)')
    return 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_submodule_sha():
    import subprocess
    result = subprocess.run(
        ['git', '-C',
         os.path.join(os.path.dirname(__file__), '..', 'repos', 'piFlow'),
         'rev-parse', 'HEAD'],
        capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else 'unknown'


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Gate (g) checkpoint harness — record or verify.')
    sub = parser.add_subparsers(dest='cmd')

    rec = sub.add_parser('--record', help='Record baseline outputs.')
    rec.add_argument('--checkpoint', required=True,
                     help='Path or HF id for the denoising model safetensors.')
    rec.add_argument('--output', default='derivations/_gate_g_baseline.pt',
                     help='Where to save the baseline tensor bundle.')

    ver = sub.add_parser('--verify', help='Verify against recorded baseline.')
    ver.add_argument('--checkpoint', required=True,
                     help='Path or HF id for the denoising model safetensors.')
    ver.add_argument('--baseline', default='derivations/_gate_g_baseline.pt',
                     help='Path to the baseline tensor bundle from --record.')

    # Support flat flag style: script.py --record --checkpoint ... --output ...
    args, _ = parser.parse_known_args()
    if '--record' in sys.argv:
        parser2 = argparse.ArgumentParser()
        parser2.add_argument('--checkpoint', required=True)
        parser2.add_argument('--output', default='derivations/_gate_g_baseline.pt')
        a = parser2.parse_args([x for x in sys.argv[1:] if x != '--record'])
        return cmd_record(a)
    elif '--verify' in sys.argv:
        parser2 = argparse.ArgumentParser()
        parser2.add_argument('--checkpoint', required=True)
        parser2.add_argument('--baseline', default='derivations/_gate_g_baseline.pt')
        a = parser2.parse_args([x for x in sys.argv[1:] if x != '--verify'])
        return cmd_verify(a)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
