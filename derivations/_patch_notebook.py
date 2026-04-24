"""Patch cell 18 with a C=1 exact comparison."""
import json

PATH = '/Users/youssefhemimy/Documents/nocm-piflow/derivations/01_posterior_rederivation.ipynb'
nb = json.load(open(PATH))

NEW_CELL_18 = '''\
# (c) Trig (VP) schedule: general formula vs exact brute-force
#
# The formula computes p(x0 | x_t) via importance reweighting:
#   log p(x0 | x_t) ∝ log p(x_t|x0) - log p(x_s|x0) + log p(x0|x_s)
#
# The logweight update sums over ALL channels jointly, so a single-channel
# brute force is only exact when C=1. We test with C=1 explicitly.

import math as _math

B1, K1, C1 = 2, 4, 1          # C=1 for exact 1D comparison

torch.manual_seed(99)          # independent seed — no dependency on cell 17 state
gm_means1      = torch.randn(B1, K1, C1)
gm_logweights1 = torch.randn(B1, K1, 1)
gm_vars1       = torch.rand(B1, 1, 1).abs() + 0.1

t_val = torch.rand(B1, 1) * 0.4 + 0.3
s_val = (t_val + torch.rand(B1, 1) * 0.2 + 0.1).clamp(max=0.95)

alpha_t_trig = torch.cos(_math.pi * t_val / 2)
sigma_t_trig = torch.sin(_math.pi * t_val / 2)
alpha_s_trig = torch.cos(_math.pi * s_val / 2)
sigma_s_trig = torch.sin(_math.pi * s_val / 2)

x0_true = torch.randn(B1, C1)
x_t_vp  = alpha_t_trig * x0_true + sigma_t_trig * torch.randn(B1, C1)
x_s_vp  = alpha_s_trig * x0_true + sigma_s_trig * torch.randn(B1, C1)

out_trig = posterior_general(
    alpha_t_trig, sigma_t_trig, alpha_s_trig, sigma_s_trig,
    x_t_vp, x_s_vp, gm_means1, gm_vars1, gm_logweights1)


def brute_force_1d(at, st, as_, ss,
                   mu_k_vals, var_k_val, logw_k_vals,
                   xt_val, xs_val, n_grid=20_000, std_range=12):
    """Exact 1D integration (C=1 case).
    Posterior ∝ p(x_t|x0)/p(x_s|x0) · p(x0|x_s).
    """
    K  = len(mu_k_vals)
    at, st, as_, ss = float(at), float(st), float(as_), float(ss)
    xt, xs, vk = float(xt_val), float(xs_val), float(var_k_val)

    nu_sc   = at*xt/st**2 - as_*xs/ss**2
    zeta_sc = at**2/st**2 - as_**2/ss**2
    d0      = vk*zeta_sc + 1
    center  = (float(mu_k_vals[0]) + vk*nu_sc) / (d0 if abs(d0) > 1e-8 else 1e-8)
    half    = std_range * (vk**0.5 + st / max(at, 1e-8))
    grid    = torch.linspace(center - half, center + half, n_grid)

    log_gm = torch.stack([
        float(logw_k_vals[k]) - 0.5*(grid - float(mu_k_vals[k]))**2 / vk
        for k in range(K)
    ]).logsumexp(dim=0)

    log_ratio = (-0.5*(xt - at*grid)**2/st**2) - (-0.5*(xs - as_*grid)**2/ss**2)

    log_post = log_gm + log_ratio
    log_post = log_post - log_post.logsumexp(dim=0)
    return float((log_post.exp() * grid).sum())


for b in range(B1):
    mean_analytic = float(out_trig[b, 0])
    mean_brute = brute_force_1d(
        float(alpha_t_trig[b, 0]), float(sigma_t_trig[b, 0]),
        float(alpha_s_trig[b, 0]), float(sigma_s_trig[b, 0]),
        gm_means1[b, :, 0], gm_vars1[b, 0, 0], gm_logweights1[b, :, 0],
        x_t_vp[b, 0], x_s_vp[b, 0])
    diff = abs(mean_analytic - mean_brute)
    status = "PASS" if diff < 0.01 else "FAIL"
    print(f"b={b}: analytic={mean_analytic:.6f}  brute={mean_brute:.6f}  |diff|={diff:.2e}  {status}")

assert all(
    abs(float(out_trig[b, 0]) - brute_force_1d(
        float(alpha_t_trig[b, 0]), float(sigma_t_trig[b, 0]),
        float(alpha_s_trig[b, 0]), float(sigma_s_trig[b, 0]),
        gm_means1[b, :, 0], gm_vars1[b, 0, 0], gm_logweights1[b, :, 0],
        x_t_vp[b, 0], x_s_vp[b, 0])) < 0.01
    for b in range(B1)
), "Trig brute-force mismatch"
print("PASS: trig posterior matches exact 1D brute-force for all batch elements")
'''

patched = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and (
        'brute_force_posterior_mean' in ''.join(cell['source']) or
        'brute_force_1d' in ''.join(cell['source'])
    ):
        cell['source'] = NEW_CELL_18
        print(f'Patched cell index {i}')
        patched = True
        break

if not patched:
    print('ERROR: target cell not found')
else:
    with open(PATH, 'w') as f:
        json.dump(nb, f, indent=1)
    print('Saved.')
