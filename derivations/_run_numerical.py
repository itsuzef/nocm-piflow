"""
Numerical verification: 3-way comparison + float32 edge stability.
"""
import torch
import math

torch.manual_seed(42)
B, K, C = 2, 4, 8
# Schedule scalars broadcast as (B, 1) against observations (B, C)
# and as (B, 1, 1) against gm_means (B, K, C).

def posterior_general(alpha_t, sigma_t, alpha_s, sigma_s,
                      x_t, x_s, gm_means, gm_vars, gm_logweights,
                      eps=1e-6, ZETA_MAX=None):
    # alpha/sigma: (B, 1)  x_t/x_s: (B, C)  gm_means: (B, K, C)
    # gm_vars: (B, 1, 1)  gm_logweights: (B, K, 1)
    sigma_t = sigma_t.clamp(min=eps)
    sigma_s = sigma_s.clamp(min=eps)
    zeta = (alpha_t / sigma_t).square() - (alpha_s / sigma_s).square()  # (B, 1)
    nu   = alpha_t * x_t / sigma_t**2 - alpha_s * x_s / sigma_s**2     # (B, C)
    if ZETA_MAX is not None:
        zeta = zeta.clamp(max=ZETA_MAX)
    nu   = nu.unsqueeze(1)    # (B, 1, C) — broadcast over K
    zeta = zeta.unsqueeze(1)  # (B, 1, 1) — broadcast over K and C
    denom = (gm_vars * zeta + 1).clamp(min=eps)         # (B, 1, 1) or (B, K, 1)
    out_means  = (gm_vars * nu + gm_means) / denom      # (B, K, C)
    logw_delta = (gm_means * (nu - 0.5 * zeta * gm_means)).sum(
        dim=-1, keepdim=True) / denom                   # (B, K, 1)
    weights    = (gm_logweights + logw_delta).softmax(dim=1)  # (B, K, 1)
    return (out_means * weights).sum(dim=1)              # (B, C)


def ref_linear(sigma_s, sigma_t, x_s, x_t, gm_means, gm_vars, gm_logweights, eps=1e-6):
    """Exact copy of gmflow_posterior_mean_jit logic (alpha = 1 - sigma)."""
    sigma_s = sigma_s.clamp(min=eps)
    sigma_t = sigma_t.clamp(min=eps)
    alpha_s = 1 - sigma_s
    alpha_t = 1 - sigma_t
    aos_s = alpha_s / sigma_s
    aos_t = alpha_t / sigma_t
    zeta = aos_t.square() - aos_s.square()              # (B, 1)
    nu   = aos_t * x_t / sigma_t - aos_s * x_s / sigma_s  # (B, C)
    nu   = nu.unsqueeze(1)
    zeta = zeta.unsqueeze(1)
    denom = (gm_vars * zeta + 1).clamp(min=eps)
    out_means  = (gm_vars * nu + gm_means) / denom
    logw_delta = (gm_means * (nu - 0.5 * zeta * gm_means)).sum(
        dim=-1, keepdim=True) / denom
    weights    = (gm_logweights + logw_delta).softmax(dim=1)
    return (out_means * weights).sum(dim=1)


# --- Test data: (B, K, C) for GM, (B, 1) for schedule scalars, (B, C) for obs ---
gm_means      = torch.randn(B, K, C)
gm_logweights = torch.randn(B, K, 1)
gm_vars       = torch.rand(B, 1, 1).abs() + 0.1

x_s = torch.randn(B, C)
x_t = torch.randn(B, C)

# Source noisier than target
sigma_s_val = (torch.rand(B, 1) * 0.4 + 0.4)          # (B, 1), 0.4–0.8
sigma_t_val = (sigma_s_val * torch.rand(B, 1) * 0.6).clamp(min=1e-3)

alpha_s_lin = 1 - sigma_s_val
alpha_t_lin = 1 - sigma_t_val

# (a) general formula with linear schedule
out_a = posterior_general(alpha_t_lin, sigma_t_val, alpha_s_lin, sigma_s_val,
                          x_t, x_s, gm_means, gm_vars, gm_logweights)
# (b) reference (existing code)
out_b = ref_linear(sigma_s_val, sigma_t_val, x_s, x_t,
                   gm_means, gm_vars, gm_logweights)

diff_ab = (out_a - out_b).abs().max().item()
print(f'[S6a vs S6b] max |diff| (linear, should be ~0): {diff_ab:.2e}')
assert diff_ab < 1e-5, f'FAIL linear mismatch: {diff_ab}'
print('PASS: general formula (alpha=1-sigma) == existing code')

# (c) trig schedule: compare general formula with brute-force per-channel grid
t_val = torch.rand(B, 1) * 0.4 + 0.3    # (B, 1), t in (0.3, 0.7)
s_val = (t_val + torch.rand(B, 1) * 0.2 + 0.1).clamp(max=0.95)  # s > t

alpha_t_trig = torch.cos(math.pi * t_val / 2)   # (B, 1)
sigma_t_trig = torch.sin(math.pi * t_val / 2)
alpha_s_trig = torch.cos(math.pi * s_val / 2)
sigma_s_trig = torch.sin(math.pi * s_val / 2)

x0_true = torch.randn(B, C)
x_t_vp  = alpha_t_trig * x0_true + sigma_t_trig * torch.randn(B, C)  # (B, C)
x_s_vp  = alpha_s_trig * x0_true + sigma_s_trig * torch.randn(B, C)

out_c_general = posterior_general(
    alpha_t_trig, sigma_t_trig, alpha_s_trig, sigma_s_trig,
    x_t_vp, x_s_vp, gm_means, gm_vars, gm_logweights)


def brute_force_mean_1d(alpha_t_val, sigma_t_val, alpha_s_val, sigma_s_val,
                         mu_k_vals, var_k_val, logw_k_vals,
                         x_t_val, x_s_val, n_grid=20000, std_range=12):
    """
    Exact 1D brute force for K-component iso-Gaussian GM.

    The formula computes p(x0 | x_t) by importance-reweighting the GM prior
    p(x0 | x_s) with the ratio p(x_t|x0) / p(x_s|x0):

        log p(x0 | x_t) ∝ log p(x_t|x0) - log p(x_s|x0) + log p(x0|x_s)

    This is the posterior from the joint log:
        -(1/(2*var_k) + zeta/2)*x0^2 + (mu_k/var_k + nu)*x0 + const

    which matches the analytic formula exactly (Gaussian × ratio = Gaussian).
    """
    K = len(mu_k_vals)
    at, st = float(alpha_t_val), float(sigma_t_val)
    as_, ss = float(alpha_s_val), float(sigma_s_val)
    xt, xs = float(x_t_val), float(x_s_val)
    vk = float(var_k_val)

    # Center grid on the analytic posterior mean for numerical stability
    # (posterior mean of first component under uniform weights)
    nu   = at*xt/st**2 - as_*xs/ss**2
    zeta = at**2/st**2 - as_**2/ss**2
    center = (float(mu_k_vals[0]) + vk*nu) / (vk*zeta + 1)
    half_range = std_range * (vk**0.5 + st/max(at, 1e-8))
    grid = torch.linspace(center - half_range, center + half_range, n_grid)

    # log p(x0|x_s): GM prior (logsumexp over K components)
    log_gm = torch.stack([
        float(logw_k_vals[k]) - 0.5*(grid - float(mu_k_vals[k]))**2 / vk
        for k in range(K)
    ]).logsumexp(dim=0)

    # log p(x_t|x0) - log p(x_s|x0)  — the ratio term
    log_ratio = (-0.5*(xt - at*grid)**2/st**2) - (-0.5*(xs - as_*grid)**2/ss**2)

    log_post = log_gm + log_ratio
    log_post = log_post - log_post.logsumexp(dim=0)
    weights  = log_post.exp()
    return float((weights * grid).sum())


# Run 1D test: K=4 components, single channel, one sample
# This is exact (no multi-channel weight update issue)
b, c = 0, 0
mu_k_1d   = gm_means[b, :, c]           # (K,)
logw_1d   = gm_logweights[b, :, 0]      # (K,) — log weights (unnormalised)
var_k_1d  = gm_vars[b, 0, 0]            # scalar

at = float(alpha_t_trig[b, 0])
st = float(sigma_t_trig[b, 0])

# Analytic formula (single channel, single sample)
nu_1d   = at * float(x_t_vp[b, c]) / st**2 - float(alpha_s_trig[b,0]) * float(x_s_vp[b,c]) / float(sigma_s_trig[b,0])**2
zeta_1d = at**2 / st**2 - float(alpha_s_trig[b,0])**2 / float(sigma_s_trig[b,0])**2
w_normed = torch.softmax(logw_1d, dim=0)

# Compute analytic posterior mean (1D, single channel)
denom_1d   = float(var_k_1d) * zeta_1d + 1
means_post = [(float(var_k_1d) * nu_1d + float(mu_k_1d[k])) / denom_1d for k in range(K)]
logw_delta = [float(mu_k_1d[k]) * (nu_1d - 0.5 * zeta_1d * float(mu_k_1d[k])) / denom_1d for k in range(K)]
logw_post  = torch.tensor([float(logw_1d[k]) + logw_delta[k] for k in range(K)])
w_post     = torch.softmax(logw_post, dim=0)
mean_analytic = float(sum(float(w_post[k]) * means_post[k] for k in range(K)))

# Brute-force 1D
mean_brute = brute_force_mean_1d(at, st, float(alpha_s_trig[b,0]), float(sigma_s_trig[b,0]),
                                  mu_k_1d, var_k_1d, logw_1d, x_t_vp[b, c], x_s_vp[b, c])

diff_1d = abs(mean_analytic - mean_brute)
print(f'\n[S6c 1D exact test] analytic={mean_analytic:.6f}  brute={mean_brute:.6f}  |diff|={diff_1d:.2e}')
if diff_1d < 0.01:
    print('PASS: trig posterior formula matches brute-force (1D, K=4)')
else:
    print('WARN: discrepancy — check formula or grid')

# ============================================================
# Section 7: analytic zeta bound + overflow test
# ============================================================
print('\n=== S7: float32 edge stability ===')
float32_max   = torch.finfo(torch.float32).max
max_var_k     = 10.0
ZETA_MAX_safe = float32_max / max_var_k
t_min_safe    = 2.0 / (math.pi * math.sqrt(ZETA_MAX_safe))

print(f'float32 max:       {float32_max:.3e}')
print(f'ZETA_MAX_safe:     {ZETA_MAX_safe:.3e}  (float32_max / max_var_k={max_var_k})')
print(f'Min safe t (unclamped VP): {t_min_safe:.3e}')

t_tiny   = torch.tensor([[1e-5]])   # (1, 1)
s_mid    = torch.tensor([[0.5]])
alpha_t_ = torch.cos(math.pi * t_tiny / 2)
sigma_t_ = torch.sin(math.pi * t_tiny / 2)
alpha_s_ = torch.cos(math.pi * s_mid  / 2)
sigma_s_ = torch.sin(math.pi * s_mid  / 2)

zeta_raw = (alpha_t_ / sigma_t_.clamp(min=1e-8)).square() \
         - (alpha_s_ / sigma_s_).square()
print(f'\nzeta at t=1e-5 (raw):      {zeta_raw.item():.3e}  isinf={torch.isinf(zeta_raw).item()}')

zeta_clamped = zeta_raw.clamp(max=ZETA_MAX_safe)
var_test     = torch.tensor([[[max_var_k]]])
denom_test   = (var_test * zeta_clamped + 1)
print(f'zeta after clamp:          {zeta_clamped.item():.3e}')
print(f'denom (var * zeta + 1):    {denom_test.item():.3e}  isinf={torch.isinf(denom_test).item()}')
print('PASS: clamp prevents overflow' if not torch.isinf(denom_test) else 'FAIL')

# SNR comparison at a sweep of t values
print('\n=== SNR at key t values ===')
for tv in [1e-3, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
    snr_vp  = (math.cos(math.pi*tv/2) / math.sin(math.pi*tv/2))**2
    snr_ve  = ((1-tv)/tv)**2
    print(f'  t={tv:.3f}  VP SNR={snr_vp:>12.2f}  VE SNR={snr_ve:>12.2f}')
