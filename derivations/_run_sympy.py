from sympy import *

alpha_t, sigma_t = symbols('alpha_t sigma_t', positive=True)
alpha_s, sigma_s = symbols('alpha_s sigma_s', positive=True)
x_t, x_s = symbols('x_t x_s', real=True)
mu_k, var_k = symbols('mu_k var_k', real=True, positive=True)

# Section 3: posterior terms
zeta = (alpha_t/sigma_t)**2 - (alpha_s/sigma_s)**2
nu   = alpha_t*x_t/sigma_t**2 - alpha_s*x_s/sigma_s**2
denom_k = var_k * zeta + 1
out_mean_k = (var_k * nu + mu_k) / denom_k
logweight_delta_k = mu_k * (nu - Rational(1,2) * zeta * mu_k) / denom_k

print('=== S3: posterior terms ===')
print('zeta       :', simplify(zeta))
print('nu         :', simplify(nu))
print('denom_k    :', simplify(denom_k))
print('out_mean_k :', simplify(out_mean_k))
print('logw_delta :', simplify(logweight_delta_k))

# Section 4: linear reduction
ls = {alpha_t: 1 - sigma_t, alpha_s: 1 - sigma_s}
zeta_lin = simplify(zeta.subs(ls))
nu_lin   = simplify(nu.subs(ls))
aos_t = (1 - sigma_t) / sigma_t
aos_s = (1 - sigma_s) / sigma_s
zeta_code = aos_t**2 - aos_s**2
nu_code   = aos_t * x_t / sigma_t - aos_s * x_s / sigma_s

print('\n=== S4: linear reduction ===')
print('zeta diff (should be 0):', simplify(zeta_lin - zeta_code))
print('nu   diff (should be 0):', simplify(nu_lin   - nu_code))

# Section 5: trig (VP)
t, s = symbols('t s', positive=True)
ts = {
    alpha_t: cos(pi*t/2), sigma_t: sin(pi*t/2),
    alpha_s: cos(pi*s/2), sigma_s: sin(pi*s/2),
}
zeta_trig = trigsimp(zeta.subs(ts))
vp_id     = simplify(cos(pi*t/2)**2 + sin(pi*t/2)**2 - 1)
mt        = out_mean_k.subs(ts)
lim_t0    = simplify(limit(mt.subs(s, Rational(1, 2)), t, 0, '+'))
lim_t1    = simplify(limit(mt.subs(s, Rational(1, 2)), t, 1, '-'))

print('\n=== S5: trig (VP) ===')
print('zeta_trig       :', zeta_trig)
print('VP id (0?)      :', vp_id)
print('lim t->0 (s=0.5):', lim_t0)
print('lim t->1 (s=0.5):', lim_t1)

# Section 8: LaTeX export
print('\n=== S8: LaTeX ===')
print('zeta :', latex(zeta))
print('nu   :', latex(nu))
print('denom:', latex(denom_k))
print('mean :', latex(out_mean_k))
print('logw :', latex(logweight_delta_k))
