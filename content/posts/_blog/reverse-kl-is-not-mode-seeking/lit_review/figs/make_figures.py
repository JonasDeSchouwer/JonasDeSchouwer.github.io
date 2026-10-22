"""Recreate the canonical forward-vs-reverse-KL figures for lit_review.typ.

Fits are computed numerically (dense-grid quadrature + coarse-to-fine grid search
over the 2 Gaussian parameters), no closed forms assumed except where exact
(moment matching for forward KL; precision/covariance diagonals for Fig 2).

Run:  uv run -- python lit_review/figs/make_figures.py
Outputs SVGs next to this file.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = __file__.rsplit("/", 1)[0]
X = np.linspace(-10, 10, 8001)
DX = X[1] - X[0]

# Target: asymmetric two-Gaussian mixture. The left component is the HIGHER PEAK
# (narrow, weight 0.4); the right component carries the LARGER MASS (weight 0.6).
# This exhibits Minka's nuance: reverse KL prefers the largest-mass mode, not the
# highest peak.
W = (0.4, 0.6)
MU = (-2.5, 2.0)
SD = (0.45, 1.1)


def norm_pdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


P = W[0] * norm_pdf(X, *[MU[0], SD[0]]) + W[1] * norm_pdf(X, MU[1], SD[1])
P = np.maximum(P, 1e-300)
LOGP = np.log(P)


def q_pdf(mu, sd):
    return np.maximum(norm_pdf(X, mu, sd), 1e-300)


def kl_qp(mu, sd):  # reverse KL(q||p)
    q = q_pdf(mu, sd)
    return np.trapezoid(q * (np.log(q) - LOGP), dx=DX)


def kl_pq(mu, sd):  # forward KL(p||q)
    q = q_pdf(mu, sd)
    return np.trapezoid(P * (LOGP - np.log(q)), dx=DX)


def alpha_div(mu, sd, a):  # Zhu-Rohwer alpha-divergence D_a(p||q); a=0 -> KL(q||p), a=1 -> KL(p||q)
    if a == 0:
        return kl_qp(mu, sd)
    if a == 1:
        return kl_pq(mu, sd)
    q = q_pdf(mu, sd)
    expo = a * LOGP + (1 - a) * np.log(q)
    integral = np.trapezoid(np.exp(np.minimum(expo, 700)), dx=DX)
    return (1 - integral) / (a * (1 - a))


def js_pi(mu, sd, pi):  # Huszar generalized JS_pi(P||Q), pi->0 ~ forward KL, pi->1 ~ reverse KL
    q = q_pdf(mu, sd)
    m = np.maximum(pi * P + (1 - pi) * q, 1e-300)
    t1 = np.trapezoid(P * (LOGP - np.log(m)), dx=DX)
    t2 = np.trapezoid(q * (np.log(q) - np.log(m)), dx=DX)
    return pi * t1 + (1 - pi) * t2


def fit(objective, mu_range=(-6, 6), logsd_range=(np.log(0.05), np.log(5)), n=61, rounds=4):
    """Coarse-to-fine grid search over (mu, log sd)."""
    mu_lo, mu_hi = mu_range
    ls_lo, ls_hi = logsd_range
    best = None
    for _ in range(rounds):
        mus = np.linspace(mu_lo, mu_hi, n)
        lss = np.linspace(ls_lo, ls_hi, n)
        vals = np.array([[objective(m, np.exp(ls)) for ls in lss] for m in mus])
        i, j = np.unravel_index(np.argmin(vals), vals.shape)
        best = (mus[i], np.exp(lss[j]), vals[i, j])
        dmu = (mu_hi - mu_lo) / (n - 1) * 3
        dls = (ls_hi - ls_lo) / (n - 1) * 3
        mu_lo, mu_hi = mus[i] - dmu, mus[i] + dmu
        ls_lo, ls_hi = lss[j] - dls, lss[j] + dls
    return best


def moment_match():
    m = W[0] * MU[0] + W[1] * MU[1]
    v = sum(w * (sd**2 + (mu - m) ** 2) for w, mu, sd in zip(W, MU, SD))
    return m, np.sqrt(v)


STYLE = dict(lw=1.8)
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6})


# ---------------------------------------------------------------- Figure 1: 1-D
def fig1():
    mm, ms = moment_match()  # forward-KL optimum (exact)
    # reverse KL: two local minima, found from mode-local searches
    r1 = fit(kl_qp, mu_range=(MU[0] - 1.5, MU[0] + 1.5))
    r2 = fit(kl_qp, mu_range=(MU[1] - 1.5, MU[1] + 1.5))
    (glob, loc) = (r2, r1) if r2[2] <= r1[2] else (r1, r2)
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    ax.fill_between(X, P, color="0.82", label="$p$ (mixture)")
    ax.plot(X, q_pdf(mm, ms), color="tab:blue", **STYLE, label="forward KL fit")
    ax.plot(X, q_pdf(glob[0], glob[1]), color="tab:red", **STYLE, label="reverse KL fit (global)")
    ax.plot(X, q_pdf(loc[0], loc[1]), color="tab:red", ls="--", lw=1.4, label="reverse KL fit (local min.)")
    ax.set_xlim(-6.5, 7)
    ax.set_ylim(bottom=0)
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_fkl_rkl_1d.svg")
    print("fig1: FKL", (round(mm, 2), round(ms, 2)), "RKL glob", glob, "RKL loc", loc)


# ------------------------------------------------- Figure 2: 2-D factorized fit
def fig2():
    rho = 0.9
    Sig = np.array([[1.0, rho], [rho, 1.0]])
    Lam = np.linalg.inv(Sig)
    v_rev = 1.0 / np.diag(Lam)  # reverse-KL optimum for diagonal q (exact)
    v_fwd = np.diag(Sig)  # forward-KL optimum: moment matching (exact)

    def ellipse(cov, k):
        t = np.linspace(0, 2 * np.pi, 200)
        L = np.linalg.cholesky(cov)
        pts = (L @ np.vstack([np.cos(t), np.sin(t)])) * k
        return pts[0], pts[1]

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.8), sharex=True, sharey=True)
    for ax, v, title in zip(
        axes,
        [v_rev, v_fwd],
        [r"(a) reverse KL$(q\,\|\,p)$: compact", r"(b) forward KL$(p\,\|\,q)$: broad"],
    ):
        for k in (1.0, 2.0):
            ex, ey = ellipse(Sig, k)
            ax.plot(ex, ey, color="tab:blue", lw=1.3)
            qx, qy = ellipse(np.diag(v), k)
            ax.plot(qx, qy, color="tab:red", lw=1.3)
        ax.set_title(title, fontsize=8.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_factorized_2d.svg")
    print("fig2: v_rev", v_rev, "v_fwd", v_fwd)


# --------------------------------------------------- Figure 3: alpha-divergence
def fig3():
    alphas = [-5, 0, 0.5, 1, 5]
    labels = [r"$\alpha=-5$", r"$\alpha=0$ (reverse KL)", r"$\alpha=\frac{1}{2}$", r"$\alpha=1$ (forward KL)", r"$\alpha=5$"]
    fig, axes = plt.subplots(1, len(alphas), figsize=(9.6, 1.9), sharey=True)
    for ax, a, lab in zip(axes, alphas, labels):
        cands = [
            fit(lambda m, s: alpha_div(m, s, a), mu_range=(MU[0] - 1.5, MU[0] + 1.5)),
            fit(lambda m, s: alpha_div(m, s, a), mu_range=(MU[1] - 1.5, MU[1] + 1.5)),
            fit(lambda m, s: alpha_div(m, s, a)),
        ]
        mu, sd, _ = min(cands, key=lambda c: c[2])
        ax.fill_between(X, P, color="0.82")
        ax.plot(X, q_pdf(mu, sd), color="tab:red", lw=1.5)
        ax.set_title(lab, fontsize=8.5)
        ax.set_xlim(-6.5, 7)
        ax.set_ylim(bottom=0)
        ax.set_yticks([])
        print("fig3 alpha", a, "->", (round(mu, 2), round(sd, 2)))
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_alpha_sweep.svg")


# ------------------------------------------------------- Figure 4: JS_pi sweep
def fig4():
    pis = [0.1, 0.5, 0.99]
    fig, axes = plt.subplots(1, len(pis), figsize=(5.9, 1.9), sharey=True)
    for ax, pi in zip(axes, pis):
        cands = [
            fit(lambda m, s: js_pi(m, s, pi), mu_range=(MU[0] - 1.5, MU[0] + 1.5)),
            fit(lambda m, s: js_pi(m, s, pi), mu_range=(MU[1] - 1.5, MU[1] + 1.5)),
            fit(lambda m, s: js_pi(m, s, pi)),
        ]
        mu, sd, _ = min(cands, key=lambda c: c[2])
        ax.fill_between(X, P, color="0.82")
        ax.plot(X, q_pdf(mu, sd), color="tab:red", lw=1.5)
        ax.set_title(rf"$\pi={pi}$", fontsize=8.5)
        ax.set_xlim(-6.5, 7)
        ax.set_ylim(bottom=0)
        ax.set_yticks([])
        print("fig4 pi", pi, "->", (round(mu, 2), round(sd, 2)))
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_js_sweep.svg")


# -------------------------- Figure 5: MiniLLM-style three-peak mixture (Fig. 2)
def fig5():
    w, mu, sd = (0.1, 0.15, 0.75), (1.0, 4.0, 8.0), (0.4, 0.4, 0.8)
    x = np.linspace(-3, 13, 8001)
    dx = x[1] - x[0]
    p = np.maximum(sum(wi * norm_pdf(x, mi, si) for wi, mi, si in zip(w, mu, sd)), 1e-300)
    logp = np.log(p)

    def q(m, s):
        return np.maximum(norm_pdf(x, m, s), 1e-300)

    def rkl(m, s):
        qq = q(m, s)
        return np.trapezoid(qq * (np.log(qq) - logp), dx=dx)

    mm = sum(wi * mi for wi, mi in zip(w, mu))  # forward-KL optimum (moment match, exact)
    ms = np.sqrt(sum(wi * (si**2 + (mi - mm) ** 2) for wi, mi, si in zip(w, mu, sd)))
    glob = min([fit(rkl, mu_range=(m - 1.5, m + 1.5)) for m in mu], key=lambda c: c[2])
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    ax.fill_between(x, p, color="0.82", label="$p$ (three-peak mixture)")
    ax.plot(x, q(mm, ms), color="tab:blue", **STYLE, label="forward KL fit")
    ax.plot(x, q(glob[0], glob[1]), color="tab:red", **STYLE, label="reverse KL fit")
    ax.set_xlim(-1.5, 12.5)
    ax.set_ylim(bottom=0)
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_minillm_3peak.svg")
    print("fig5: FKL", (round(mm, 2), round(float(ms), 2)), "RKL", glob)


# -------- Figure 6: Bishop Fig. 10.3 — bimodal 2-D mixture fit by one Gaussian
def fit_cd(objective, x0, spans, rounds=6, n=41):
    """Cyclic coordinate-wise coarse-to-fine search."""
    x = list(x0)
    for _ in range(rounds):
        for i, sp in enumerate(spans):
            cand = np.linspace(x[i] - sp, x[i] + sp, n)
            vals = [objective(x[:i] + [float(c)] + x[i + 1 :]) for c in cand]
            x[i] = float(cand[int(np.argmin(vals))])
        spans = [sp / 2 for sp in spans]
    return x


def fig6():
    th = np.pi / 4  # components tilted along the diagonal, as in Bishop's figure
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    comp_cov = R @ np.diag([0.75**2, 0.45**2]) @ R.T
    mus = [np.array([-1.2, -1.2]), np.array([1.2, 1.2])]
    w = (0.5, 0.5)

    g = np.linspace(-4.2, 4.2, 337)
    XX, YY = np.meshgrid(g, g)
    pts = np.stack([XX, YY], axis=-1)
    da = (g[1] - g[0]) ** 2

    def gauss2(mean, cov):
        d = pts - mean
        L = np.linalg.inv(cov)
        quad = d[..., 0] ** 2 * L[0, 0] + 2 * d[..., 0] * d[..., 1] * L[0, 1] + d[..., 1] ** 2 * L[1, 1]
        return np.exp(-0.5 * quad) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))

    P2 = w[0] * gauss2(mus[0], comp_cov) + w[1] * gauss2(mus[1], comp_cov)
    logP2 = np.log(np.maximum(P2, 1e-300))

    def cov_of(ls1, ls2, t):
        s1, s2, r = np.exp(ls1), np.exp(ls2), np.tanh(t)
        return np.array([[s1 * s1, r * s1 * s2], [r * s1 * s2, s2 * s2]])

    def rkl(params):  # KL(q||p) = -H(q) - E_q[log p], H(q) closed-form
        C = cov_of(*params[2:])
        ent = 0.5 * np.log((2 * np.pi * np.e) ** 2 * np.linalg.det(C))
        return -ent - np.sum(gauss2(np.array(params[:2]), C) * logP2) * da

    mmean = w[0] * mus[0] + w[1] * mus[1]  # forward-KL optimum (moment match, exact)
    mcov = sum(wi * (comp_cov + np.outer(mi - mmean, mi - mmean)) for wi, mi in zip(w, mus))
    fits = [("(a) forward KL", mmean, mcov)]
    for lab, m0 in [("(b) reverse KL, local min. 1", mus[0]), ("(c) reverse KL, local min. 2", mus[1])]:
        best = fit_cd(rkl, [m0[0], m0[1], np.log(0.6), np.log(0.6), 0.0], spans=[0.8, 0.8, 0.7, 0.7, 0.8])
        fits.append((lab, np.array(best[:2]), cov_of(*best[2:])))
        print("fig6", lab, "-> mean", np.round(best[:2], 2), "cov", np.round(cov_of(*best[2:]), 2).tolist())

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.1), sharex=True, sharey=True)
    for ax, (title, m, C) in zip(axes, fits):
        Q = gauss2(m, C)
        ax.contour(XX, YY, P2, levels=np.array([0.06, 0.25, 0.55, 0.85]) * P2.max(), colors="tab:blue", linewidths=1.1)
        ax.contour(XX, YY, Q, levels=np.linspace(0.1, 0.9, 7) * Q.max(), colors="tab:red", linewidths=1.1, alpha=0.7)
        ax.set_title(title, fontsize=8.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-3.6, 3.6)
        ax.set_ylim(-3.6, 3.6)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_bishop_3panel.svg", bbox_inches="tight")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    print("done")
