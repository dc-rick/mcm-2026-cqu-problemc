"""T001: audit and re-estimate the bicycle fleet constant.

Reproduces 343,638 from 02_recon_spatial.py's Poisson-occupancy estimator,
then re-estimates the fleet with independent methods, each with a 95% interval
where the method admits one:

  A  observed lower bound (union of distinct bikeids)
  B  capture-recapture (Lincoln-Petersen / Chapman, Chao1) on two-period splits
  C  zero-truncated beta-binomial on days-seen, MLE + bootstrap
  D  the paper's homogeneous Poisson model corrected to the 9-day window
  E  (robustness) zero-truncated negative binomial on trips-per-bike

Writes data/runs/T001/fleet_estimate.json, methods.csv, fig_fleet_estimates.png.
Reads data/ only; never writes outside data/runs/T001/ and 交付/.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize, special

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import lib  # noqa: E402

ROOT = lib.ROOT
OUT = os.path.join(ROOT, "data", "runs", "T001")
os.makedirs(OUT, exist_ok=True)

DAYS9 = ["2017-05-%02d" % d for d in range(10, 20)]
present9 = [d for d in DAYS9 if d != "2017-05-17"]          # 5/17 has no records
F_OLD = 343638.0
rng = np.random.default_rng(20261117)


def occ_fleet(T, D):
    """Solve mu/(1-exp(-mu)) = T/D; return (mu, F) for the homogeneous model."""
    r = T / D
    mu = optimize.brentq(lambda m: m / (1 - np.exp(-m)) - r, 1e-9, 50)
    return mu, T / mu


# ---------------------------------------------------------------- data
df = lib.load_trips(None)                       # Beijing bbox filter
df9 = df[df.date.isin(DAYS9)]
D9 = df9.bikeid.nunique()
D14 = df.bikeid.nunique()
T9 = len(df9)
print(f"lib filter: all days rows={len(df)} bikes={D14}; 9 days rows={T9} bikes={D9}")

# per-day table under lib filter
day_rows = []
for d, g in df.groupby("date"):
    mu, F = occ_fleet(len(g), g.bikeid.nunique())
    day_rows.append(dict(date=str(d), trips=int(len(g)), used=int(g.bikeid.nunique()),
                         T_over_D=len(g) / g.bikeid.nunique(), mu=mu, fleet=F))

# ---------------------------------------------------------------- reproduce
# 02_recon_spatial.py uses a cruder filter: drop only lat<35 or lon<110.
raw = pd.read_parquet(os.path.join(lib.DATA, "trips.parquet"))
crude = raw[(raw.start_lat > 35) & (raw.start_lon > 110) &
            (raw.end_lat > 35) & (raw.end_lon > 110)].copy()
crude["date"] = crude.starttime.dt.normalize().astype(str).str[:10]
g16 = crude[crude.date == "2017-05-16"]
T16, D16 = len(g16), g16.bikeid.nunique()
mu16, F16 = occ_fleet(T16, D16)
g16lib = df[df.date == "2017-05-16"]
mu16l, F16l = occ_fleet(len(g16lib), g16lib.bikeid.nunique())

# the 12 extra records the crude filter keeps
key = "orderid" if "orderid" in raw.columns else None
extra = g16[~g16.orderid.isin(g16lib.orderid)] if key else None
repro = {
    "day": "2017-05-16", "filter": "start/end lat>35, lon>110 (02_recon_spatial.py)",
    "trips": int(T16), "used_bikes": int(D16), "T_over_D": float(T16 / D16),
    "mu": float(mu16), "fleet": float(F16), "fleet_rounded": int(round(F16)),
    "matches_343638": bool(abs(F16 - F_OLD) < 1.0),
    "delta_vs_constant": float(F16 - F_OLD),
    "lib_filter_same_day": {"trips": int(len(g16lib)),
                            "used_bikes": int(g16lib.bikeid.nunique()),
                            "fleet": float(F16l),
                            "delta_vs_crude": float(F16l - F16)},
    "extra_records_kept_by_crude_filter": int(T16 - len(g16lib)),
    "extra_distinct_bikes": int(D16 - g16lib.bikeid.nunique()),
    "extra_start_lat_range": [float(extra.start_lat.min()), float(extra.start_lat.max())]
    if extra is not None and len(extra) else None,
    "extra_start_lon_range": [float(extra.start_lon.min()), float(extra.start_lon.max())]
    if extra is not None and len(extra) else None,
    "note": ("343,638 is the single-day 5/16 Poisson-occupancy estimate under "
             "02_recon_spatial.py's crude filter; the lib filter gives 343,604.8 "
             "(difference fully accounted for by the extra records)."),
}
print("repro:", json.dumps(repro, indent=1)[:600])

# ---------------------------------------------------------------- A lower bound
method_rows = []
method_rows.append(dict(method="A: observed union (9 days)", point=D9,
                        lo=D9, hi=None,
                        note="analytic lower bound; no interval exists"))
method_rows.append(dict(method="A: observed union (14 days)", point=D14,
                        lo=D14, hi=None,
                        note="analytic lower bound; no interval exists"))

# ---------------------------------------------------------------- B capture-recapture
def lincoln_petersen_set(s1, s2):
    n1, n2 = len(s1), len(s2)
    m = len(np.intersect1d(s1, s2))
    lp = n1 * n2 / m
    chap = (n1 + 1) * (n2 + 1) / (m + 1) - 1
    var = (n1 + 1) * (n2 + 1) * (n1 - m) * (n2 - m) / ((m + 1) ** 2 * (m + 2))
    se = np.sqrt(var)
    return dict(n1=n1, n2=n2, both=m, lp=lp, chapman=chap, se=se,
                lo=chap - 1.96 * se, hi=chap + 1.96 * se)


def chao1_set(s1, s2, D):
    n1, n2 = len(s1), len(s2)
    m = len(np.intersect1d(s1, s2))
    f1 = n1 + n2 - 2 * m          # seen in exactly one of the two samples
    f2 = m                        # seen in both
    extra = 0.0 if f2 == 0 else f1 ** 2 / (2 * f2)
    N = D + extra
    if f2 == 0:
        return dict(f1=f1, f2=f2, extra=extra, N=N, lo=D, hi=None)
    var = f2 * (0.5 * (f1 / f2) ** 2 + (f1 / f2) ** 3 + 0.25 * (f1 / f2) ** 4)
    if extra > 0:
        se_log = np.sqrt(np.log(1 + var / extra ** 2))
        lo = D + extra * np.exp(-1.96 * se_log)
        hi = D + extra * np.exp(1.96 * se_log)
    else:
        lo, hi = D, None
    return dict(f1=f1, f2=f2, extra=extra, N=N, var=var, lo=lo, hi=hi)


def set_of(days):
    return df9[df9.date.isin(days)].bikeid.unique()


splits = {
    "first5_last4": (present9[:5], present9[5:]),
    "odd_even": (present9[0::2], present9[1::2]),
    "first3_last6": (present9[:3], present9[3:]),
}
cr_rows = []
for name, (dA, dB) in splits.items():
    A, B = set_of(dA), set_of(dB)
    lp = lincoln_petersen_set(A, B)
    ch = chao1_set(A, B, D9)
    row = dict(split=name, daysA=";".join(dA), daysB=";".join(dB))
    row.update({f"lp_{k}": v for k, v in lp.items()})
    row.update({f"chao_{k}": v for k, v in ch.items()})
    cr_rows.append(row)
    method_rows.append(dict(method=f"B: Chapman ({name})", point=lp["chapman"],
                            lo=lp["lo"], hi=lp["hi"],
                            note=f"n1={lp['n1']:.0f}, n2={lp['n2']:.0f}, m={lp['both']:.0f}"))
    method_rows.append(dict(method=f"B: Chao1 ({name})", point=ch["N"],
                            lo=ch["lo"], hi=ch["hi"],
                            note=f"f1={ch['f1']:.0f}, f2={ch['f2']:.0f}"))

# ---------------------------------------------------------------- C beta-binomial
days_per_bike = df9.groupby("bikeid").date.nunique()
nb_k = 9
cnt = np.bincount(days_per_bike.values, minlength=nb_k + 1)[1:nb_k + 1].astype(float)
Nobs = cnt.sum()


def bb_pmf(k, a, b):
    return np.exp(special.betaln(k + a, nb_k - k + b) - special.betaln(a, b)
                  + special.gammaln(nb_k + 1) - special.gammaln(k + 1)
                  - special.gammaln(nb_k - k + 1))


def bb_loglik(par):
    a, b = np.exp(par)
    p = bb_pmf(np.arange(1, nb_k + 1), a, b)
    p0 = bb_pmf(0, a, b)
    p = np.clip(p, 1e-300, None)
    return -np.sum(cnt * np.log(p)) + Nobs * np.log(max(1 - p0, 1e-300))


def fit_bb(counts=cnt, n=Nobs):
    start = np.log([2.0, 5.0])
    res = optimize.minimize(bb_loglik, start, method="Nelder-Mead",
                            options=dict(maxiter=4000, xatol=1e-8, fatol=1e-10))
    a, b = np.exp(res.x)
    p0 = float(bb_pmf(0, a, b))
    return dict(a=a, b=b, p0=p0, N=n / (1 - p0), nll=res.fun, ok=res.success)


bb = fit_bb()
bb_boot = []
for _ in range(600):
    c = rng.multinomial(Nobs, cnt / Nobs).astype(float)
    # refit with resampled histogram (same total)
    def ll(par):
        a, b = np.exp(par)
        p = np.clip(bb_pmf(np.arange(1, nb_k + 1), a, b), 1e-300, None)
        p0 = bb_pmf(0, a, b)
        return -np.sum(c * np.log(p)) + c.sum() * np.log(max(1 - p0, 1e-300))
    r = optimize.minimize(ll, np.log([bb["a"], bb["b"]]), method="Nelder-Mead",
                          options=dict(maxiter=2000))
    a, b = np.exp(r.x)
    bb_boot.append(Nobs / (1 - bb_pmf(0, a, b)))
bb_lo, bb_hi = np.percentile(bb_boot, [2.5, 97.5])
method_rows.append(dict(method="C: zero-truncated beta-binomial (days-seen)",
                        point=bb["N"], lo=float(bb_lo), hi=float(bb_hi),
                        note=f"a={bb['a']:.3f}, b={bb['b']:.3f}, P0={bb['p0']:.5f}"))

# goodness of fit of the beta-binomial and of the paper's homogeneous model
p_obs = cnt / Nobs
p_fit = bb_pmf(np.arange(1, nb_k + 1), bb["a"], bb["b"]) / (1 - bb["p0"])
# homogeneous one-day Poisson with mu from the paper (0.8401), extended to 9 days
p_day_paper = 1 - np.exp(-mu16)
p_k_paper = np.array([special.comb(nb_k, k) * p_day_paper ** k
                      * (1 - p_day_paper) ** (nb_k - k)
                      for k in range(1, nb_k + 1)]) / (1 - (1 - p_day_paper) ** nb_k)
# homogeneous model corrected to the 9-day window:
# the solved parameter IS m = 9*mu (expected sampled trips per bike over 9 days)
m9, F9_hom = occ_fleet(T9, D9)
mu_day9 = m9 / nb_k
p_day9 = 1 - np.exp(-mu_day9)
p_k_hom9 = np.array([special.comb(nb_k, k) * p_day9 ** k
                     * (1 - p_day9) ** (nb_k - k) for k in range(1, nb_k + 1)]) / (1 - (1 - p_day9) ** nb_k)
gof = [dict(k=int(k + 1), observed=float(p_obs[k]), beta_binomial=float(p_fit[k]),
            paper_homog_1day=float(p_k_paper[k]), homog_9day=float(p_k_hom9[k]))
       for k in range(nb_k)]
print("GOF frac seen exactly 1 day: obs %.4f, bb %.4f, paper-1day %.4f, homog-9day %.4f"
      % (p_obs[0], p_fit[0], p_k_paper[0], p_k_hom9[0]))

method_rows.append(dict(method="D: homogeneous Poisson corrected to 9 days",
                        point=F9_hom, lo=D9, hi=None,
                        note=(f"same estimator as the paper, window=9 days; "
                              f"solves m/(1-e^-m)={T9/D9:.4f} -> m={m9:.3f}; "
                              f"lower bound D9 by construction; heterogeneity rejected by GOF")))

# ---------------------------------------------------------------- E zero-truncated NB
trips_per_bike = df9.groupby("bikeid").orderid.size()
nmax = int(trips_per_bike.max())
c_t = np.bincount(trips_per_bike.values, minlength=nmax + 1)[1:].astype(float)
Nt = c_t.sum()
kk = np.arange(1, nmax + 1)


def nb_loglik(par, counts=c_t):
    logr, logitq = par
    r = np.exp(logr)
    q = 1 / (1 + np.exp(-logitq))
    logp = (special.gammaln(kk + r) - special.gammaln(r) - special.gammaln(kk + 1)
            + r * np.log(q) + kk * np.log1p(-q))
    c = np.clip(np.exp(logp), 1e-300, None)
    p0 = q ** r
    return -np.sum(counts * np.log(c)) + counts.sum() * np.log(max(1 - p0, 1e-300))


res_nb = optimize.minimize(nb_loglik, np.array([np.log(0.5), np.log(0.1 / 0.9)]),
                           method="Nelder-Mead", options=dict(maxiter=4000))
print("NB fit:", res_nb.success, "r", round(np.exp(res_nb.x[0]), 4),
      "q", round(1 / (1 + np.exp(-res_nb.x[1])), 5))
r_nb = np.exp(res_nb.x[0]); q_nb = 1 / (1 + np.exp(-res_nb.x[1]))
p0_nb = q_nb ** r_nb
N_nb = Nt / (1 - p0_nb)
nb_boot = []
for _ in range(300):
    c = rng.multinomial(Nt, c_t / Nt).astype(float)
    rr = optimize.minimize(nb_loglik, res_nb.x, args=(c,), method="Nelder-Mead",
                           options=dict(maxiter=2000))
    r2 = np.exp(rr.x[0]); q2 = 1 / (1 + np.exp(-rr.x[1]))
    nb_boot.append(Nt / (1 - q2 ** r2))
nb_lo, nb_hi = np.percentile(nb_boot, [2.5, 97.5])
method_rows.append(dict(method="E: zero-truncated NB (trips-per-bike)",
                        point=N_nb, lo=float(nb_lo), hi=float(nb_hi),
                        note=f"r={r_nb:.3f}, P0={p0_nb:.5f}"))

# ---------------------------------------------------------------- summary
summary = dict(
    old_constant=F_OLD,
    observed_lower_bound_9d=int(D9),
    observed_lower_bound_14d=int(D14),
    old_below_lower_bound=bool(F_OLD < D9),
    ratio_lower_bound_over_old=float(D9 / F_OLD),
    methods=[dict(m) for m in method_rows],
    gof_days_seen=gof,
    beta_binomial=dict(a=float(bb["a"]), b=float(bb["b"]), p0=float(bb["p0"]),
                       N=float(bb["N"]), boot_lo=float(bb_lo), boot_hi=float(bb_hi),
                       boot_mean=float(np.mean(bb_boot)), boot_sd=float(np.std(bb_boot))),
    homogeneous_9day=dict(m=float(m9), mu_day=float(mu_day9), fleet=float(F9_hom)),
    nb=dict(r=float(r_nb), q=float(q_nb), p0=float(p0_nb), N=float(N_nb)),
    capture_recapture=cr_rows,
    reproduction=repro,
    per_day=day_rows,
)
with open(os.path.join(OUT, "fleet_estimate.json"), "w", encoding="utf-8") as fh:
    json.dump(summary, fh, ensure_ascii=False, indent=1, default=float)

pd.DataFrame(method_rows).to_csv(os.path.join(OUT, "methods.csv"),
                                 index=False, encoding="utf-8-sig")
pd.DataFrame(day_rows).to_csv(os.path.join(OUT, "per_day_estimates.csv"),
                              index=False, encoding="utf-8-sig")
pd.DataFrame(cr_rows).to_csv(os.path.join(OUT, "capture_recapture.csv"),
                             index=False, encoding="utf-8-sig")
pd.DataFrame(gof).to_csv(os.path.join(OUT, "gof_days_seen.csv"),
                         index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- figure
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

lib.style()
fig, ax = plt.subplots(figsize=(6.4, 3.6))
pts = [m for m in method_rows if m["method"].startswith(("B", "C", "E"))]
labels = [m["method"].replace("B: ", "").replace("C: ", "").replace("E: ", "")
          for m in pts]
x = np.arange(len(pts))
y = np.array([m["point"] for m in pts]) / 1e3
lo = np.array([m["lo"] for m in pts]) / 1e3
hi = np.array([m["hi"] for m in pts]) / 1e3
ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o", ms=5, capsize=3,
            color=lib.COL["blue"], lw=1.2)
ax.axhline(D9 / 1e3, color=lib.COL["red"], lw=1.0, ls="--",
           label=f"observed lower bound, 9 d = {D9:,}")
ax.axhline(F_OLD / 1e3, color=lib.COL["grey"], lw=1.0, ls=":",
           label="paper constant = 343,638")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
ax.set_ylabel("fleet estimate ($10^3$ bicycles)")
ax.legend(fontsize=7, loc="upper left")
ax.set_title("Fleet re-estimates with 95% intervals", loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_fleet_estimates.png"), dpi=300,
            bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

print("\nmethods:")
for m in method_rows:
    print(f"  {m['method']:48s} {m['point']:10,.0f}  [{m['lo'] if m['lo'] else '':>9} , "
          f"{m['hi'] if m['hi'] else '':>9}]  {m['note']}")
print("saved", OUT)
