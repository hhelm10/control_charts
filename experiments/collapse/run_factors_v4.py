"""Factor sweep for toy v4 (crowding retrieval): one rising lever per factor,
other factors at BASE. Same grid as v3's sweep; the mechanism is now relative
forgetting -- the q-entry is IDK when >= k_ctx fresher insertions outrank its
decayed score in the finite context (no threshold, no contradiction rule)."""
import json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from toy_v4 import run

SEEDS = range(10)
T = 800
LN2 = float(np.log(2.0))
BASE = dict(N=20, c_dec=LN2 / 20, chi=0.5, k_ctx=3, sigma="staleness_aware", B=5,  # agent
            M=50, lam_mean=0.003, lam_disp=1.0, s_env=0.0,                       # environment
            s_ask=0.0, mean_degree=None, peer="uniform",                         # communication
            alpha=0.01, n_obs_frac=1.0)                                          # observation
# With chi = 0.5 the similarity head start is Delta = ln2/c_dec = 20 steps at
# baseline, and tau_eff = Delta + k_ctx/r_ins ~ 21 -> R0 ~ 2, matching v3's clocks.

FACTORS = {
    # x gives Delta = {99, 50, 20, 10, 5, 2.5} at chi = 0.5
    "agent":         dict(x="c_dec", xs=[round(LN2 / t, 4) for t in (99, 50, 20, 10, 5, 2.5)],
                          color="N", colors=[5, 20, 100],
                          style="k_ctx", styles=[3, 7]),
    # color = edge density (degree 2 / 6 / full mesh); style = budget at FIXED
    # observation rate (alpha*B = 0.05). Policies fixed: unknown-first, random peer.
    "communication": dict(x="s_ask", xs=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
                          color="mean_degree", colors=[2, 6, None],
                          style="B_bw", styles=[5, 10]),
    # color = size of the world: M dilutes the birth rate (R0 = m*tau/M) and
    # stretches the per-question observation interval M/(alpha*B*N)
    "environment":   dict(x="lam_mean", xs=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                          color="M", colors=[25, 50, 100],
                          style="s_env", styles=[0.0, 1.0]),
    "observation":   dict(x="alpha", xs=[0.1, 0.03, 0.01, 0.003, 0.001],
                          color="B", colors=[2, 5, 10],
                          style="n_obs_frac", styles=[1.0, 0.5]),
}


def cell(kw):
    p = {k: v for k, v in kw.items() if k != "factor"}
    if "B_bw" in p:
        b = p.pop("B_bw"); p["B"] = b; p["alpha"] = 0.05 / b
    r = run(T=T, record_every=5, **p)
    last = slice(-40, None)
    return {"kw": kw, "factor": kw["factor"],
            **{k: float(r[k][last].mean()) for k in ("idk", "idk_demand", "correct", "stale", "dead_q")}}


def main():
    jobs = []
    for fname, f in FACTORS.items():
        for x in f["xs"]:
            for c in f["colors"]:
                for s in f["styles"]:
                    for seed in SEEDS:
                        kw = dict(BASE); kw.update({f["x"]: x, f["color"]: c, f["style"]: s, "seed": seed, "factor": fname})
                        jobs.append(kw)
    print(len(jobs), "runs", flush=True); t0 = time.time()
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(cell, jobs, chunksize=6))
    print(f"{time.time()-t0:.0f}s", flush=True)
    json.dump({"base": BASE, "factors": FACTORS, "results": res}, open("factor_v4_results.json", "w"))


if __name__ == "__main__":
    main()
