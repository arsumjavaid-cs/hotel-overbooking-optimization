#!/usr/bin/env python3
"""
Overbooking pipeline v4: EUR 300 is paid ONCE per walked guest (per stay), not once per night.

Keep v2, v3 and v4 in the same folder (v4 reuses them).

  tune   choose settings on Jul-Dec 2016 under per-stay accounting, for two models: A (v2) and C (v3 seasonal)
  score  score on Jan-Aug 2017 (exploratory: that window has been used before)
  table  morning table for an as-of date

How walks are accounted (earlier versions charged EUR 300 per excess room per night):
  * Hotels walk guests at CHECK-IN. A guest already in the house cannot be walked.
  * A walked guest leaves the hotel for the WHOLE stay, so their later nights free up. That is simulated
    night by night: walks on night d = excess on night d, less the walked guests from earlier nights still
    in the house (using the stay-length distribution of check-ins), limited to that night's check-ins.
  * Freed later nights that did not need freeing become empty rooms (an extra cost of walking).
  * EUR 300 is charged once per walked guest. The walk rate is walked guests per 1,000 CHECK-INS.
  * Extra bookings are still assumed to behave like the night's other bookings (as before). On nights that
    are not tight (bookings < 80% of rooms) nothing changes from what actually happened.

Decision rule: same expected-cost rule as v2, but the walk cost inside it is a scale to be chosen on
validation. Per-night equivalent of EUR 300 per stay is 300 / (average stay), about EUR 90, and multiples of
that (x1, 1.5, 2, 3) and the old EUR 300 are tried, because walking a guest also wastes their later nights.
"""
import argparse, importlib.util, json, os, pickle, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("v3", os.path.join(HERE, "overbooking_pipeline_v3.py"))
v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3)
v2 = v3.v2
HOTELS, N_H = v2.HOTELS, v2.N_H
log = v2.log
COMP = 300.0
WM = ["1", "1.5", "2", "3", "300"]          # walk-cost scale: multiple of 300/avg stay, or "300" = the old per-night EUR 300
LAMS = [1.5, 2.0, 3.0]
GS = [None, 1.0, 0.5, 0.35, 0.25, 0.15]
WINDOWS = [None, 56, 91]
VAL_TARGET = v2.VAL_TARGET


# ----------------------------------------------------------------------------- stay structure
def stay_survival(df, J=45):
    ok = df[(df.arr < v2.TRAIN_END) & (df.y == 0) & (df.n > 0)]
    out = {}
    for h in HOTELS:
        n = ok[ok.hotel == h].n.values
        out[h] = np.array([1.0] + [float((n > j).mean()) for j in range(1, J)])
    return out


def checkins_series(df):
    return df[(df.y == 0) & (df.n > 0)].groupby(["hotel", "arr"]).size()


def per_stay(nf, O, surv, chk):
    """Night-by-night walk simulation for one rule (O = rooms oversold per row). Returns one row per (hotel, h, night)."""
    d = nf[["hotel", "night", "h", "cap", "adr", "B", "C", "f_real", "tight"]].copy()
    d["O"] = np.asarray(O)
    act = (d.B - d.C).values
    d["arr"] = np.where(d.tight, (d.cap + d.O) * (1 - d.f_real), act)
    d["scale"] = np.where(act > 0, d.arr / np.maximum(act, 1e-9), 1.0)
    out = []
    for (hot, h), g in d.groupby(["hotel", "h"]):
        g = g.sort_values("night")
        days = pd.date_range(g.night.min(), g.night.max())
        g = g.set_index("night").reindex(days)
        T = len(g)
        present = g.B.notna().values
        cap = g.cap.ffill().bfill().values
        arr = g.arr.fillna(0).values
        scale = g.scale.fillna(1).values
        c_act = chk.reindex(pd.MultiIndex.from_product([[hot], days])).fillna(0).values
        chk_pol = c_act * scale
        s = surv[hot]; J = len(s)
        W = np.zeros(T); occ = np.zeros(T)
        for t in range(T):
            lo = max(0, t - J + 1)
            overlap = float(np.dot(W[lo:t][::-1], s[1:1 + (t - lo)])) if t > lo else 0.0
            need = max(0.0, max(0.0, arr[t] - cap[t]) - overlap)
            W[t] = min(need, chk_pol[t])
            occ[t] = arr[t] - overlap - W[t]
        out.append(pd.DataFrame({"hotel": hot, "h": h, "night": days[present], "tight": g.tight.fillna(False).values[present].astype(bool),
                                 "adr": g.adr.values[present], "W": W[present], "chk_pol": chk_pol[present], "occ": occ[present]}))
    return pd.concat(out, ignore_index=True)


def merge_none(res, none):
    return res.merge(none[["hotel", "h", "night", "occ"]], on=["hotel", "h", "night"], suffixes=("", "_none"))


def summarize(res, none):
    m = merge_none(res, none)
    gross = float((m.adr * (m.occ - m.occ_none)).sum() / N_H)
    walked = float(m.W.sum() / N_H)
    t = m[m.tight]
    r = {h: 1000 * t[t.hotel == h].W.sum() / max(t[t.hotel == h].chk_pol.sum(), 1e-9) for h in HOTELS}
    return dict(net=gross - COMP * walked, gross=gross, comp=COMP * walked, walked=walked,
                extra_room_nights=float((m.occ - m.occ_none).sum() / N_H),
                rate_both=1000 * t.W.sum() / max(t.chk_pol.sum(), 1e-9), rate_City=r["City Hotel"], rate_Resort=r["Resort Hotel"])


def worst_month(res):
    t = res[res.tight].copy()
    t["m"] = t.night.dt.to_period("M")
    out = {}
    for h in HOTELS:
        x = t[t.hotel == h].groupby("m").apply(lambda g: 1000 * g.W.sum() / max(g.chk_pol.sum(), 1e-9))
        out[h] = float(x.max())
    return out


def net_by_night(res, none):
    m = merge_none(res, none)
    m["net"] = m.adr * (m.occ - m.occ_none) - COMP * m.W
    return m.groupby(["hotel", "night"]).net.sum() / N_H


# ----------------------------------------------------------------------------- frames
def prepare(df, bundle, split):
    v3.activate(bundle)
    model, cats, fs = bundle["model"], bundle["cats"], bundle["fs"]
    if split == "val":
        S = df[(df.n > 0) & (df.arr >= pd.Timestamp("2016-05-01")) & (df.arr < v2.VAL_END)].reset_index(drop=True)
        src = df[(df.arr >= v2.TRAIN_END) & (df.arr < v2.VAL_END)]
        dates, d0, d1 = pd.date_range("2016-06-15", "2016-12-31"), v2.VAL_START, pd.Timestamp("2016-12-31")
    else:
        S = df[(df.n > 0) & (df.arr >= pd.Timestamp("2016-11-01")) & (df.arr <= v2.TEST_END)].reset_index(drop=True)
        src = df[(df.arr >= v2.TRAIN_END) & (df.arr <= v2.TEST_END)]
        dates, d0, d1 = pd.date_range("2016-11-01", "2017-08-31"), v2.TEST_START, v2.TEST_END
    L, Praw = v2.raw_matrix(model, S, cats, fs)
    return dict(bundle=bundle, S=S, L=L, Praw=Praw, rr=v2.raw_rows(model, src, cats, fs), dates=dates, d0=d0, d1=d1, shifts={})


def frame(prep, W):
    v3.activate(prep["bundle"])
    if W not in prep["shifts"]:
        prep["shifts"][W] = v2.rolling_shift(prep["rr"], W, prep["dates"]) if W else None
    P = v2.apply_shift(prep["Praw"], prep["L"], prep["S"], prep["shifts"][W])
    nf = v2.attach_context(v2.night_frame(prep["S"], prep["L"], P, prep["d0"], prep["d1"]), prep["bundle"])
    return v2.add_baselines(nf, prep["bundle"])


def oversell_by_hotel(nf, lam, g, weff):
    O = np.zeros(len(nf), int)
    for h in HOTELS:
        m = (nf.hotel == h).values
        v2.WALK_COST = weff[h]
        O[m] = v2.oversell(nf.f_hat.values[m], nf.v_hat.values[m], nf.cap.values[m], nf.adr.values[m], lam, g)
    v2.WALK_COST = 300.0
    return O


def weff_of(wm, nbar):
    return {h: (300.0 if wm == "300" else 300.0 * float(wm) / nbar[h]) for h in HOTELS}


# ----------------------------------------------------------------------------- tune
def stage_tune(a):
    df = v2.load(a.data)
    surv, chk = stay_survival(df), checkins_series(df)
    nbar = {h: float(surv[h].sum()) for h in HOTELS}
    log("average stay of check-ins (nights):", {h: round(v, 2) for h, v in nbar.items()})
    models = {"A": os.path.join(a.refA, "bundle_v2.pkl"), "C": os.path.join(a.refC, "variant_C", "bundle_v2.pkl")}
    summary = {}
    for name, path in models.items():
        bundle = pickle.load(open(path, "rb"))
        prep = prepare(df, bundle, "val")
        rows = []
        for W in WINDOWS:
            nf = frame(prep, W)
            none = per_stay(nf, np.zeros(len(nf), int), surv, chk)
            old = per_stay(nf, v2.oversell(nf.f_hat.values, nf.v_hat.values, nf.cap.values, nf.adr.values, bundle["lam"], bundle["g"]), surv, chk) \
                if W == bundle["W"] else None
            if old is not None:
                s = summarize(old, none)
                log(f"[{name}] OLD settings re-scored per stay on validation: net gain vs none EUR {s['net']:,.0f}, walked {s['walked']:.0f}, per 1,000 check-ins City {s['rate_City']:.1f} / Resort {s['rate_Resort']:.1f}")
            for wm in WM:
                weff = weff_of(wm, nbar)
                for lam in LAMS:
                    for g in GS:
                        res = per_stay(nf, oversell_by_hotel(nf, lam, g, weff), surv, chk)
                        rows.append(dict(W=W, wm=wm, lam=lam, g=g, **summarize(res, none)))
        G = pd.DataFrame(rows)
        G["max_rate"] = G[["rate_City", "rate_Resort"]].max(axis=1)
        G.to_csv(os.path.join(a.outdir, f"tune_grid_{name}.csv"), index=False) if os.path.isdir(a.outdir) else None
        feas = G[G.max_rate <= VAL_TARGET]
        con = feas[feas.g.notna()].sort_values("net", ascending=False).iloc[0]
        risk = feas[feas.g.isna()].sort_values("net", ascending=False).iloc[0] if feas.g.isna().any() else G[G.g.isna()].sort_values("net", ascending=False).iloc[0]
        log(f"[{name}] best CONSTRAINED on validation: net EUR {con.net:,.0f} | walk-cost scale {con.wm}, lam {con.lam}, g {con.g}, window {con.W} | per 1,000 {con.rate_City:.1f}/{con.rate_Resort:.1f}")
        log(f"[{name}] best RISK-BASED (no g) on validation: net EUR {risk.net:,.0f} | scale {risk.wm}, lam {risk.lam}, window {risk.W} | per 1,000 {risk.rate_City:.1f}/{risk.rate_Resort:.1f}")
        bw = lambda r, g_: dict(bundle, mode=("none" if pd.isna(r.W) else f"rolling {int(r.W)}d"), W=(None if pd.isna(r.W) else int(r.W)),
                                lam=float(r.lam), g=g_, weff=weff_of(r.wm, nbar), wm=r.wm)
        out = {"constrained": bw(con, float(con.g)), "risk": bw(risk, None)}
        sub = os.path.join(a.outdir, name)
        os.makedirs(sub, exist_ok=True)
        for k, b in out.items():
            os.makedirs(os.path.join(sub, k), exist_ok=True)
            pickle.dump(b, open(os.path.join(sub, k, "bundle_v2.pkl"), "wb"))
        G.to_csv(os.path.join(sub, "tune_grid.csv"), index=False)
        summary[name] = dict(constrained=dict(net=float(con.net), wm=con.wm, lam=float(con.lam), g=float(con.g), W=None if pd.isna(con.W) else int(con.W)),
                             risk=dict(net=float(risk.net), wm=risk.wm, lam=float(risk.lam), W=None if pd.isna(risk.W) else int(risk.W)))
    chosen = max(summary, key=lambda k: summary[k]["constrained"]["net"])
    summary["chosen_model"] = chosen
    log("\nCHOSEN on validation:", chosen, summary[chosen]["constrained"])
    json.dump(dict(summary, nbar=nbar), open(os.path.join(a.outdir, "tune_summary.json"), "w"), indent=1, default=float)


# ----------------------------------------------------------------------------- score
def stage_score(a):
    df = v2.load(a.data)
    surv, chk = stay_survival(df), checkins_series(df)
    rng = np.random.default_rng(0)
    allout = {}
    for name in ["A", "C"]:
        old_path = os.path.join(a.refA, "bundle_v2.pkl") if name == "A" else os.path.join(a.refC, "variant_C", "bundle_v2.pkl")
        old = pickle.load(open(old_path, "rb"))
        con = pickle.load(open(os.path.join(a.outdir, name, "constrained", "bundle_v2.pkl"), "rb"))
        risk = pickle.load(open(os.path.join(a.outdir, name, "risk", "bundle_v2.pkl"), "rb"))
        prep = prepare(df, old, "test")
        cache = {}

        def fr(W):
            if W not in cache:
                cache[W] = frame(prep, W)
            return cache[W]
        base = fr(old["W"])
        none = per_stay(base, np.zeros(len(base), int), surv, chk)
        rules = {}
        rules["Average rate, as written"] = per_stay(base, base.O_flat.values, surv, chk)
        rules["Average rate, horizon-aware"] = per_stay(base, base.O_flat_h.values, surv, chk)
        rules["No overselling"] = none
        rules["Earlier rule (EUR 300 per night), re-scored"] = per_stay(base, v2.oversell(base.f_hat.values, base.v_hat.values, base.cap.values, base.adr.values, old["lam"], old["g"]), surv, chk)
        for lab, b in [("Risk-based, re-tuned per stay", risk), ("Service-level, re-tuned per stay", con)]:
            nfx = fr(b["W"])
            if b["W"] != old["W"]:
                nonex = per_stay(nfx, np.zeros(len(nfx), int), surv, chk)
            else:
                nonex = none
            rules[lab] = (per_stay(nfx, oversell_by_hotel(nfx, b["lam"], b["g"], b["weff"]), surv, chk), nonex)
        rows, nights = [], {}
        for lab, r in rules.items():
            res, nn = (r if isinstance(r, tuple) else (r, none))
            s = summarize(res, nn)
            wmth = worst_month(res)
            rows.append(dict(policy=lab, extra_room_nights=s["extra_room_nights"], extra_gross_revenue=s["gross"], walk_compensation=s["comp"],
                             net_gain_vs_no_overselling=s["net"], guests_walked=s["walked"], walks_per_1000_checkins_both=s["rate_both"],
                             City=s["rate_City"], Resort=s["rate_Resort"], worst_month_City=wmth["City Hotel"], worst_month_Resort=wmth["Resort Hotel"]))
            nights[lab] = net_by_night(res, nn)
        T = pd.DataFrame(rows)
        pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
        log(f"\n===== model {name}: Jan-Aug 2017, per-stay accounting (EUR 300 once per walked guest) =====")
        log(T.round(1).to_string(index=False))
        wk = pd.DataFrame(nights).reset_index()
        wk["w"] = wk.night.dt.isocalendar().week.astype(int) + 100 * wk.night.dt.year
        wk = wk.groupby("w")[list(nights)].sum()
        def ci(x, y):
            b = [(lambda s: s[x] - s[y])(wk.iloc[rng.integers(0, len(wk), len(wk))].sum()) for _ in range(4000)]
            return float((wk[x] - wk[y]).sum()), np.percentile(b, [2.5, 97.5]).round(0).tolist(), float((wk[x] > wk[y]).mean())
        cmp_ = {"service-level re-tuned vs earlier rule": ci("Service-level, re-tuned per stay", "Earlier rule (EUR 300 per night), re-scored"),
                "service-level vs risk-based (both re-tuned)": ci("Service-level, re-tuned per stay", "Risk-based, re-tuned per stay"),
                "service-level re-tuned vs horizon-aware average": ci("Service-level, re-tuned per stay", "Average rate, horizon-aware"),
                "service-level re-tuned vs no overselling": ci("Service-level, re-tuned per stay", "No overselling")}
        for k, (d, c, sh) in cmp_.items():
            log(f"  {k}: EUR {d:,.0f}  95% CI {c}  weeks better: {sh:.2f}")
        allout[name] = dict(table=T.to_dict("records"), comparisons={k: dict(diff=v[0], ci=v[1], share_weeks=v[2]) for k, v in cmp_.items()},
                            settings=dict(constrained={k: (v if not isinstance(v, dict) else v) for k, v in con.items() if k in ("mode", "lam", "g", "weff", "wm")},
                                          risk={k: v for k, v in risk.items() if k in ("mode", "lam", "weff", "wm")}))
        # compatibility JSON so the table stage can show a status line
        r0 = T[T.policy == "Service-level, re-tuned per stay"].iloc[0]
        compat = dict(overall=[dict(rule="Service-level (per stay)", walks_per_1000_City=r0.City, walks_per_1000_Resort=r0.Resort)],
                      by_hotel={}, bootstrap_95ci_EUR={"saving_vs_flat_as_written": [0, 0, 0], "saving_vs_flat_fairer": [0, 0, 0], "saving_vs_no_overselling": [0, 0, 0]})
        json.dump(compat, open(os.path.join(a.outdir, name, "constrained", "proof_2017_v2.json"), "w"))
    json.dump(allout, open(os.path.join(a.outdir, "score_2017.json"), "w"), indent=1, default=float)


# ----------------------------------------------------------------------------- table
def decide4(nf, bundle):
    nf = nf.copy()
    weff = bundle.get("weff", {h: 300.0 for h in HOTELS})
    nf = v2.add_rule(nf, "new", oversell_by_hotel(nf, bundle["lam"], bundle["g"], weff))
    nf = v2.add_rule(nf, "cost_only", oversell_by_hotel(nf, bundle["lam"], None, weff))
    return nf


def stage_table(a):
    sub = os.path.join(a.outdir, a.model, "constrained")
    v3.activate(pickle.load(open(os.path.join(sub, "bundle_v2.pkl"), "rb")))
    v2.decide = decide4
    a2 = argparse.Namespace(**vars(a)); a2.outdir = sub
    v2.stage_table(a2)
    import openpyxl
    p = os.path.join(sub, f"morning_table_v2_{pd.Timestamp(a.asof).date()}.xlsx")
    wb = openpyxl.load_workbook(p)
    ws = wb["Morning table"]
    ws["A3"] = ("STATUS: costs here treat EUR 300 as paid ONCE per walked guest. 'Expected guests walked that night' is in excess room-nights "
                "(a walked guest also frees later nights). Exploratory: the 2017 window was used before.")
    wb.save(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["tune", "score", "table"])
    ap.add_argument("--data", default="hotels.csv")
    ap.add_argument("--outdir", default="outputs_v4")
    ap.add_argument("--refA", default="outputs", help="folder with the v2 bundle (model A)")
    ap.add_argument("--refC", default="outputs_v3", help="folder with the v3 variant_C bundle (model C)")
    ap.add_argument("--asof")
    ap.add_argument("--capacity")
    ap.add_argument("--model", default="C", help="A or C, for the table")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    {"tune": stage_tune, "score": stage_score, "table": stage_table}[a.stage](a)


if __name__ == "__main__":
    main()
