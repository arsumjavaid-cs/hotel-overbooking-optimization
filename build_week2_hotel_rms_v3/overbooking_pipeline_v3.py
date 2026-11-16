#!/usr/bin/env python3
"""
Overbooking pipeline v3: accounts for the seasonality of cancellations.

Keep this file in the same folder as overbooking_pipeline_v2.py (v3 reuses its functions).

  fit    train on arrivals < 2016-07-01; compare four seasonal designs on Jul-Dec 2016; freeze the best
  prove  score the frozen rule on Jan-Aug 2017 (THIRD look: v1 and v2 already used that window)
  table  morning table (xlsx) for an as-of date

The problem: v2 let the model learn month effects from ONE year of training (Jul 2015 - Jun 2016).
In that year the cancellation level was also drifting, so month and drift were tangled together,
and the model learned the wrong season. v3 separates three things:
  1. booking-level risk       (gradient boosting on bin-1 + adopted bin-2 columns, no calendar columns)
  2. seasonal shape per hotel (a smooth annual curve, 2 harmonics of day-of-year, fitted with a linear
                               trend term so the 2015->2016 drift is not mistaken for season)
  3. level drift              (daily rolling shift from resolved bookings, as in v2, now measured on top
                               of the seasonal curve so it no longer chases the season)
Four designs are compared on validation:
  A  calendar columns inside the model (this is v2)
  B  no calendar columns, no seasonal layer
  C  no calendar columns + seasonal layer (the new design)
  E  smooth annual curve as model inputs, no separate layer
"""
import argparse, importlib.util, json, os, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("v2", os.path.join(HERE, "overbooking_pipeline_v2.py"))
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)

HOTELS, N_H, CONFIGS = v2.HOTELS, v2.N_H, v2.CONFIGS
TRAIN_START = pd.Timestamp("2015-07-01")
TREND_END = 1.0                      # trend value (years since TRAIN_START) at the end of training
FS = "b2"                            # feature set adopted in v2 (bin 1 + mild bin 2)
LAMS3 = [1.5, 2.0, 3.0]
G3 = [None, 1.0, 0.5, 0.35, 0.25, 0.15]
WINDOWS3 = [56, 91]
CFG_TRY = [3, 0]
STATE = {"cal": "cal", "season": None}
log = v2.log
expit, logit = v2.expit, v2.logit


# ----------------------------------------------------------------------------- seasonal pieces
def harm(dates, H=2):
    doy = pd.DatetimeIndex(dates).dayofyear.values / 365.25
    cols = []
    for h in range(1, H + 1):
        cols += [np.sin(2 * np.pi * h * doy), np.cos(2 * np.pi * h * doy)]
    return np.column_stack(cols)


def season_offset(hotel, dates, season):
    out = np.zeros(len(hotel))
    hotel = np.asarray(hotel)
    Hm = harm(dates)
    for h in HOTELS:
        m = hotel == h
        if m.any():
            s = season[h]
            out[m] = s["b0"] + s["bt"] * TREND_END + Hm[m] @ np.asarray(s["bh"])
    return out


def fit_offset_logit(Z, y, off, l2=1.0, iters=40):
    Zc = np.column_stack([np.ones(len(Z)), Z])
    b = np.zeros(Zc.shape[1])
    pen = np.r_[0.0, np.ones(Zc.shape[1] - 1)] * l2
    for _ in range(iters):
        p = expit(off + Zc @ b)
        w = p * (1 - p)
        g = Zc.T @ (p - y) + pen * b
        H = (Zc * w[:, None]).T @ Zc + np.diag(pen)
        step = np.linalg.solve(H, g)
        b -= step
        if np.abs(step).max() < 1e-7:
            break
    return b


def fit_season(p_oof, y, hotel, arr):
    out = {}
    trend = (pd.DatetimeIndex(arr) - TRAIN_START).days.values / 365.25
    Hm = harm(arr)
    for h in HOTELS:
        m = hotel == h
        b = fit_offset_logit(np.column_stack([Hm[m], trend[m]]), y[m], logit(p_oof[m]))
        out[h] = dict(b0=float(b[0]), bh=[float(x) for x in b[1:5]], bt=float(b[5]))
    return out


# ----------------------------------------------------------------------------- patched v2 hooks
_orig_make_X, _orig_raw_matrix, _orig_raw_rows = v2.make_X, v2.raw_matrix, v2.raw_rows


def make_X3(d, k, cats, fs):
    X = _orig_make_X(d, k, cats, fs)
    cal = STATE["cal"]
    if cal in ("none", "harm"):
        X = X.drop(columns=["month", "week", "dom"])
    if cal == "harm":
        Hm = harm(d.arr.values)
        for j, nm in enumerate(["doy_s1", "doy_c1", "doy_s2", "doy_c2"]):
            X[nm] = Hm[:, j]
    return X


def raw_matrix3(model, d, cats, fs):
    L, P = _orig_raw_matrix(model, d, cats, fs)
    if STATE["season"]:
        off = season_offset(d.hotel.values, d.arr.values, STATE["season"])
        P = expit(logit(P) + off[:, None])
    return L, P


def raw_rows3(model, d, cats, fs):
    rr = _orig_raw_rows(model, d, cats, fs)
    if STATE["season"]:
        off = season_offset(rr.hotel.values, rr.arr.values, STATE["season"])
        rr["p"] = expit(logit(rr.p.values) + off)
    return rr


v2.make_X, v2.raw_matrix, v2.raw_rows = make_X3, raw_matrix3, raw_rows3


def activate(bundle):
    STATE["cal"] = bundle.get("cal", "cal")
    STATE["season"] = bundle.get("season")


# ----------------------------------------------------------------------------- fit helpers
def stack3(d, cats):
    Xs, ys, hs, ks, ar = [], [], [], [], []
    for k in range(N_H):
        m = v2.live_mask(d, k)
        if m.any():
            s = d[m]
            Xs.append(v2.make_X(s, k, cats, FS)); ys.append(s.y.values); hs.append(s.hotel.values)
            ks.append(np.full(m.sum(), k)); ar.append(s.arr.values)
    return pd.concat(Xs), np.concatenate(ys), np.concatenate(hs), np.concatenate(ks), np.concatenate(ar)


def new_model(i):
    return HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0, **CONFIGS[i])


def oof_predictions(Xtr, ytr, arr, cfg_i):
    iso = pd.DatetimeIndex(arr).isocalendar()
    wk = iso.year.astype(int).values * 100 + iso.week.astype(int).values
    codes = pd.factorize(wk)[0]
    perm = np.random.default_rng(0).permutation(codes.max() + 1)
    fold = perm[codes] % 4
    oof = np.zeros(len(ytr))
    for f in range(4):
        tr_m = fold != f
        m = new_model(cfg_i).fit(Xtr[tr_m], ytr[tr_m])
        oof[~tr_m] = m.predict_proba(Xtr[~tr_m])[:, 1]
    return oof


def month_calibration(nf):
    t = nf[nf.tight].copy()
    t["m"] = t.night.dt.month
    g = t.groupby(["hotel", "m"]).agg(predicted=("E", "sum"), actual=("C", "sum")).reset_index()
    g["pred_over_actual"] = g.predicted / g.actual.replace(0, np.nan)
    return g


def stage_fit(a):
    os.makedirs(a.outdir, exist_ok=True)
    df = v2.load(a.data)
    tr = df[df.arr < v2.TRAIN_END].copy()
    va = df[(df.arr >= v2.TRAIN_END) & (df.arr < v2.VAL_END)].copy()
    cats = v2.build_cats(tr)
    # business inputs from training only (same as v2)
    occ = v2.occupancy(df[df.arr < v2.VAL_END], pd.Timestamp("2015-07-01"), pd.Timestamp("2016-12-31"))
    cap = {h: int(round(np.percentile(occ[h]["2015-09-01":"2016-12-31"], 99.5))) for h in HOTELS}
    ok = tr[(tr.y == 0) & (tr.adr > 0)]
    adr = {(h, int(m)): float(v) for (h, m), v in ok.groupby(["hotel", "month"]).adr.mean().items()}
    adr_fb = {h: float(ok[ok.hotel == h].adr.mean()) for h in HOTELS}
    flat_rate = {h: float(tr[tr.hotel == h].y.mean()) for h in HOTELS}
    trS = tr[tr.n > 0].reset_index(drop=True)
    Ltr = np.stack([v2.live_mask(trS, k) for k in range(N_H)], 1)
    nf_tr = v2.night_frame(trS, Ltr, None, pd.Timestamp("2015-09-01"), pd.Timestamp("2016-06-30"))
    gg = nf_tr.groupby(["hotel", "h"])[["C", "B"]].sum()
    flat_rate_h = {(h, int(hh)): float(r.C / r.B) for (h, hh), r in gg.iterrows()}
    base_bundle = dict(cats=cats, fs=FS, cap=cap, adr=adr, adr_fallback=adr_fb, flat_rate=flat_rate, flat_rate_h=flat_rate_h)
    vaS = df[(df.n > 0) & (df.arr >= pd.Timestamp("2016-05-01")) & (df.arr < v2.VAL_END)].reset_index(drop=True)
    dates = pd.date_range("2016-06-15", "2016-12-31")

    variants = {"A calendar in model (v2)": ("cal", False), "B no calendar": ("none", False),
                "C no calendar + seasonal layer": ("none", True), "E annual curve as inputs": ("harm", False)}
    allgrid, store, cache = [], {}, {}
    for name, (cal, use_layer) in variants.items():
        log(f"\n=== variant {name}")
        STATE["cal"], STATE["season"] = cal, None
        if cal in cache:                          # B and C share one booking-level model
            model, cfg_i, Xtr, ytr, htr, ktr, atr = cache[cal]
            Xva, yva, hva, kva, ava = stack3(va, cats)
        else:
            Xtr, ytr, htr, ktr, atr = stack3(tr, cats)
            Xva, yva, hva, kva, ava = stack3(va, cats)
            best = None
            for i in CFG_TRY:
                m = new_model(i).fit(Xtr, ytr)
                ll = log_loss(yva, m.predict_proba(Xva)[:, 1])
                if best is None or ll < best[0]:
                    best = (ll, i, m)
            _, cfg_i, model = best
            cache[cal] = (model, cfg_i, Xtr, ytr, htr, ktr, atr)
        season = None
        if use_layer:
            oof = oof_predictions(Xtr, ytr, atr, cfg_i)
            season = fit_season(oof, ytr, htr, atr)
            STATE["season"] = season
            log("  seasonal curve fitted on out-of-fold training predictions:", {h: [round(x, 2) for x in s["bh"]] for h, s in season.items()})
        p = model.predict_proba(Xva)[:, 1]
        if season:
            p = expit(logit(p) + season_offset(hva, ava, season))
        sh = v2.static_shift(p, yva, hva)
        padj = expit(logit(p) + np.array([sh[h] for h in hva]))
        log(f"  validation AUC {roc_auc_score(yva, p):.4f}; log-loss with NO validation fitting {log_loss(yva, p):.4f}; "
            f"with a val-fitted shift {log_loss(yva, padj):.4f}")
        bundle = dict(base_bundle, model=model, cal=cal, season=season, config=CONFIGS[cfg_i])
        L, Praw = v2.raw_matrix(model, vaS, cats, FS)
        rr = v2.raw_rows(model, va, cats, FS)
        modes = {"none": None}
        for W in WINDOWS3:
            modes[f"rolling {W}d"] = v2.rolling_shift(rr, W, dates)
        frames = {}
        for mname, shf in modes.items():
            P = v2.apply_shift(Praw, L, vaS, shf)
            nf = v2.add_baselines(v2.attach_context(v2.night_frame(vaS, L, P, v2.VAL_START, pd.Timestamp("2016-12-31")), bundle), bundle)
            frames[mname] = nf
            for lam in LAMS3:
                for g in G3:
                    O = v2.oversell(nf.f_hat.values, nf.v_hat.values, nf.cap.values, nf.adr.values, lam, g)
                    r = v2.add_rule(nf[nf.tight].copy(), "m", O[nf.tight.values])
                    allgrid.append(dict(variant=name, mode=mname, lam=lam, g=g, cost=r.cost_m.sum() / N_H,
                                        rate_City=v2.walk_rate(r, "m", "City Hotel"), rate_Resort=v2.walk_rate(r, "m", "Resort Hotel")))
        store[name] = dict(bundle=bundle, frames=frames, ll_raw=log_loss(yva, p), ll_adj=log_loss(yva, padj),
                           auc=roc_auc_score(yva, p))
    G = pd.DataFrame(allgrid)
    G["max_rate"] = G[["rate_City", "rate_Resort"]].max(axis=1)
    feas = G[(G.max_rate <= v2.VAL_TARGET) & G.g.notna()]
    log("\nVALIDATION, best cost per variant among settings with <= %.0f walks per 1,000 at both hotels:" % v2.VAL_TARGET)
    summary = {}
    for name in variants:
        x = feas[feas.variant == name].sort_values("cost")
        y = G[(G.variant == name) & G.g.isna()].sort_values("cost").iloc[0]
        row = x.iloc[0]
        summary[name] = dict(cost=float(row.cost), mode=row["mode"], lam=float(row.lam), g=float(row.g), rate_City=float(row.rate_City),
                             rate_Resort=float(row.rate_Resort), auc=store[name]["auc"], logloss_no_val_fit=store[name]["ll_raw"],
                             logloss_val_shift=store[name]["ll_adj"], cost_only_best=float(y.cost))
        log(f"  {name:34s} EUR {row.cost:>9,.0f}  [{row['mode']}, lam {row.lam}, g {row.g}]  walks {row.rate_City:.1f}/{row.rate_Resort:.1f}"
            f" | AUC {store[name]['auc']:.4f}, log-loss {store[name]['ll_raw']:.4f} (no val fit)")
    pick = feas.sort_values("cost").iloc[0]
    log("\nCHOSEN:", dict(pick))
    chosen = store[pick.variant]
    b = chosen["bundle"]
    b.update(mode=pick["mode"], W=(int(pick["mode"].split()[1][:-1]) if pick["mode"].startswith("rolling") else None),
             lam=float(pick.lam), g=float(pick.g), variant=pick.variant)
    cal_tab = {}
    for name in variants:
        r = summary[name]
        cal_tab[name] = month_calibration(store[name]["frames"][r["mode"]])
    log("\nPredicted / actual further cancels by calendar month on tight validation decisions (1.00 = perfect):")
    tab = pd.concat({k: v.pivot(index="m", columns="hotel", values="pred_over_actual") for k, v in cal_tab.items()}, axis=1).round(2)
    log(tab.to_string())
    with open(os.path.join(a.outdir, "bundle_v2.pkl"), "wb") as f:      # same name so the v2 table/prove stages can load it
        pickle.dump(b, f)
    for name in variants:                                             # every variant, with its own best validation setting
        s = summary[name]
        vb = dict(store[name]["bundle"], mode=s["mode"], W=(int(s["mode"].split()[1][:-1]) if s["mode"].startswith("rolling") else None),
                  lam=s["lam"], g=s["g"], variant=name)
        sub = os.path.join(a.outdir, "variant_" + name.split()[0])
        os.makedirs(sub, exist_ok=True)
        with open(os.path.join(sub, "bundle_v2.pkl"), "wb") as f:
            pickle.dump(vb, f)
    G.to_csv(os.path.join(a.outdir, "validation_grid_v3.csv"), index=False)
    tab.to_csv(os.path.join(a.outdir, "validation_calibration_by_month_v3.csv"))
    json.dump(dict(variants=summary, chosen=dict(variant=pick.variant, mode=pick["mode"], lam=float(pick.lam), g=float(pick.g)),
                   seasonal_curve=b["season"], capacity=cap), open(os.path.join(a.outdir, "settings_v3.json"), "w"), indent=1, default=float)
    log("saved bundle_v2.pkl (v3 settings), settings_v3.json, validation tables")


# ----------------------------------------------------------------------------- prove / table
def retitle_xlsx(path):
    import openpyxl
    wb = openpyxl.load_workbook(path)
    ws = wb["Morning table"]
    ws["A3"] = str(ws["A3"].value).replace("second look", "third look")
    if "Proof Jan-Aug 2017" in wb.sheetnames:
        p = wb["Proof Jan-Aug 2017"]
        p["A1"] = "Third look, Jan-Aug 2017 (settings frozen on Jul-Dec 2016; the 2017 window was already used by v1 and v2)"
    wb.save(path)


def stage_prove(a):
    if a.variant:                                   # e.g. --variant C scores that design with its own validation settings
        a.outdir = os.path.join(a.outdir, "variant_" + a.variant)
    bundle = pickle.load(open(os.path.join(a.outdir, "bundle_v2.pkl"), "rb"))
    activate(bundle)
    v2.stage_prove(a)
    pj = os.path.join(a.outdir, "proof_2017_v2.json")
    out = json.load(open(pj))
    out["note"] = "EXPLORATORY third look: the Jan-Aug 2017 window was already used by v1 and v2."
    d = pd.read_csv(os.path.join(a.outdir, "backtest_detail_2017_v2.csv"), parse_dates=["night"])
    g = month_calibration(d)
    out["calibration_by_month"] = g.round(3).to_dict("records")
    json.dump(out, open(pj, "w"), indent=1, default=float)
    log("\nPredicted / actual further cancels by month on tight 2017 decisions:")
    log(g.pivot(index="m", columns="hotel", values="pred_over_actual").round(2).to_string())
    ref = os.path.join(a.ref, "proof_2017_v2.json") if a.ref else None
    if ref and os.path.exists(ref):
        r = json.load(open(ref))["overall"][0]
        n = out["overall"][0]
        log(f"\nv2 for reference: EUR {r['cost_EUR']:,.0f}, walks per 1,000 City {r['walks_per_1000_City']:.1f} / Resort {r['walks_per_1000_Resort']:.1f}")
        log(f"v3 this run     : EUR {n['cost_EUR']:,.0f}, walks per 1,000 City {n['walks_per_1000_City']:.1f} / Resort {n['walks_per_1000_Resort']:.1f}")


def stage_table(a):
    if a.variant:
        a.outdir = os.path.join(a.outdir, "variant_" + a.variant)
    bundle = pickle.load(open(os.path.join(a.outdir, "bundle_v2.pkl"), "rb"))
    activate(bundle)
    v2.stage_table(a)
    retitle_xlsx(os.path.join(a.outdir, f"morning_table_v2_{pd.Timestamp(a.asof).date()}.xlsx"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["fit", "prove", "table"])
    ap.add_argument("--data", default="hotels.csv")
    ap.add_argument("--outdir", default="outputs_v3")
    ap.add_argument("--ref", default="outputs", help="folder holding v2 results, for the comparison line")
    ap.add_argument("--asof")
    ap.add_argument("--capacity")
    ap.add_argument("--variant", help="A, B, C or E: score that design instead of the validation winner")
    a = ap.parse_args()
    {"fit": stage_fit, "prove": stage_prove, "table": stage_table}[a.stage](a)


if __name__ == "__main__":
    main()
