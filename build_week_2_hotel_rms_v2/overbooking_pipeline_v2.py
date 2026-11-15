#!/usr/bin/env python3
"""
Overbooking pipeline v2: the walk standard is a hard constraint, bin-2 columns are tried, drift is tracked.

Stages:
  fit       train on arrivals < 2016-07-01; choose EVERY setting on Jul-Dec 2016 only; freeze
  prove     score the frozen rule on Jan-Aug 2017.  EXPLORATORY: v1 already used that window once.
  table     morning table (xlsx) for an as-of date
  segments  guest types for marketing

What changed from v1
  1. Brand standard: at most 10 walked guests per 1,000 arrivals is a constraint, not a price.
     Per decision, the oversell is the cheapest one whose EXPECTED walk rate stays under
     g x 10 per 1,000. g (a safety factor), the variance multiplier and the drift handling are
     chosen on validation so that BOTH hotels stay under 7 per 1,000 there (30% headroom).
  2. Bin-2 columns tried, with the caveat beside each (see BIN2_CAVEATS). The file only holds
     the FINAL value of each amendable column, so a value can include changes made after the
     morning being predicted. That can flatter validation scores; production would read the
     value as it stands that morning.
  3. Drift: v1 fixed a log-odds shift on validation. City late cancellations kept falling in
     2017 and the shift went stale. v2 can refit the shift every morning from bookings whose
     outcome is already known (rolling window W days). W is chosen on validation.
"""
import argparse, json, os, pickle, warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import adjusted_rand_score, log_loss, roc_auc_score, silhouette_score

warnings.filterwarnings("ignore")

TRAIN_END = pd.Timestamp("2016-07-01")
VAL_START = pd.Timestamp("2016-07-15")
VAL_END = pd.Timestamp("2017-01-01")
TEST_START = pd.Timestamp("2017-01-01")
TEST_END = pd.Timestamp("2017-08-31")
WALK_COST = 300.0
FLAT_SAFETY = 3
N_H = 14
PRESSURE = 0.80
O_MAX = 150
TARGET = 10.0            # brand standard: walked guests per 1,000 arrivals
VAL_TARGET = 7.0         # what validation must show for BOTH hotels (30% headroom)
LAMBDAS = [1.5, 2.0, 3.0, 4.0]
G_GRID = [None, 1.0, 0.7, 0.5, 0.35, 0.25, 0.15]     # None = cost only (no constraint)
WINDOWS = [28, 56, 91]
MIN_ROWS = 2000
ADOPT_GAIN = 0.003       # adopt the mild bin-2 set only if validation log-loss improves by this much
CONFIGS = [
    dict(max_leaf_nodes=15, learning_rate=0.10, max_iter=150, l2_regularization=1.0, min_samples_leaf=200),
    dict(max_leaf_nodes=31, learning_rate=0.08, max_iter=200, l2_regularization=1.0, min_samples_leaf=100),
    dict(max_leaf_nodes=63, learning_rate=0.05, max_iter=250, l2_regularization=3.0, min_samples_leaf=100),
    dict(max_leaf_nodes=15, learning_rate=0.05, max_iter=300, l2_regularization=5.0, min_samples_leaf=300),
]
MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July",
                                      "August", "September", "October", "November", "December"], 1)}
HOTELS = ["City Hotel", "Resort Hotel"]
CATS = ["hotel", "country_f", "market_segment", "distribution_channel", "reserved_room_type",
        "customer_type", "agent_f"]
NUMS = ["lead_time", "month", "week", "dom", "dow", "is_repeated_guest", "previous_cancellations",
        "previous_bookings_not_canceled", "has_company", "k", "age"]
B2_CATS = ["meal"]
B2_NUMS = ["nights_we", "nights_wk", "adults", "children", "babies", "dwl", "adr_f"]
LEAKY_NUMS = ["booking_changes", "special_requests"]
BIN2_CAVEATS = {
    "stays_in_weekend_nights / stays_in_week_nights": "Final length of stay; can be amended before arrival.",
    "adults / children / babies": "Final party size; can change before arrival.",
    "meal": "Final board basis; can be amended.",
    "days_in_waiting_list": "Final once the booking is confirmed; 97% are zero.",
    "adr": "Final average rate; amendments change it. A derived value (revenue / nights).",
    "booking_changes (NOT ADOPTED)": "Counted until check-in or cancellation, so a cancelled booking has less time to "
                                     "accumulate changes: partly an outcome proxy.",
    "total_of_special_requests (NOT ADOPTED)": "Accumulates until arrival or cancellation: same truncation by the outcome.",
}


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- data
def load(path):
    df = pd.read_csv(path)
    df["arr"] = pd.to_datetime(dict(year=df.arrival_date_year, month=df.arrival_date_month.map(MONTHS),
                                    day=df.arrival_date_day_of_month))
    df["rsd"] = pd.to_datetime(df.reservation_status_date)
    df["n"] = (df.stays_in_weekend_nights + df.stays_in_week_nights).astype(int)
    df["dbc"] = (df.arr - df.rsd).dt.days
    df["cancelled"] = (df.reservation_status == "Canceled")
    df["y"] = df.is_canceled.astype(int)
    df["month"] = df.arr.dt.month
    df["week"] = df.arrival_date_week_number
    df["dom"] = df.arrival_date_day_of_month
    df["dow"] = df.arr.dt.dayofweek
    df["has_company"] = df.company.notna().astype(int)
    df["agent_raw"] = df.agent.fillna(-1).astype(int)
    df["children"] = df.children.fillna(0)
    df["nights_we"] = df.stays_in_weekend_nights
    df["nights_wk"] = df.stays_in_week_nights
    df["dwl"] = df.days_in_waiting_list
    df["adr_f"] = df.adr.clip(0, 1000)
    df["special_requests"] = df.total_of_special_requests
    return df.reset_index(drop=True)


def build_cats(tr):
    cats = {"country": sorted(set(tr.country.fillna("NA").value_counts().head(20).index) | {"OTH", "NA"}),
            "agent": sorted(set(int(a) for a in tr.agent_raw.value_counts().head(30).index) | {-1, -2})}
    cats["hotel"] = HOTELS
    cats["country_f"] = cats["country"]
    cats["agent_f"] = [str(a) for a in cats["agent"]]
    for c in ["market_segment", "distribution_channel", "reserved_room_type", "customer_type", "meal"]:
        cats[c] = sorted(tr[c].astype(str).unique())
    return cats


def make_X(d, k, cats, fs):
    X = pd.DataFrame(index=d.index)
    X["hotel"] = d.hotel
    c = d.country.fillna("NA")
    X["country_f"] = c.where(c.isin(cats["country"]), "OTH")
    X["agent_f"] = d.agent_raw.where(d.agent_raw.isin(cats["agent"]), -2).astype(str)
    for col in ["market_segment", "distribution_channel", "reserved_room_type", "customer_type"]:
        X[col] = d[col].astype(str)
    for col in ["lead_time", "month", "week", "dom", "dow", "is_repeated_guest", "previous_cancellations",
                "previous_bookings_not_canceled", "has_company"]:
        X[col] = d[col].values
    X["k"] = k
    X["age"] = d.lead_time.values - k
    cat_cols, num_cols = list(CATS), list(NUMS)
    if fs in ("b2", "b2leaky"):
        X["meal"] = d.meal.astype(str)
        for col in B2_NUMS:
            X[col] = d[col].values
        cat_cols += B2_CATS; num_cols += B2_NUMS
    if fs == "b2leaky":
        for col in LEAKY_NUMS:
            X[col] = d[col].values
        num_cols += LEAKY_NUMS
    for col in cat_cols:
        X[col] = pd.Categorical(X[col], categories=cats[col])
    return X[cat_cols + num_cols]


def live_mask(d, k):
    return (d.lead_time.values > k) & ~(d.cancelled.values & (d.dbc.values > k))


def stack(d, cats, fs):
    Xs, ys, hs, ks = [], [], [], []
    for k in range(N_H):
        m = live_mask(d, k)
        if m.any():
            s = d[m]
            Xs.append(make_X(s, k, cats, fs)); ys.append(s.y.values); hs.append(s.hotel.values)
            ks.append(np.full(m.sum(), k))
    return pd.concat(Xs), np.concatenate(ys), np.concatenate(hs), np.concatenate(ks)


def expit(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def static_shift(p, y, hot):
    out = {}
    for h in HOTELS:
        mk = hot == h
        lo, hi = -3.0, 3.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if expit(logit(p[mk]) + mid).mean() > y[mk].mean():
                hi = mid
            else:
                lo = mid
        out[h] = (lo + hi) / 2
    return out


def raw_matrix(model, d, cats, fs):
    L = np.zeros((len(d), N_H), bool)
    P = np.full((len(d), N_H), np.nan)
    for k in range(N_H):
        m = live_mask(d, k)
        L[:, k] = m
        if m.any():
            P[m, k] = model.predict_proba(make_X(d[m], k, cats, fs))[:, 1]
    return L, P


def raw_rows(model, d, cats, fs):
    parts = []
    for k in range(N_H):
        m = live_mask(d, k)
        if m.any():
            s = d[m]
            parts.append(pd.DataFrame({"hotel": s.hotel.values, "arr": s.arr.values, "y": s.y.values,
                                       "p": model.predict_proba(make_X(s, k, cats, fs))[:, 1]}))
    return pd.concat(parts, ignore_index=True)


def rolling_shift(rr, W, dates):
    """Daily log-odds shift per hotel, fitted on bookings that ARRIVED in [t-W, t-1] (outcomes known on morning t)."""
    out = {}
    for h in HOTELS:
        x = rr[rr.hotel == h]
        a = x.arr.values.astype("datetime64[D]"); p = logit(x.p.values); y = x.y.values.astype(float)
        o = np.argsort(a); a, p, y = a[o], p[o], y[o]
        vals = []
        for t in dates:
            lo = np.searchsorted(a, np.datetime64(t - pd.Timedelta(days=W)), "left")
            hi = np.searchsorted(a, np.datetime64(t), "left")
            if hi - lo < MIN_ROWS:
                vals.append(0.0); continue
            pp, yy = p[lo:hi], y[lo:hi]
            l_, h_ = -3.0, 3.0
            for _ in range(40):
                mid = (l_ + h_) / 2
                if expit(pp + mid).mean() > yy.mean():
                    h_ = mid
                else:
                    l_ = mid
            vals.append((l_ + h_) / 2)
        out[h] = pd.Series(vals, index=pd.DatetimeIndex(dates))
    return pd.DataFrame(out)


def apply_shift(P, L, d, shift):
    """shift: None, dict {hotel: delta} or DataFrame of daily deltas. Decision date for (booking, k) is arr - k."""
    out = np.full_like(P, np.nan)
    for k in range(N_H):
        m = L[:, k]
        if not m.any():
            continue
        hot = d.hotel.values[m]
        if shift is None:
            delta = np.zeros(m.sum())
        elif isinstance(shift, dict):
            delta = np.array([shift[h] for h in hot])
        else:
            t = pd.DatetimeIndex(d.arr.values[m]) - pd.Timedelta(days=k)
            delta = np.zeros(m.sum())
            for h in HOTELS:
                mh = hot == h
                delta[mh] = shift[h].reindex(t[mh]).fillna(0.0).values
        out[m, k] = expit(logit(P[m, k]) + delta)
    return out


# ----------------------------------------------------------------------------- nights
def night_frame(S, L, P, d0, d1):
    n = S.n.values
    arr = S.arr.values.astype("datetime64[D]")
    idx = np.repeat(np.arange(len(S)), n)
    j = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    night = arr[idx] + j.astype("timedelta64[D]")
    keep = (night >= np.datetime64(d0)) & (night <= np.datetime64(d1))
    idx, j, night = idx[keep], j[keep], night[keep]
    hot, yv = S.hotel.values[idx], S.y.values[idx]
    frames = []
    for h in range(N_H):
        k = h - j
        ih = (k < 0) & (yv == 0)
        kk = np.clip(k, 0, N_H - 1)
        live = np.where(k >= 0, L[idx, kk], False)
        present = live | ih
        p = np.zeros(len(idx)) if P is None else np.where(live, np.nan_to_num(P[idx, kk]), 0.0)
        t = pd.DataFrame({"hotel": hot[present], "night": night[present], "ih": ih[present].astype(int),
                          "p": p[present], "v": (p * (1 - p))[present], "y": yv[present]})
        g = t.groupby(["hotel", "night"]).agg(B=("y", "size"), inhouse=("ih", "sum"), E=("p", "sum"),
                                              V=("v", "sum"), C=("y", "sum")).reset_index()
        g["h"] = h
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def occupancy(S, d0, d1):
    S = S[(S.y == 0) & (S.n > 0)]
    days = (d1 - d0).days + 1
    out = {}
    for h, g in S.groupby("hotel"):
        diff = np.zeros(days + 2, int)
        for s, n in zip((g.arr - d0).dt.days.values, g.n.values):
            lo, hi = max(s, 0), min(s + n, days)
            if hi > lo:
                diff[lo] += 1; diff[hi] -= 1
        out[h] = pd.Series(np.cumsum(diff)[:days], index=pd.date_range(d0, periods=days))
    return out


# ----------------------------------------------------------------------------- rules
def oversell(f, v, cap, adr, lam, g=None):
    """Cheapest oversell O (EUR adr per empty room, EUR 300 per walk); with g, only O whose expected
    walk rate is <= g * TARGET per 1,000 arrivals are allowed (O = 0 if none)."""
    O = np.arange(O_MAX + 1)[None, :]
    tot = np.asarray(cap)[:, None] + O
    m = tot * f[:, None] - O
    s = np.sqrt(np.maximum(tot * v[:, None] * lam ** 2, 1e-9))
    z = m / s
    ez = s * norm.pdf(z) + m * norm.cdf(z)
    ew = ez - m
    cost = adr[:, None] * ez + WALK_COST * ew
    if g is not None:
        rate = 1000 * ew / np.maximum(tot * (1 - f[:, None]), 1e-9)
        cost = np.where(rate <= g * TARGET, cost, np.inf)
    return cost.argmin(1)


def risk_columns(O, f, v, cap, lam):
    tot = np.asarray(cap) + O
    m = tot * f - O
    s = np.sqrt(np.maximum(tot * v * lam ** 2, 1e-9))
    z = m / s
    ez = s * norm.pdf(z) + m * norm.cdf(z)
    ew = ez - m
    return norm.cdf((O - 0.5 - tot * f) / s), ew, 1000 * ew / np.maximum(tot * (1 - f), 1e-9)


def realised(O, f_real, cap, adr):
    arrivals = (cap + O) * (1 - f_real)
    walk = np.maximum(0.0, arrivals - cap)
    empty = np.maximum(0.0, cap - arrivals)
    return arrivals, walk, empty, adr * empty + WALK_COST * walk


def flat_oversell(B, rate, safety=FLAT_SAFETY):
    return np.maximum(0, np.rint(rate * B).astype(int) - safety)


def attach_context(nf, bundle):
    nf = nf.copy()
    nf["cap"] = nf.hotel.map(bundle["cap"])
    nf["adr"] = [bundle["adr"].get((h, m), bundle["adr_fallback"][h]) for h, m in zip(nf.hotel, nf.night.dt.month)]
    B = nf.B.clip(lower=1)
    nf["f_hat"] = np.where(nf.B > 0, nf.E / B, 0.0)
    nf["v_hat"] = np.where(nf.B > 0, nf.V / B, 0.0)
    nf["f_real"] = np.where(nf.B > 0, nf.C / B, 0.0)
    nf["tight"] = nf.B >= PRESSURE * nf.cap
    return nf


def add_rule(nf, name, O):
    nf[f"O_{name}"] = O
    a, w, e, c = realised(O, nf.f_real.values, nf.cap.values, nf.adr.values)
    nf[f"arr_{name}"], nf[f"walk_{name}"], nf[f"empty_{name}"], nf[f"cost_{name}"] = a, w, e, c
    return nf


def add_baselines(nf, bundle):
    nf = nf.copy()
    nf = add_rule(nf, "flat", flat_oversell(nf.B.values, nf.hotel.map(bundle["flat_rate"]).values))
    rh = np.array([bundle["flat_rate_h"][(h, hh)] for h, hh in zip(nf.hotel, nf.h)])
    nf = add_rule(nf, "flat_h", flat_oversell(nf.B.values, rh))
    nf = add_rule(nf, "none", np.zeros(len(nf), int))
    return nf


def walk_rate(t, rule, hotel=None):
    x = t if hotel is None else t[t.hotel == hotel]
    return 1000 * x[f"walk_{rule}"].sum() / max(x[f"arr_{rule}"].sum(), 1e-9)


# ----------------------------------------------------------------------------- stage: fit
def stage_fit(a):
    os.makedirs(a.outdir, exist_ok=True)
    df = load(a.data)
    tr = df[df.arr < TRAIN_END].copy()
    va = df[(df.arr >= TRAIN_END) & (df.arr < VAL_END)].copy()
    log(f"train bookings {len(tr):,}; validation bookings {len(va):,}")
    cats = build_cats(tr)

    # ---- 1. which columns: bin 1 vs bin 1 + mild bin 2 vs (tried, never adopted) + leaky bin 2
    abl = {}
    for fs in ["bin1", "b2", "b2leaky"]:
        Xtr, ytr, _, _ = stack(tr, cats, fs)
        Xva, yva, hva, kva = stack(va, cats, fs)
        best = None
        for i, cfg in enumerate(CONFIGS):
            m = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0, **cfg)
            m.fit(Xtr, ytr)
            p = m.predict_proba(Xva)[:, 1]
            ll = log_loss(yva, p)
            if best is None or ll < best["ll_raw"]:
                best = dict(cfg=i, model=m, ll_raw=ll, auc=roc_auc_score(yva, p), p=p)
        sh = static_shift(best["p"], yva, hva)
        padj = expit(logit(best["p"]) + np.array([sh[h] for h in hva]))
        best["ll"] = log_loss(yva, padj)
        best["ll_far"] = log_loss(yva[kva >= 7], padj[kva >= 7])
        best["ll_near"] = log_loss(yva[kva < 7], padj[kva < 7])
        abl[fs] = best
        log(f"  [{fs}] config {best['cfg']}: val AUC {best['auc']:.4f}, log-loss {best['ll']:.4f} "
            f"(far>=7d {best['ll_far']:.4f}, near<7d {best['ll_near']:.4f}) after a val-fitted shift")
    gain = abl["bin1"]["ll"] - abl["b2"]["ll"]
    fs = "b2" if gain >= ADOPT_GAIN else "bin1"
    log(f"mild bin-2 log-loss gain {gain:+.4f} (adopt if >= {ADOPT_GAIN}) -> feature set: {fs}")
    log(f"leaky bin-2 gain over bin1 {abl['bin1']['ll'] - abl['b2leaky']['ll']:+.4f}  (shown for information, never adopted)")
    model = abl[fs]["model"]

    # ---- 2. business inputs from training only
    occ = occupancy(df[df.arr < VAL_END], pd.Timestamp("2015-07-01"), pd.Timestamp("2016-12-31"))
    cap = {h: int(round(np.percentile(occ[h]["2015-09-01":"2016-12-31"], 99.5))) for h in HOTELS}
    ok = tr[(tr.y == 0) & (tr.adr > 0)]
    adr = {(h, int(m)): float(v) for (h, m), v in ok.groupby(["hotel", "month"]).adr.mean().items()}
    adr_fb = {h: float(ok[ok.hotel == h].adr.mean()) for h in HOTELS}
    flat_rate = {h: float(tr[tr.hotel == h].y.mean()) for h in HOTELS}
    trS = tr[tr.n > 0].reset_index(drop=True)
    Ltr = np.stack([live_mask(trS, k) for k in range(N_H)], 1)
    nf_tr = night_frame(trS, Ltr, None, pd.Timestamp("2015-09-01"), pd.Timestamp("2016-06-30"))
    gg = nf_tr.groupby(["hotel", "h"])[["C", "B"]].sum()
    flat_rate_h = {(h, int(hh)): float(r.C / r.B) for (h, hh), r in gg.iterrows()}
    bundle = dict(model=model, cats=cats, fs=fs, cap=cap, adr=adr, adr_fallback=adr_fb, flat_rate=flat_rate,
                  flat_rate_h=flat_rate_h, config=CONFIGS[abl[fs]["cfg"]])
    log("capacity estimate:", cap)

    # ---- 3. drift handling, variance multiplier, constraint factor: all on validation
    vaS = df[(df.n > 0) & (df.arr >= pd.Timestamp("2016-05-01")) & (df.arr < VAL_END)].reset_index(drop=True)
    L, Praw = raw_matrix(model, vaS, cats, fs)
    rr = raw_rows(model, va, cats, fs)
    Xva, yva, hva, _ = stack(va, cats, fs)
    dates = pd.date_range("2016-06-15", "2016-12-31")
    modes = {"none": None, "static (in-sample reference)": static_shift(abl[fs]["p"], yva, hva)}
    for W in WINDOWS:
        modes[f"rolling {W}d"] = rolling_shift(rr, W, dates)
    grid = []
    frames = {}
    for mname, sh in modes.items():
        P = apply_shift(Praw, L, vaS, sh)
        nf = add_baselines(attach_context(night_frame(vaS, L, P, VAL_START, pd.Timestamp("2016-12-31")), bundle), bundle)
        frames[mname] = nf
        for lam in LAMBDAS:
            for g in G_GRID:
                O = oversell(nf.f_hat.values, nf.v_hat.values, nf.cap.values, nf.adr.values, lam, g)
                r = add_rule(nf[nf.tight].copy(), "m", O[nf.tight.values])
                grid.append(dict(mode=mname, lam=lam, g=g, cost=r.cost_m.sum() / N_H,
                                 rate_City=walk_rate(r, "m", "City Hotel"), rate_Resort=walk_rate(r, "m", "Resort Hotel"),
                                 walked=r.walk_m.sum() / N_H, empty=r.empty_m.sum() / N_H))
    G = pd.DataFrame(grid)
    G["max_rate"] = G[["rate_City", "rate_Resort"]].max(axis=1)
    allowed = G[~G["mode"].str.startswith("static")]
    feas = allowed[(allowed.max_rate <= VAL_TARGET) & allowed.g.notna()]
    if len(feas):
        pick = feas.sort_values("cost").iloc[0]
    else:
        pick = allowed[allowed.g.notna()].sort_values("max_rate").iloc[0]
        log("WARNING: no setting met the validation target; using the lowest walk rate.")
    log("\nValidation grid, best cost per mode among settings meeting <= %.0f per 1,000 for both hotels:" % VAL_TARGET)
    for mname in modes:
        x = G[(G["mode"] == mname) & (G.max_rate <= VAL_TARGET) & G.g.notna()]
        y = G[(G["mode"] == mname) & G.g.isna()].sort_values("cost").iloc[0]
        s = (f"  {mname:30s} constrained best: EUR {x.cost.min():,.0f}" if len(x) else f"  {mname:30s} constrained: none feasible")
        log(s + f" | cost-only best EUR {y.cost:,.0f} at {y.max_rate:.1f} per 1,000")
    log("\nCHOSEN:", dict(pick))
    nf = frames[pick["mode"]]
    t = nf[nf.tight]
    summary = {}
    for name, rule in [("none", "none"), ("flat_h", "flat_h"), ("flat", "flat")]:
        summary[name] = dict(cost=float(t[f"cost_{rule}"].sum() / N_H), City=walk_rate(t, rule, "City Hotel"),
                             Resort=walk_rate(t, rule, "Resort Hotel"))
    summary["new"] = {k: float(pick[k]) for k in ["cost", "rate_City", "rate_Resort"]}
    log("validation comparison:", json.dumps(summary, indent=1))

    bundle.update(mode=pick["mode"], W=(int(pick["mode"].split()[1][:-1]) if pick["mode"].startswith("rolling") else None),
                  lam=float(pick["lam"]), g=(None if pd.isna(pick["g"]) else float(pick["g"])))
    with open(os.path.join(a.outdir, "bundle_v2.pkl"), "wb") as f:
        pickle.dump(bundle, f)
    G.to_csv(os.path.join(a.outdir, "validation_grid_v2.csv"), index=False)
    settings = dict(feature_set=fs, mode=bundle["mode"], window_days=bundle["W"], lambda_=bundle["lam"], g=bundle["g"],
                    walk_standard_per_1000=TARGET, validation_target_per_1000=VAL_TARGET, capacity_estimate=cap,
                    bin2_ablation={k: dict(logloss=float(v["ll"]), auc=float(v["auc"]), far=float(v["ll_far"]),
                                           near=float(v["ll_near"])) for k, v in abl.items()},
                    bin2_caveats=BIN2_CAVEATS, validation_comparison=summary)
    with open(os.path.join(a.outdir, "settings_v2.json"), "w") as f:
        json.dump(settings, f, indent=1, default=float)
    log("saved bundle_v2.pkl, settings_v2.json, validation_grid_v2.csv")


# ----------------------------------------------------------------------------- stage: prove
def decide(nf, bundle):
    nf = nf.copy()
    O = oversell(nf.f_hat.values, nf.v_hat.values, nf.cap.values, nf.adr.values, bundle["lam"], bundle["g"])
    nf = add_rule(nf, "new", O)
    O2 = oversell(nf.f_hat.values, nf.v_hat.values, nf.cap.values, nf.adr.values, bundle["lam"], None)
    nf = add_rule(nf, "cost_only", O2)
    return nf


def stage_prove(a):
    bundle = pickle.load(open(os.path.join(a.outdir, "bundle_v2.pkl"), "rb"))
    df = load(a.data)
    fs, model, cats = bundle["fs"], bundle["model"], bundle["cats"]
    S = df[(df.n > 0) & (df.arr >= pd.Timestamp("2016-11-01")) & (df.arr <= TEST_END)].reset_index(drop=True)
    L, Praw = raw_matrix(model, S, cats, fs)
    shift = None
    if bundle["W"]:
        rr = raw_rows(model, df[(df.arr >= TRAIN_END) & (df.arr <= TEST_END)], cats, fs)
        shift = rolling_shift(rr, bundle["W"], pd.date_range("2016-11-01", "2017-08-31"))
    P = apply_shift(Praw, L, S, shift)
    nf = add_baselines(attach_context(night_frame(S, L, P, TEST_START, TEST_END), bundle), bundle)
    nf = decide(nf, bundle)
    nf.to_csv(os.path.join(a.outdir, "backtest_detail_2017_v2.csv"), index=False)
    t = nf[nf.tight].copy()
    t["m"] = t.night.dt.to_period("M").astype(str)
    names = {"new": "New rule (constrained)", "cost_only": "Same model, cost only", "flat_h": "Flat rule, fairer version",
             "none": "No overselling", "flat": "Flat rule as written"}
    rows = []
    for r, nm in names.items():
        row = {"rule": nm, "cost_EUR": t[f"cost_{r}"].sum() / N_H, "rooms_walked": t[f"walk_{r}"].sum() / N_H,
               "rooms_empty": t[f"empty_{r}"].sum() / N_H, "walks_per_1000_both": walk_rate(t, r),
               "walks_per_1000_City": walk_rate(t, r, "City Hotel"), "walks_per_1000_Resort": walk_rate(t, r, "Resort Hotel")}
        for h in HOTELS:
            row[f"worst_month_{h.split()[0]}"] = max(walk_rate(t[t.m == mo], r, h) for mo in sorted(t.m.unique())
                                                    if (t[(t.m == mo) & (t.hotel == h)].shape[0] > 0))
        row["avg_oversell"] = t[f"O_{r}"].mean()
        rows.append(row)
    summ = pd.DataFrame(rows)
    by_hotel = {}
    for h in HOTELS:
        x = t[t.hotel == h]
        by_hotel[h] = pd.DataFrame([{"rule": nm, "cost_EUR": x[f"cost_{r}"].sum() / N_H, "rooms_walked": x[f"walk_{r}"].sum() / N_H,
                                     "rooms_empty": x[f"empty_{r}"].sum() / N_H, "walks_per_1000": walk_rate(x, r)}
                                    for r, nm in names.items()])
    by_h = {}
    for hh in [0, 1, 3, 7, 13]:
        x = t[t.h == hh]
        by_h[hh] = {"new": {h.split()[0]: walk_rate(x, "new", h) for h in HOTELS},
                    "cost_only": {h.split()[0]: walk_rate(x, "cost_only", h) for h in HOTELS},
                    "avg_oversell_new": {h.split()[0]: float(x[x.hotel == h].O_new.mean()) for h in HOTELS},
                    "avg_oversell_hindsight_perfect": {h.split()[0]: float(((x.cap * x.f_real / (1 - x.f_real).clip(lower=1e-9))[x.hotel == h]).mean()) for h in HOTELS}}
    nn = t.groupby(["hotel", "night"]).agg(new=("cost_new", "sum"), flat=("cost_flat", "sum"), flat_h=("cost_flat_h", "sum"),
                                           none=("cost_none", "sum")).reset_index()
    for c in ["new", "flat", "flat_h", "none"]:
        nn[c] /= N_H
    nn["week"] = nn.night.dt.isocalendar().week.astype(int) + 100 * nn.night.dt.year
    wk = nn.groupby("week")[["new", "flat", "flat_h", "none"]].sum()
    rng = np.random.default_rng(0)
    bs = np.array([(lambda s: [s.flat - s.new, s.flat_h - s.new, s.none - s.new])(wk.iloc[rng.integers(0, len(wk), len(wk))].sum())
                   for _ in range(4000)])
    tot = wk.sum()
    ci = {k: [float(v), *np.percentile(bs[:, i], [2.5, 97.5]).tolist()] for i, (k, v) in enumerate(
        [("saving_vs_flat_as_written", tot.flat - tot.new), ("saving_vs_flat_fairer", tot.flat_h - tot.new),
         ("saving_vs_no_overselling", tot.none - tot.new)])}
    cal = {int(hh): {"predicted": float(t[t.h == hh].E.sum()), "actual": float(t[t.h == hh].C.sum())} for hh in [0, 1, 3, 7, 13]}
    out = dict(note="EXPLORATORY second look: the Jan-Aug 2017 window was already used once by v1.",
               tight_decisions=int(len(t)), overall=summ.to_dict("records"), by_hotel={h: v.to_dict("records") for h, v in by_hotel.items()},
               by_horizon=by_h, bootstrap_95ci_EUR=ci, calibration=cal,
               compliance={h: bool(walk_rate(t, "new", h) <= TARGET) for h in HOTELS},
               settings=dict(feature_set=fs, mode=bundle["mode"], lam=bundle["lam"], g=bundle["g"]))
    json.dump(out, open(os.path.join(a.outdir, "proof_2017_v2.json"), "w"), indent=1, default=float)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    log("\nEXPLORATORY SECOND LOOK, Jan-Aug 2017 (settings:", out["settings"], ")")
    log(summ.round(1).to_string(index=False))
    for h, v in by_hotel.items():
        log(f"\n{h}\n", v.round(1).to_string(index=False))
    log("\nWalks per 1,000 by horizon:", json.dumps(by_h, indent=0))
    log("\nBootstrap 95% CI of saving (EUR):", json.dumps(ci))
    log("calibration (predicted vs actual further cancellations):", json.dumps(cal))
    log("compliance with 10 per 1,000 (hotel level):", out["compliance"])


# ----------------------------------------------------------------------------- stage: table
def stage_table(a):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    bundle = pickle.load(open(os.path.join(a.outdir, "bundle_v2.pkl"), "rb"))
    for h, v in (json.loads(a.capacity) if a.capacity else {}).items():
        bundle["cap"][h] = int(v)
    asof = pd.Timestamp(a.asof)
    df = load(a.data)
    fs, model, cats = bundle["fs"], bundle["model"], bundle["cats"]
    S = df[(df.n > 0) & (df.arr >= asof - pd.Timedelta(days=75)) & (df.arr <= asof + pd.Timedelta(days=N_H))].reset_index(drop=True)
    L, Praw = raw_matrix(model, S, cats, fs)
    shift = None
    if bundle["W"]:
        win = df[(df.arr >= asof - pd.Timedelta(days=bundle["W"])) & (df.arr < asof)]
        shift = rolling_shift(raw_rows(model, win, cats, fs), bundle["W"], [asof])
        log("today's drift shift:", shift.round(3).to_dict("records"))
    P = apply_shift(Praw, L, S, shift)
    nf = night_frame(S, L, P, asof, asof + pd.Timedelta(days=N_H - 1))
    nf = attach_context(nf[(nf.night - asof).dt.days == nf.h], bundle)
    nf = add_baselines(nf, bundle)
    nf = decide(nf, bundle)
    nf["p_walk"], nf["exp_walk"], nf["rate"] = risk_columns(nf.O_new.values, nf.f_hat.values, nf.v_hat.values, nf.cap.values, bundle["lam"])
    nf["limit"] = nf.cap + nf.O_new
    nf["can_sell"] = nf.limit - nf.B
    nf = nf.sort_values(["hotel", "night"])

    pp = os.path.join(a.outdir, "proof_2017_v2.json")
    pr = json.load(open(pp)) if os.path.exists(pp) else None
    wb = Workbook(); ws = wb.active; ws.title = "Morning table"
    head = PatternFill("solid", fgColor="1F3A5F"); white = Font(color="FFFFFF", bold=True)
    key = PatternFill("solid", fgColor="FFF2CC"); tight_fill = PatternFill("solid", fgColor="FCE4D6")
    thin = Side(style="thin", color="BBBBBB")
    ws["A1"] = f"Rooms to oversell, next {N_H} nights   |   data as of the morning of {asof.date()}"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A2"] = ("Booking limit = rooms + rooms to oversell. 'Can still sell' = limit minus bookings on the books (negative = stop "
                "selling). Each figure keeps the expected walk rate under the brand standard (10 per 1,000 arrivals), with a safety "
                "factor. Shaded rows: bookings already >= 80% of rooms, so the limit is binding.")
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="top"); ws.merge_cells("A2:L2"); ws.row_dimensions[2].height = 48
    if pr:
        bits = []
        for r in pr["overall"][:1]:
            bits.append(f"Jan-Aug 2017 second look (exploratory, not a clean proof): City {r['walks_per_1000_City']:.1f}, "
                        f"Resort {r['walks_per_1000_Resort']:.1f} walked per 1,000 arrivals (limit 10).")
        ws["A3"] = "STATUS: " + " ".join(bits) + " See 'Proof' sheet."
    else:
        ws["A3"] = "STATUS: run the 'prove' stage."
    ws["A3"].font = Font(bold=True, color="C00000"); ws["A3"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("A3:L3"); ws.row_dimensions[3].height = 34
    cols = ["Hotel", "Night", "Day", "Days out", "Rooms (est.)", "Bookings on the books", "Expected cancels still to come",
            "ROOMS TO OVERSELL", "Booking limit", "Can still sell", "Expected walks per 1,000 arrivals (limit 10)",
            "Chance we walk at least one guest", "Cost-only rule says (not constrained)", "Flat rule says (for comparison)"]
    r0 = 4
    for j, c in enumerate(cols, 1):
        x = ws.cell(r0, j, c); x.fill = head; x.font = white
        x.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
    ws.row_dimensions[r0].height = 62
    for i, r in enumerate(nf.itertuples(), r0 + 1):
        vals = [r.hotel, r.night.date(), r.night.strftime("%a"), int(r.h), int(r.cap), int(r.B), round(float(r.E), 1), int(r.O_new),
                int(r.limit), int(r.can_sell), round(float(r.rate), 1), float(r.p_walk), int(r.O_cost_only), int(r.O_flat)]
        for j, v in enumerate(vals, 1):
            c = ws.cell(i, j, v); c.border = Border(top=thin, bottom=thin, left=thin, right=thin); c.alignment = Alignment(horizontal="center")
            if r.tight:
                c.fill = tight_fill
            if j == 8:
                c.fill = key; c.font = Font(bold=True)
            if j == 12:
                c.number_format = "0%"
            if j == 2:
                c.number_format = "yyyy-mm-dd"
    for j, w in enumerate([14, 12, 6, 8, 10, 12, 14, 13, 10, 10, 16, 14, 16, 16], 1):
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.freeze_panes = "A5"

    ws2 = wb.create_sheet("Proof Jan-Aug 2017")
    if pr:
        ws2["A1"] = "Second look, Jan-Aug 2017 (settings frozen on Jul-Dec 2016; the 2017 window was already used once by v1)"
        ws2["A1"].font = Font(bold=True, size=13)
        ws2["A2"] = ("Cost = EUR per empty room (monthly hotel ADR) + EUR 300 per guest walked, against a perfect-hindsight policy; "
                     "decisions where bookings were already >= 80% of rooms; per-night average over the 14 daily decisions. "
                     "Walks are modelled (extra rooms assumed to behave like the night's other bookings).")
        ws2["A2"].alignment = Alignment(wrap_text=True, vertical="top"); ws2.merge_cells("A2:J2"); ws2.row_dimensions[2].height = 48
        row = 4

        def put(title, recs, row):
            ws2.cell(row, 1, title).font = Font(bold=True)
            hdr = list(recs[0].keys())
            for j, hn in enumerate(hdr, 1):
                c = ws2.cell(row + 1, j, hn); c.fill = head; c.font = white; c.alignment = Alignment(wrap_text=True)
            for i, rec in enumerate(recs, row + 2):
                for j, hn in enumerate(hdr, 1):
                    v = rec[hn]; c = ws2.cell(i, j, round(v, 1) if isinstance(v, float) else v)
                    if hn.startswith("cost"):
                        c.number_format = "#,##0"
            return row + len(recs) + 3
        row = put("Both hotels (walks per 1,000 arrivals; the standard is 10)", pr["overall"], row)
        for h, rec in pr["by_hotel"].items():
            row = put(h, rec, row)
        for lab, k in [("Saving vs flat rule as written", "saving_vs_flat_as_written"), ("Saving vs fairer flat rule", "saving_vs_flat_fairer"),
                       ("Saving vs no overselling", "saving_vs_no_overselling")]:
            ws2.cell(row, 1, lab + " (EUR, 95% CI by week)").font = Font(bold=True)
            for j, v in enumerate(pr["bootstrap_95ci_EUR"][k], 2):
                ws2.cell(row, j, round(v))
            row += 1
        ws2.column_dimensions["A"].width = 52
        for j in range(2, 11):
            ws2.column_dimensions[get_column_letter(j)].width = 16
    ws3 = wb.create_sheet("Columns used")
    ws3["A1"] = "Second-bin columns (amendable): caveat beside each"; ws3["A1"].font = Font(bold=True, size=13)
    for i, (k, v) in enumerate(BIN2_CAVEATS.items(), 3):
        ws3.cell(i, 1, k).font = Font(bold=True); ws3.cell(i, 2, v).alignment = Alignment(wrap_text=True)
    ws3.cell(len(BIN2_CAVEATS) + 4, 1, f"Feature set in this run: {fs}")
    ws3.column_dimensions["A"].width = 52; ws3.column_dimensions["B"].width = 100
    out = os.path.join(a.outdir, f"morning_table_v2_{asof.date()}.xlsx")
    wb.save(out)
    cols_out = ["hotel", "night", "h", "cap", "B", "E", "O_new", "limit", "can_sell", "rate", "p_walk", "O_cost_only", "O_flat", "tight"]
    nf[cols_out].to_csv(os.path.join(a.outdir, f"morning_table_v2_{asof.date()}.csv"), index=False)
    log("wrote", out)
    log(nf[cols_out].round(2).to_string(index=False))


# ----------------------------------------------------------------------------- stage: segments
def seg_matrix(d, med_adr, top_countries):
    X = pd.DataFrame(index=d.index)
    X["lead"] = np.log1p(d.lead_time)
    X["nights"] = np.log1p(d.n.clip(0, 21))
    X["weekend_share"] = (d.stays_in_weekend_nights / d.n.replace(0, np.nan)).fillna(0)
    X["adults"] = d.adults.clip(0, 4)
    X["kids"] = ((d.children + d.babies) > 0).astype(float)
    adr = d.adr.where((d.adr > 0) & (d.adr < 500))
    X["adr"] = np.log1p(adr.fillna(med_adr))
    X["repeat"] = d.is_repeated_guest.astype(float)
    X["prev_cancel"] = (d.previous_cancellations > 0).astype(float)
    X["summer"] = d.month.isin([6, 7, 8, 9]).astype(float)
    num = X.copy()
    cat = pd.DataFrame({"hotel": d.hotel, "meal": d.meal.astype(str), "segment": d.market_segment, "channel": d.distribution_channel,
                        "cust": d.customer_type, "country": d.country.fillna("NA").where(d.country.fillna("NA").isin(top_countries), "OTH")})
    return num, pd.get_dummies(cat).astype(float)


def stage_segments(a):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    df = load(a.data)
    base = df[df.arr < VAL_END]
    med_adr = float(base.adr[(base.adr > 0) & (base.adr < 500)].median())
    top = list(base.country.fillna("NA").value_counts().head(10).index)
    num, cat = seg_matrix(df, med_adr, top)
    mu, sd = num.loc[base.index].mean(), num.loc[base.index].std().replace(0, 1)
    Z = np.hstack([((num - mu) / sd).values, cat.values * 0.5])
    Zb = Z[base.index.values]
    rng = np.random.default_rng(0)
    samp = rng.choice(len(Zb), 40000, replace=False)
    sil_idx = rng.choice(samp, 8000, replace=False)
    res = []
    for k in range(4, 11):
        km = MiniBatchKMeans(k, n_init=5, random_state=0, batch_size=4096).fit(Zb[samp])
        sil = silhouette_score(Zb[sil_idx], km.predict(Zb[sil_idx]))
        h = len(samp) // 2
        k1 = MiniBatchKMeans(k, n_init=3, random_state=1, batch_size=4096).fit(Zb[samp[:h]])
        k2 = MiniBatchKMeans(k, n_init=3, random_state=2, batch_size=4096).fit(Zb[samp[h:]])
        ari = adjusted_rand_score(k1.predict(Zb[sil_idx]), k2.predict(Zb[sil_idx]))
        res.append((k, sil, ari)); log(f"  k={k}: silhouette {sil:.3f}, stability (ARI between halves) {ari:.3f}")
    cand = [r for r in res if 5 <= r[0] <= 8]
    k = max(cand, key=lambda r: r[1] + r[2])[0]
    log("chosen k:", k)
    km = MiniBatchKMeans(k, n_init=10, random_state=0, batch_size=4096).fit(Zb[samp])
    df["seg"] = km.predict(Z)
    d = df.copy(); d["rev"] = np.where(d.y == 0, d.adr.clip(0, 1000) * d.n, 0.0)
    kept = d[d.y == 0]
    rows = []
    for s, g in d.groupby("seg"):
        kp = g[g.y == 0]
        r = dict(segment=int(s), bookings=len(g), share_of_bookings=len(g) / len(d), share_of_revenue=g.rev.sum() / d.rev.sum(),
                 city_share=(g.hotel == "City Hotel").mean(), median_lead_days=g.lead_time.median(), median_nights=g.n.median(),
                 avg_adults=g.adults.mean(), with_children_share=((g.children + g.babies) > 0).mean(),
                 median_adr=kp.adr[kp.adr > 0].median(), repeat_guest_share=g.is_repeated_guest.mean(),
                 summer_share=g.month.isin([6, 7, 8, 9]).mean(), top_market_segment=g.market_segment.value_counts().index[0],
                 top_segment_share=g.market_segment.value_counts(normalize=True).iloc[0],
                 top_channel=g.distribution_channel.value_counts().index[0], top_country=g.country.fillna("NA").value_counts().index[0],
                 top_country_share=g.country.fillna("NA").value_counts(normalize=True).iloc[0],
                 cancel_rate=g.y.mean(), cancel_rate_2017=g[g.arr >= TEST_START].y.mean(), nonrefund_share=(g.deposit_type == "Non Refund").mean(),
                 meal_top=g.meal.value_counts().index[0], special_requests_avg=g.special_requests.mean())
        rows.append(r)
    prof = pd.DataFrame(rows).sort_values("share_of_bookings", ascending=False)
    prof.to_csv(os.path.join(a.outdir, "guest_types_profile.csv"), index=False)
    df[["hotel", "arr", "seg"]].to_csv(os.path.join(a.outdir, "guest_type_assignments.csv"), index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
    log(prof.round(3).T.to_string())
    seg_x = pd.crosstab(df.seg, df.market_segment, normalize="index").round(3)
    log("\nShare of each guest type inside the existing market_segment labels:\n", seg_x.to_string())
    names_p = os.path.join(a.outdir, "segment_names.json")
    names = json.load(open(names_p)) if os.path.exists(names_p) else {}
    if names:
        wb = Workbook(); ws = wb.active; ws.title = "Guest types"
        ws["A1"] = "Guest types found in the booking data"; ws["A1"].font = Font(bold=True, size=14)
        ws["A2"] = ("Built from how and when guests booked and what they booked (lead time, stay, party, rate, board, channel, country, "
                    "history). Cancellation behaviour was NOT used to form the groups; it is shown afterwards as a finding.")
        ws["A2"].alignment = Alignment(wrap_text=True, vertical="top"); ws.merge_cells("A2:H2"); ws.row_dimensions[2].height = 34
        fields = [("segment_name", "Guest type"), ("description", "Who they are"), ("share_of_bookings", "Share of bookings"),
                  ("share_of_revenue", "Share of revenue"), ("city_share", "City Hotel share"), ("median_lead_days", "Median days booked ahead"),
                  ("median_nights", "Median nights"), ("median_adr", "Median rate (EUR/night)"), ("with_children_share", "Travel with children"),
                  ("repeat_guest_share", "Repeat guests"), ("top_market_segment", "Main market segment"), ("top_segment_share", "...share of it"),
                  ("top_channel", "Main channel"), ("top_country", "Main country"), ("cancel_rate", "Cancel / no-show rate"),
                  ("cancel_rate_2017", "...in 2017"), ("nonrefund_share", "Non-refundable deposit"), ("marketing_idea", "Marketing idea")]
        for j, (_, lab) in enumerate(fields, 1):
            c = ws.cell(4, j, lab); c.font = Font(color="FFFFFF", bold=True); c.fill = PatternFill("solid", fgColor="1F3A5F")
            c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        ws.row_dimensions[4].height = 48
        for i, r in enumerate(prof.itertuples(index=False), 5):
            rd = r._asdict(); nm = names.get(str(rd["segment"]), {})
            rd["segment_name"] = nm.get("name", f"Segment {rd['segment']}"); rd["description"] = nm.get("who", ""); rd["marketing_idea"] = nm.get("idea", "")
            for j, (key_, _) in enumerate(fields, 1):
                v = rd[key_]; c = ws.cell(i, j, round(v, 3) if isinstance(v, float) else v)
                c.alignment = Alignment(wrap_text=True, vertical="top")
                if key_ in ("share_of_bookings", "share_of_revenue", "city_share", "with_children_share", "repeat_guest_share",
                            "top_segment_share", "cancel_rate", "cancel_rate_2017", "nonrefund_share"):
                    c.number_format = "0%"
            ws.row_dimensions[i].height = 90
        for j, w in enumerate([24, 46, 11, 11, 11, 12, 10, 12, 12, 11, 16, 11, 11, 10, 12, 10, 12, 52], 1):
            ws.column_dimensions[get_column_letter(j)].width = w
        ws.freeze_panes = "C5"
        ws2 = wb.create_sheet("How it was built")
        for i, t_ in enumerate([
                f"{len(base):,} bookings arriving before 2017 were used to form {k} groups (k-means on standardised stay and booking "
                f"characteristics plus channel, segment, customer type, board, hotel and top-10 countries); all {len(df):,} bookings were then assigned.",
                "Chosen k from 5-8 by silhouette plus stability (agreement between two independent half-sample fits). "
                + "; ".join(f"k={r[0]}: sil {r[1]:.2f}, stability {r[2]:.2f}" for r in res),
                "Not used to form groups: cancellation, no-show, deposit type, special requests, parking, booking changes. They are profiled afterwards.",
                "Rate (ADR) is a final value and can be amended; stay length and party size are final values too (bin 2, amendable).",
                "These are statistical groups, not labelled truth. Treat names as hypotheses to test in campaigns."], 1):
            ws2.cell(i, 1, t_).alignment = Alignment(wrap_text=True, vertical="top"); ws2.row_dimensions[i].height = 48
        ws2.column_dimensions["A"].width = 120
        wb.save(os.path.join(a.outdir, "guest_types.xlsx")); log("wrote guest_types.xlsx")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["fit", "prove", "table", "segments"])
    ap.add_argument("--data", default="hotels.csv")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--asof")
    ap.add_argument("--capacity")
    a = ap.parse_args()
    {"fit": stage_fit, "prove": stage_prove, "table": stage_table, "segments": stage_segments}[a.stage](a)


if __name__ == "__main__":
    main()
