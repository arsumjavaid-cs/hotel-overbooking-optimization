#!/usr/bin/env python3
"""
Overbooking pipeline: how many rooms to oversell, per hotel, for each of the next 14 nights.

Stages (run in this order; see README.md):
  fit    train on arrivals before 2016-07-01, choose settings on Jul-Dec 2016, freeze them
  prove  evaluate the frozen rule ONCE on Jan-Aug 2017 against the flat rule (EUR and rooms)
  table  build the morning table (xlsx) for an as-of date

Information rules (the "bins"):
  Model features (bin 1, known and fixed at booking):
      hotel, lead_time, arrival month / week number / day of month / weekday,
      country, market_segment, distribution_channel, is_repeated_guest,
      previous_cancellations, previous_bookings_not_canceled, reserved_room_type,
      agent, company (present or not), customer_type
      + two derived values: k (days to arrival on the morning of the decision) and
        age (days the booking has already been on the books = lead_time - k).
  Not used as features:
      arrival_date_year (cannot extrapolate to a new year), deposit_type (failed the
      leakage test: 99% cancel rate for Non Refund), every bin-2 and bin-3 column.
  Bin-2 / bin-3 columns used ONLY as bookkeeping, never as predictors:
      stays_in_*_nights  (bin 2) - to place a booking on the nights it covers.
          Caveat: the file holds the FINAL length of stay; ~15% of bookings were amended.
      reservation_status / reservation_status_date (bin 3) - to reconstruct what the hotel
          already knew on a past morning (a booking cancelled before that morning is not
          on the books), and as the label when scoring. The PMS knows past cancellations
          on the day they happen, so this is not a leak.
      adr (bin 2) - the value of an empty room, as a monthly hotel average from training.
"""
import argparse, json, os, pickle, sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- settings
TRAIN_END = pd.Timestamp("2016-07-01")      # train on arrivals before this date
VAL_START = pd.Timestamp("2016-07-15")      # first validation night (gives bookings a 2-week buffer)
VAL_END = pd.Timestamp("2017-01-01")        # validation = Jul-Dec 2016
TEST_START = pd.Timestamp("2017-01-01")
TEST_END = pd.Timestamp("2017-08-31")       # test = Jan-Aug 2017
WALK_COST = 300.0                           # EUR per guest we cannot house (from the brief)
FLAT_SAFETY = 3                             # "a few rooms for safety" in the flat rule (assumed 3)
N_H = 14                                    # decision horizons 0..13 days out (tonight + 13 nights)
PRESSURE = 0.80                             # score only decisions where bookings on the books >= 80% of capacity
O_MAX = 150                                 # search limit for rooms to oversell
LAMBDAS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]              # variance inflation candidates (chosen on validation)
CONFIGS = [
    dict(max_leaf_nodes=15, learning_rate=0.10, max_iter=150, l2_regularization=1.0, min_samples_leaf=200),
    dict(max_leaf_nodes=31, learning_rate=0.08, max_iter=200, l2_regularization=1.0, min_samples_leaf=100),
    dict(max_leaf_nodes=63, learning_rate=0.05, max_iter=250, l2_regularization=3.0, min_samples_leaf=100),
    dict(max_leaf_nodes=15, learning_rate=0.05, max_iter=300, l2_regularization=5.0, min_samples_leaf=300),
]
MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July",
                                      "August", "September", "October", "November", "December"], 1)}
CATS = ["hotel", "country_f", "market_segment", "distribution_channel", "reserved_room_type",
        "customer_type", "agent_f"]
NUMS = ["lead_time", "month", "week", "dom", "dow", "is_repeated_guest", "previous_cancellations",
        "previous_bookings_not_canceled", "has_company", "k", "age"]
HOTELS = ["City Hotel", "Resort Hotel"]


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- data
def load(path):
    df = pd.read_csv(path)
    df["arr"] = pd.to_datetime(dict(year=df.arrival_date_year, month=df.arrival_date_month.map(MONTHS),
                                    day=df.arrival_date_day_of_month))
    df["rsd"] = pd.to_datetime(df.reservation_status_date)
    df["n"] = (df.stays_in_weekend_nights + df.stays_in_week_nights).astype(int)
    df["dbc"] = (df.arr - df.rsd).dt.days                     # days before arrival the status was set
    df["cancelled"] = (df.reservation_status == "Canceled")
    df["y"] = df.is_canceled.astype(int)                      # 1 = cancelled or no-show
    df["month"] = df.arr.dt.month
    df["week"] = df.arrival_date_week_number
    df["dom"] = df.arrival_date_day_of_month
    df["dow"] = df.arr.dt.dayofweek
    df["has_company"] = df.company.notna().astype(int)
    df["agent_raw"] = df.agent.fillna(-1).astype(int)
    return df.reset_index(drop=True)


def build_cats(tr):
    cats = {"country": list(tr.country.fillna("NA").value_counts().head(20).index) ,
            "agent": [int(a) for a in tr.agent_raw.value_counts().head(30).index]}
    cats["country"] = sorted(set(cats["country"]) | {"OTH", "NA"})
    cats["agent"] = sorted(set(cats["agent"]) | {-1, -2})
    cats["hotel"] = HOTELS
    cats["country_f"] = cats["country"]
    cats["agent_f"] = [str(a) for a in cats["agent"]]
    for c in ["market_segment", "distribution_channel", "reserved_room_type", "customer_type"]:
        cats[c] = sorted(tr[c].astype(str).unique())
    return cats


def make_X(d, k, cats):
    X = pd.DataFrame(index=d.index)
    X["hotel"] = d.hotel
    c = d.country.fillna("NA")
    X["country_f"] = c.where(c.isin(cats["country"]), "OTH")
    a = d.agent_raw
    X["agent_f"] = a.where(a.isin(cats["agent"]), -2).astype(str)
    for col in ["market_segment", "distribution_channel", "reserved_room_type", "customer_type"]:
        X[col] = d[col].astype(str)
    for col in ["lead_time", "month", "week", "dom", "dow", "is_repeated_guest", "previous_cancellations",
                "previous_bookings_not_canceled", "has_company"]:
        X[col] = d[col].values
    X["k"] = k
    X["age"] = d.lead_time.values - k
    for col in CATS:
        X[col] = pd.Categorical(X[col], categories=cats[col])
    return X[CATS + NUMS]


def live_mask(d, k):
    """On the morning k days before arrival: booked earlier than that morning and not already cancelled."""
    return (d.lead_time.values > k) & ~(d.cancelled.values & (d.dbc.values > k))


def stack(d, cats):
    Xs, ys, hs, ks = [], [], [], []
    for k in range(N_H):
        m = live_mask(d, k)
        if m.any():
            s = d[m]
            Xs.append(make_X(s, k, cats)); ys.append(s.y.values); hs.append(s.hotel.values)
            ks.append(np.full(m.sum(), k))
    return pd.concat(Xs), np.concatenate(ys), np.concatenate(hs), np.concatenate(ks)


def expit(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def adjust(p, hotel, delta):
    return expit(logit(p) + np.array([delta[h] for h in hotel]))


def predict_matrix(model, d, cats, delta):
    """Live flags L and cancel probabilities P for every booking at every k = 0..13."""
    L = np.zeros((len(d), N_H), bool)
    P = np.full((len(d), N_H), np.nan)
    for k in range(N_H):
        m = live_mask(d, k)
        L[:, k] = m
        if model is not None and m.any():
            s = d[m]
            P[m, k] = adjust(model.predict_proba(make_X(s, k, cats))[:, 1], s.hotel.values, delta)
    return L, P


# ----------------------------------------------------------------------------- nights
def night_frame(S, L, P, d0, d1):
    """One row per (hotel, night, h): bookings on the books B, already in-house, expected further
    cancellations E, their variance V, realised further cancellations C. h = days between the morning
    of the decision and the night."""
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
    """Rooms occupied per night by guests who checked out (one booking = one room)."""
    S = S[(S.y == 0) & (S.n > 0)]
    days = (d1 - d0).days + 1
    out = {}
    for h, g in S.groupby("hotel"):
        diff = np.zeros(days + 2, int)
        a = (g.arr - d0).dt.days.values
        for s, n in zip(a, g.n.values):
            lo, hi = max(s, 0), min(s + n, days)
            if hi > lo:
                diff[lo] += 1; diff[hi] -= 1
        out[h] = pd.Series(np.cumsum(diff)[:days], index=pd.date_range(d0, periods=days))
    return out


# ----------------------------------------------------------------------------- rules
def model_oversell(f, v, cap, adr, lam):
    """Rooms to oversell that minimise expected cost: EUR adr per empty room + EUR 300 per guest walked.
    Selling cap+O rooms, cancellations C ~ Normal((cap+O)*f, (cap+O)*v*lam^2)  [f, v per booking]."""
    O = np.arange(O_MAX + 1)[None, :]
    tot = np.asarray(cap)[:, None] + O
    m = tot * f[:, None] - O
    s = np.sqrt(np.maximum(tot * v[:, None] * lam ** 2, 1e-9))
    z = m / s
    ez = s * norm.pdf(z) + m * norm.cdf(z)               # E[(C-O)+]  = expected empty rooms
    cost = adr[:, None] * ez + WALK_COST * (ez - m)      # (ez - m) = E[(O-C)+] = expected walks
    return cost.argmin(1)


def risk_columns(O, f, v, cap, lam):
    tot = np.asarray(cap) + O
    m = tot * f - O
    s = np.sqrt(np.maximum(tot * v * lam ** 2, 1e-9))
    z = m / s
    ez = s * norm.pdf(z) + m * norm.cdf(z)
    p_walk = norm.cdf((O - 0.5 - tot * f) / s)
    return p_walk, ez - m, ez


def realised(O, f_real, cap, adr):
    """Sell cap+O rooms to bookings that behave like the night's real bookings."""
    arrivals = (cap + O) * (1 - f_real)
    walk = np.maximum(0.0, arrivals - cap)
    empty = np.maximum(0.0, cap - arrivals)
    return walk, empty, adr * empty + WALK_COST * walk


def flat_oversell(B, rate, safety=FLAT_SAFETY):
    return np.maximum(0, np.rint(rate * B).astype(int) - safety)


def attach_context(nf, bundle):
    nf = nf.copy()
    nf["cap"] = nf.hotel.map(bundle["cap"])
    nf["adr"] = [bundle["adr"].get((h, m), bundle["adr_fallback"][h]) for h, m in zip(nf.hotel, nf.night.dt.month)]
    nf["f_hat"] = np.where(nf.B > 0, nf.E / nf.B.clip(lower=1), 0.0)
    nf["v_hat"] = np.where(nf.B > 0, nf.V / nf.B.clip(lower=1), 0.0)
    nf["f_real"] = np.where(nf.B > 0, nf.C / nf.B.clip(lower=1), 0.0)
    nf["tight"] = nf.B >= PRESSURE * nf.cap
    return nf


def apply_rules(nf, bundle, lam):
    nf = nf.copy()
    nf["O_model"] = model_oversell(nf.f_hat.values, nf.v_hat.values, nf.cap.values, nf.adr.values, lam)
    rate = nf.hotel.map(bundle["flat_rate"]).values
    nf["O_flat"] = flat_oversell(nf.B.values, rate)
    rate_h = np.array([bundle["flat_rate_h"][(h, hh)] for h, hh in zip(nf.hotel, nf.h)])
    nf["O_flat_h"] = flat_oversell(nf.B.values, rate_h)
    nf["O_none"] = 0
    nf["O_oracle"] = np.rint(nf.cap * nf.f_real / (1 - nf.f_real).clip(lower=1e-9)).astype(int)
    for r in ["model", "flat", "flat_h", "none"]:
        w, e, c = realised(nf[f"O_{r}"].values, nf.f_real.values, nf.cap.values, nf.adr.values)
        nf[f"walk_{r}"], nf[f"empty_{r}"], nf[f"cost_{r}"] = w, e, c
    return nf


def total_cost(nf, rules=("model",)):
    t = nf[nf.tight]
    return {r: float(t[f"cost_{r}"].sum() / N_H) for r in rules}


# ----------------------------------------------------------------------------- stage: fit
def stage_fit(a):
    os.makedirs(a.outdir, exist_ok=True)
    df = load(a.data)
    tr = df[df.arr < TRAIN_END].copy()
    va = df[(df.arr >= TRAIN_END) & (df.arr < VAL_END)].copy()
    log(f"train bookings {len(tr):,} (arrivals < {TRAIN_END.date()}), validation bookings {len(va):,}")
    cats = build_cats(tr)

    Xtr, ytr, _, _ = stack(tr, cats)
    Xva, yva, hva, kva = stack(va, cats)
    log(f"stacked rows: train {len(Xtr):,}, validation {len(Xva):,}; train cancel rate {ytr.mean():.3f}, val {yva.mean():.3f}")

    best = None
    for i, cfg in enumerate(CONFIGS):
        m = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0, **cfg)
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xva)[:, 1]
        ll, auc = log_loss(yva, p), roc_auc_score(yva, p)
        log(f"  config {i}: validation logloss {ll:.4f}  AUC {auc:.4f}")
        if best is None or ll < best[0]:
            best = (ll, i, m, p)
    ll, ci, model, pva = best
    log(f"chosen config {ci} {CONFIGS[ci]}")

    # settings choice 2: log-odds shift per hotel so mean predicted cancel rate = mean actual on validation
    delta = {}
    for h in HOTELS:
        mk = hva == h
        lo, hi = -2.0, 2.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if expit(logit(pva[mk]) + mid).mean() > yva[mk].mean():
                hi = mid
            else:
                lo = mid
        delta[h] = (lo + hi) / 2
    log("log-odds shift per hotel (drift correction fitted on validation):", {h: round(v, 3) for h, v in delta.items()})
    padj = adjust(pva, hva, delta)
    log(f"validation after shift: logloss {log_loss(yva, padj):.4f}  AUC {roc_auc_score(yva, padj):.4f}")

    # capacity estimate from nightly occupancy before 2017 (robust plateau: 99.5th percentile)
    occ = occupancy(df[df.arr < VAL_END], pd.Timestamp("2015-07-01"), pd.Timestamp("2016-12-31"))
    cap = {h: int(round(np.percentile(occ[h]["2015-09-01":"2016-12-31"], 99.5))) for h in HOTELS}
    log("estimated room capacity (override in settings.json if you know the real number):", cap)

    # monthly ADR from training checked-out stays (value of an empty room)
    ok = tr[(tr.y == 0) & (tr.adr > 0)]
    adr = {(h, int(m)): float(v) for (h, m), v in ok.groupby(["hotel", "month"]).adr.mean().items()}
    adr_fb = {h: float(ok[ok.hotel == h].adr.mean()) for h in HOTELS}

    # flat-rule rates from training: hotel average cancellation rate, and horizon-aware version
    flat_rate = {h: float(tr[tr.hotel == h].y.mean()) for h in HOTELS}
    trS = tr[tr.n > 0].reset_index(drop=True)
    Ltr = np.stack([live_mask(trS, k) for k in range(N_H)], 1)
    nf_tr = night_frame(trS, Ltr, None, pd.Timestamp("2015-09-01"), pd.Timestamp("2016-06-30"))
    g = nf_tr.groupby(["hotel", "h"])[["C", "B"]].sum()
    flat_rate_h = {(h, int(hh)): float(r.C / r.B) for (h, hh), r in g.iterrows()}
    log("flat rule average cancellation rate (all bookings, training):", {h: round(v, 3) for h, v in flat_rate.items()})

    bundle = dict(model=model, cats=cats, delta=delta, cap=cap, adr=adr, adr_fallback=adr_fb,
                  flat_rate=flat_rate, flat_rate_h=flat_rate_h, config=CONFIGS[ci], lam=None)

    # settings choice 3: variance multiplier lambda, by validation euros on tight decisions
    vaS = df[(df.n > 0) & (df.arr >= pd.Timestamp("2016-05-01")) & (df.arr < VAL_END)].reset_index(drop=True)
    Lva, Pva = predict_matrix(model, vaS, cats, delta)
    nf = attach_context(night_frame(vaS, Lva, Pva, VAL_START, pd.Timestamp("2016-12-31")), bundle)
    res = {}
    for lam in LAMBDAS:
        r = apply_rules(nf, bundle, lam)
        res[lam] = total_cost(r, ("model", "flat", "flat_h", "none"))
        log(f"  lambda {lam}: validation cost/season-basis EUR model {res[lam]['model']:,.0f}")
    lam = min(LAMBDAS, key=lambda x: res[x]["model"])
    bundle["lam"] = lam
    log(f"chosen lambda {lam}")
    r = apply_rules(nf, bundle, lam)
    t = r[r.tight]
    val = {"decisions_scored": int(len(t)), "nights": int(t.groupby(["hotel", "night"]).ngroups),
           "cost_EUR": total_cost(r, ("model", "flat", "flat_h", "none")),
           "walk_rooms": {k: float(t[f"walk_{k}"].sum() / N_H) for k in ["model", "flat", "flat_h", "none"]},
           "empty_rooms": {k: float(t[f"empty_{k}"].sum() / N_H) for k in ["model", "flat", "flat_h", "none"]},
           "predicted_vs_actual_cancels_on_tight_decisions": {"predicted": float(t.E.sum()), "actual": float(t.C.sum())}}
    log("VALIDATION (Jul 15 - Dec 2016, tight decisions):", json.dumps(val, indent=1))

    with open(os.path.join(a.outdir, "bundle.pkl"), "wb") as f:
        pickle.dump(bundle, f)
    settings = {"train_arrivals_before": str(TRAIN_END.date()), "validation_nights": "2016-07-15..2016-12-31",
                "gbm_config": CONFIGS[ci], "log_odds_shift": delta, "lambda": lam, "capacity_estimate": cap,
                "walk_cost_eur": WALK_COST, "flat_rule_safety_rooms": FLAT_SAFETY, "tight_threshold": PRESSURE,
                "flat_rule_avg_cancel_rate": flat_rate, "validation": val}
    with open(os.path.join(a.outdir, "settings.json"), "w") as f:
        json.dump(settings, f, indent=1)
    log("saved bundle.pkl and settings.json")


# ----------------------------------------------------------------------------- stage: prove
def stage_prove(a):
    with open(os.path.join(a.outdir, "bundle.pkl"), "rb") as f:
        bundle = pickle.load(f)
    df = load(a.data)
    S = df[(df.n > 0) & (df.arr >= pd.Timestamp("2016-11-01")) & (df.arr <= TEST_END)].reset_index(drop=True)
    L, P = predict_matrix(bundle["model"], S, bundle["cats"], bundle["delta"])
    nf = attach_context(night_frame(S, L, P, TEST_START, TEST_END), bundle)
    nf = apply_rules(nf, bundle, bundle["lam"])
    nf.to_csv(os.path.join(a.outdir, "backtest_detail_2017.csv"), index=False)
    t = nf[nf.tight].copy()
    rules = {"model": "New rule (model)", "flat": "Flat rule as written", "flat_h": "Flat rule, fairer version",
             "none": "No overselling"}

    def block(d, extra=None):
        rows = []
        for r, name in rules.items():
            rows.append({"rule": name, "cost_EUR": d[f"cost_{r}"].sum() / N_H, "rooms_walked": d[f"walk_{r}"].sum() / N_H,
                         "rooms_empty": d[f"empty_{r}"].sum() / N_H,
                         "avg_oversell_rooms": d[f"O_{r}"].mean()})
        return pd.DataFrame(rows)

    overall = block(t)
    by_hotel = {h: block(t[t.hotel == h]) for h in HOTELS}
    by_h = {}
    for hh in [0, 1, 3, 7, 13]:
        x = t[t.h == hh]
        by_h[hh] = pd.DataFrame([{"rule": name, "cost_EUR": x[f"cost_{r}"].sum(), "rooms_walked": x[f"walk_{r}"].sum(),
                                  "rooms_empty": x[f"empty_{r}"].sum()} for r, name in rules.items()])
    t["month_"] = t.night.dt.to_period("M").astype(str)
    by_month = t.groupby("month_").apply(lambda g: pd.Series({
        "nights_scored": g.groupby(["hotel", "night"]).ngroups,
        "cost_model": g.cost_model.sum() / N_H, "cost_flat": g.cost_flat.sum() / N_H,
        "cost_flat_h": g.cost_flat_h.sum() / N_H})).reset_index()

    # paired bootstrap by week over nights: saving vs the two flat rules
    nn = t.groupby(["hotel", "night"]).agg(model=("cost_model", "sum"), flat=("cost_flat", "sum"),
                                           flat_h=("cost_flat_h", "sum")).reset_index()
    for c in ["model", "flat", "flat_h"]:
        nn[c] /= N_H
    nn["week"] = nn.night.dt.isocalendar().week.astype(int) + 100 * nn.night.dt.year
    wk = nn.groupby("week")[["model", "flat", "flat_h"]].sum()
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(4000):
        s = wk.iloc[rng.integers(0, len(wk), len(wk))].sum()
        boots.append([s.flat - s.model, s.flat_h - s.model])
    boots = np.array(boots)
    ci = {"saving_vs_flat_as_written": [float(wk.flat.sum() - wk.model.sum()), *np.percentile(boots[:, 0], [2.5, 97.5]).tolist()],
          "saving_vs_flat_fairer": [float(wk.flat_h.sum() - wk.model.sum()), *np.percentile(boots[:, 1], [2.5, 97.5]).tolist()]}
    share_better = {"vs_flat_as_written": float((nn.model < nn.flat).mean()), "vs_flat_fairer": float((nn.model < nn.flat_h).mean())}

    diag = {}
    for hh in [0, 1, 3, 7, 13]:
        x = t[t.h == hh]
        diag[hh] = {"predicted_cancels": float(x.E.sum()), "actual_cancels": float(x.C.sum()), "decisions": int(len(x))}

    out = {"tight_decisions": int(len(t)), "nights_scored": int(len(nn)),
           "nights_in_window": int(nf.groupby(["hotel", "night"]).ngroups),
           "overall": overall.to_dict("records"), "by_hotel": {h: v.to_dict("records") for h, v in by_hotel.items()},
           "by_horizon": {str(k): v.to_dict("records") for k, v in by_h.items()},
           "by_month": by_month.to_dict("records"), "bootstrap_95ci_EUR": ci, "share_of_nights_model_cheaper": share_better,
           "calibration_by_horizon": diag}
    with open(os.path.join(a.outdir, "proof_2017.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    pd.set_option("display.width", 200)
    log(f"\nTEST Jan-Aug 2017: {len(t):,} tight decisions over {len(nn)} hotel-nights (of {out['nights_in_window']} hotel-nights)")
    log("\nOVERALL (EUR / rooms are per-night-average over the 14 daily decisions)\n", overall.round(1).to_string(index=False))
    for h, v in by_hotel.items():
        log(f"\n{h}\n", v.round(1).to_string(index=False))
    for k, v in by_h.items():
        log(f"\nSingle decision {k} day(s) out (sum over nights)\n", v.round(0).to_string(index=False))
    log("\nBy month\n", by_month.round(0).to_string(index=False))
    log("\nBootstrap 95% CI of saving (EUR): ", json.dumps(ci))
    log("share of hotel-nights where new rule is cheaper:", share_better)
    log("calibration (predicted vs actual further cancellations on tight decisions):", json.dumps(diag))


# ----------------------------------------------------------------------------- stage: table
def stage_table(a):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    with open(os.path.join(a.outdir, "bundle.pkl"), "rb") as f:
        bundle = pickle.load(f)
    for h, v in (json.loads(a.capacity) if a.capacity else {}).items():
        bundle["cap"][h] = int(v)
    asof = pd.Timestamp(a.asof)
    df = load(a.data)
    S = df[(df.n > 0) & (df.arr >= asof - pd.Timedelta(days=75)) & (df.arr <= asof + pd.Timedelta(days=N_H))].reset_index(drop=True)
    L, P = predict_matrix(bundle["model"], S, bundle["cats"], bundle["delta"])
    nf = night_frame(S, L, P, asof, asof + pd.Timedelta(days=N_H - 1))
    nf = nf[(nf.night - asof).dt.days == nf.h]
    nf = attach_context(nf, bundle)
    nf = apply_rules(nf, bundle, bundle["lam"])
    O = nf.O_model.values
    nf["p_walk"], nf["exp_walk"], nf["exp_empty"] = risk_columns(O, nf.f_hat.values, nf.v_hat.values, nf.cap.values, bundle["lam"])
    nf["limit"] = nf.cap + nf.O_model
    nf["can_sell"] = nf.limit - nf.B
    nf = nf.sort_values(["hotel", "night"])

    wb = Workbook()
    ws = wb.active
    ws.title = "Morning table"
    head = PatternFill("solid", fgColor="1F3A5F"); white = Font(color="FFFFFF", bold=True)
    key = PatternFill("solid", fgColor="FFF2CC"); tight_fill = PatternFill("solid", fgColor="FCE4D6")
    thin = Side(style="thin", color="BBBBBB")
    ws["A1"] = f"Rooms to oversell, next {N_H} nights   |   data as of the morning of {asof.date()}"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A2"] = ("Booking limit = capacity + rooms to oversell. 'Can still sell' = limit minus bookings on the books "
                "(negative = stop selling). Shaded rows: bookings already >= 80% of capacity, so the limit is binding.")
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("A2:L2"); ws.row_dimensions[2].height = 34
    ws["A3"] = ("STATUS: Jan-Aug 2017 test - Resort figures beat 'no overselling'; City figures did NOT (City late cancellations "
                "fell sharply in 2017, so the rule oversold too much). Do not act on City numbers until recalibrated. See 'Proof' sheet.")
    ws["A3"].font = Font(bold=True, color="C00000"); ws["A3"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("A3:L3"); ws.row_dimensions[3].height = 34
    cols = ["Hotel", "Night", "Day", "Days out", "Rooms (est.)", "Bookings on the books", "Expected cancels still to come",
            "ROOMS TO OVERSELL", "Booking limit", "Can still sell", "Chance we walk a guest", "Flat rule says (for comparison)"]
    r0 = 4
    for j, c in enumerate(cols, 1):
        x = ws.cell(r0, j, c); x.fill = head; x.font = white
        x.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
    ws.row_dimensions[r0].height = 48
    for i, r in enumerate(nf.itertuples(), r0 + 1):
        vals = [r.hotel, r.night.date(), r.night.strftime("%a"), int(r.h), int(r.cap), int(r.B), round(float(r.E), 1),
                int(r.O_model), int(r.limit), int(r.can_sell), float(r.p_walk), int(r.O_flat)]
        for j, v in enumerate(vals, 1):
            c = ws.cell(i, j, v)
            c.border = Border(top=thin, bottom=thin, left=thin, right=thin)
            c.alignment = Alignment(horizontal="center")
            if r.tight:
                c.fill = tight_fill
            if j == 8:
                c.fill = key; c.font = Font(bold=True)
            if j == 11:
                c.number_format = "0%"
            if j == 2:
                c.number_format = "yyyy-mm-dd"
    for j, w in enumerate([14, 12, 6, 8, 10, 12, 14, 13, 10, 10, 12, 16], 1):
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.freeze_panes = "A5"

    # proof sheet
    pp = os.path.join(a.outdir, "proof_2017.json")
    ws2 = wb.create_sheet("Proof Jan-Aug 2017")
    if os.path.exists(pp):
        pr = json.load(open(pp))
        ws2["A1"] = "What each rule would have cost us, Jan-Aug 2017 (rule frozen on Jul-Dec 2016, evaluated once)"
        ws2["A1"].font = Font(bold=True, size=13)
        ws2["A2"] = ("Cost = EUR per empty room (monthly hotel ADR) + EUR 300 per guest walked, measured against a perfect "
                     "hindsight policy. Scored on decisions where bookings were already >= 80% of capacity. Totals are the "
                     "average over the 14 daily decisions each night receives.")
        ws2["A2"].alignment = Alignment(wrap_text=True, vertical="top"); ws2.merge_cells("A2:F2"); ws2.row_dimensions[2].height = 48
        row = 4
        def put(title, recs, row):
            ws2.cell(row, 1, title).font = Font(bold=True)
            hdr = list(recs[0].keys())
            for j, hname in enumerate(hdr, 1):
                c = ws2.cell(row + 1, j, hname); c.fill = head; c.font = white
            for i, rec in enumerate(recs, row + 2):
                for j, hname in enumerate(hdr, 1):
                    v = rec[hname]; c = ws2.cell(i, j, round(v, 1) if isinstance(v, float) else v)
                    if hname.startswith("cost"):
                        c.number_format = "#,##0"
            return row + len(recs) + 3
        row = put("Both hotels", pr["overall"], row)
        for h, rec in pr["by_hotel"].items():
            row = put(h, rec, row)
        ci = pr["bootstrap_95ci_EUR"]
        ws2.cell(row, 1, "Saving vs flat rule as written (EUR, 95% CI by week)").font = Font(bold=True)
        for j, v in enumerate(ci["saving_vs_flat_as_written"], 2):
            ws2.cell(row, j, round(v))
        ws2.cell(row + 1, 1, "Saving vs fairer flat rule (EUR, 95% CI by week)").font = Font(bold=True)
        for j, v in enumerate(ci["saving_vs_flat_fairer"], 2):
            ws2.cell(row + 1, j, round(v))
        for j, w in enumerate([52, 16, 14, 14, 18], 1):
            ws2.column_dimensions[get_column_letter(j)].width = w
    else:
        ws2["A1"] = "Run the 'prove' stage to fill this sheet."

    ws3 = wb.create_sheet("Notes")
    notes = [
        "How to read: the model estimates, for each booking still on the books, the chance it cancels or no-shows before arrival "
        "given how many days are left. It adds these up per night, then picks the oversell that minimises expected cost "
        "(EUR 300 per walked guest vs the value of an empty room).",
        "Rooms (est.) are the plateau of nightly occupancy in 2015-2016 (99.5th percentile), not the real room count. "
        "Run with --capacity '{\"City Hotel\": N, \"Resort Hotel\": M}' to overwrite.",
        "The recommended oversell assumes extra bookings behave like the night's other bookings, and that enough demand exists "
        "to fill the limit. On quiet nights the number is a ceiling for when demand arrives, not a prediction of sales.",
        "Stay length is the final value in the file (bin 2); it only places bookings on nights and is not a predictor.",
        "The model is frozen on arrivals before 2016-07-01. Refit periodically; see README.md.",
    ]
    for i, t_ in enumerate(notes, 1):
        c = ws3.cell(i, 1, t_); c.alignment = Alignment(wrap_text=True, vertical="top"); ws3.row_dimensions[i].height = 60
    ws3.column_dimensions["A"].width = 110
    out = os.path.join(a.outdir, f"morning_table_{asof.date()}.xlsx")
    wb.save(out)
    nf[["hotel", "night", "h", "cap", "B", "E", "O_model", "limit", "can_sell", "p_walk", "O_flat", "tight"]].to_csv(
        os.path.join(a.outdir, f"morning_table_{asof.date()}.csv"), index=False)
    log("wrote", out)
    log(nf[["hotel", "night", "h", "cap", "B", "E", "O_model", "limit", "can_sell", "p_walk", "O_flat", "tight"]].round(2).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["fit", "prove", "table"])
    ap.add_argument("--data", default="hotels.csv")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--asof", help="morning date for the table, e.g. 2017-08-15")
    ap.add_argument("--capacity", help='JSON override of room counts, e.g. \'{"City Hotel": 226, "Resort Hotel": 185}\'')
    a = ap.parse_args()
    {"fit": stage_fit, "prove": stage_prove, "table": stage_table}[a.stage](a)


if __name__ == "__main__":
    main()
