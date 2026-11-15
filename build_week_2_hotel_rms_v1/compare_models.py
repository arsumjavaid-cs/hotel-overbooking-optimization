"""Compare model families on the SAME split as v2: train < 1 Jul 2016, judge on Jul-Dec 2016 (validation only).
Same feature set as v2 (bin 1 + mild bin 2). Euro result uses the v2 decision rule with v2's chosen settings,
per-night accounting (as in v2). The 2017 test window is NOT touched.
    python compare_models.py --data hotels.csv --ref outputs
"""
import argparse, importlib.util, json, os, pickle, sys, time
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("v2", os.path.join(here, "overbooking_pipeline_v2.py"))
v2 = importlib.util.module_from_spec(spec); sys.modules["v2"] = v2; spec.loader.exec_module(v2)


class Wrap:
    """Gives any sklearn model the .predict_proba(DataFrame) the v2 functions expect."""
    def __init__(self, est, kind, cat_cols, num_cols):
        self.est, self.kind, self.cat_cols, self.num_cols = est, kind, cat_cols, num_cols
    def prep(self, X):
        X = X.copy()
        for c in self.cat_cols:
            X[c] = X[c].cat.codes if self.kind == "codes" else X[c].astype(str)
        return X
    def fit(self, X, y):
        self.est.fit(self.prep(X), y); return self
    def predict_proba(self, X):
        return self.est.predict_proba(self.prep(X))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="hotels.csv"); ap.add_argument("--ref", default="outputs")
    ap.add_argument("--out", default="outputs_compare"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bundle = pickle.load(open(os.path.join(a.ref, "bundle_v2.pkl"), "rb"))
    fs, cats = bundle["fs"], bundle["cats"]
    df = v2.load(a.data)
    tr = df[df.arr < v2.TRAIN_END].copy()
    va = df[(df.arr >= v2.TRAIN_END) & (df.arr < v2.VAL_END)].copy()
    Xtr, ytr, _, _ = v2.stack(tr, cats, fs)
    Xva, yva, hva, kva = v2.stack(va, cats, fs)
    cat_cols = [c for c in Xtr.columns if str(Xtr[c].dtype) == "category"]
    num_cols = [c for c in Xtr.columns if c not in cat_cols]
    print(f"train rows {len(Xtr):,}  validation rows {len(Xva):,}  features {len(cat_cols)} cat + {len(num_cols)} num")

    lr = make_pipeline(ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore", min_frequency=50), cat_cols),
                                          ("n", StandardScaler(), num_cols)]),
                       LogisticRegression(max_iter=300, C=1.0))
    rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=50, max_features="sqrt", n_jobs=-1, random_state=0)
    models = {
        "Base rate only (no model)": None,
        "Logistic regression": Wrap(lr, "str", cat_cols, num_cols),
        "Random forest": Wrap(rf, "codes", cat_cols, num_cols),
        "Gradient boosting (v2)": bundle["model"],
    }
    st = json.load(open(os.path.join(a.ref, "settings_v2.json")))
    W = st["window_days"]; lam = st["lambda_"]; g = st["g"]
    vaS = df[(df.n > 0) & (df.arr >= pd.Timestamp("2016-05-01")) & (df.arr < v2.VAL_END)].reset_index(drop=True)
    dates = pd.date_range("2016-06-15", "2016-12-31")
    rows = []
    for name, m in models.items():
        t0 = time.time()
        if m is None:
            base = {h: ytr[(Xtr.hotel == h).values].mean() for h in v2.HOTELS}
            p = np.array([base[h] for h in hva])
            class Const:
                def predict_proba(self, X):
                    q = np.array([base[h] for h in X.hotel.astype(str)]); return np.c_[1 - q, q]
            mm = Const()
        else:
            if name != "Gradient boosting (v2)":
                m.fit(Xtr, ytr)
            mm = m
            p = mm.predict_proba(Xva)[:, 1]
        sh = v2.static_shift(p, yva, hva)
        padj = v2.expit(v2.logit(p) + np.array([sh[h] for h in hva]))
        rec = dict(model=name, auc=roc_auc_score(yva, p), logloss_raw=log_loss(yva, p), brier_raw=brier_score_loss(yva, p),
                   logloss_after_shift=log_loss(yva, padj))
        # euro result with v2's rolling shift + constrained rule at v2's chosen settings
        L, Praw = v2.raw_matrix(mm, vaS, cats, fs)
        rr = v2.raw_rows(mm, va, cats, fs)
        shift = v2.rolling_shift(rr, W, dates) if W else None
        P = v2.apply_shift(Praw, L, vaS, shift)
        nf = v2.add_baselines(v2.attach_context(v2.night_frame(vaS, L, P, v2.VAL_START, pd.Timestamp("2016-12-31")), bundle), bundle)
        O = v2.oversell(nf.f_hat.values, nf.v_hat.values, nf.cap.values, nf.adr.values, lam, g)
        r = v2.add_rule(nf[nf.tight].copy(), "m", O[nf.tight.values])
        rec.update(val_cost_eur=r.cost_m.sum() / v2.N_H, walks_per_1000_City=v2.walk_rate(r, "m", "City Hotel"),
                   walks_per_1000_Resort=v2.walk_rate(r, "m", "Resort Hotel"),
                   none_cost_eur=r.cost_none.sum() / v2.N_H, seconds=time.time() - t0)
        rows.append(rec); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()})
    out = pd.DataFrame(rows); out.to_csv(os.path.join(a.out, "model_comparison_validation.csv"), index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
