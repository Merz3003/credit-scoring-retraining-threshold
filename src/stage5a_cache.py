"""
Этап 5a: предрасчёт моделей и предсказаний.

Модель, обученная в месяц i, одинакова для всех стратегий - различается только
то, в какие месяцы происходит переобучение. Поэтому достаточно обучить по одной
модели на каждый месяц тестового периода и закэшировать предсказания.

Выход: cache_tw{N}.npz - предсказания, сигналы и денежные результаты
"""

import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from paths import PANEL, REJECTED, cache, require

TRAIN_MONTHS = int(sys.argv[1]) if len(sys.argv) > 1 else 24
TEST_START, TEST_END = "2014-01", "2015-12"

SPARSE = ["mths_since_last_record", "mths_since_last_major_derog", "mths_since_last_delinq"]
CATS = ["emp_length", "home_ownership", "verification_status", "purpose",
        "addr_state", "application_type"]
DROP = ["funded_amnt", "int_rate", "total_rec_prncp", "total_rec_int", "total_pymnt",
        "recoveries", "collection_recovery_fee", "default", "default_12m", "observable_at",
        "ead", "lgd", "cash_result", "id", "issue_d", "loan_status", "term",
        "earliest_cr_line", "last_pymnt_d", "issue", "ym"]
BINS, LABELS = (0, 12, 36, np.inf), ["<1 года", "1-3 года", ">3 лет"]


def prep(df):
    X = df.drop(columns=[c for c in DROP if c in df.columns])
    for col in SPARSE:
        b = pd.cut(X[col], bins=BINS, labels=LABELS)
        X[col] = b.cat.add_categories("не было").fillna("не было").astype(str)
    return pd.get_dummies(X, columns=CATS + SPARSE, dummy_na=True)


def main():
    df = pd.read_parquet(require(PANEL))
    rej = pd.read_pickle(require(REJECTED))
    months = [m for m in sorted(df["ym"].unique()) if TEST_START <= m <= TEST_END]
    T = len(months)

    Xall = prep(df)
    cols = Xall.columns
    bins_appl = np.histogram_bin_edges(rej["Amount Requested"].dropna().clip(0, 40000), bins=10)

    idx_month = {m: np.where(df["ym"].values == m)[0] for m in months}
    cash = df["cash_result"].values
    y12 = df["default_12m"].values

    # Предсказания модели i на месяце j (j >= i) и на наблюдаемой выборке месяца j
    pred = np.full((T, T), None, dtype=object)
    calib = np.full((T, T), np.nan)
    edges = [None] * T
    ref_score = [None] * T
    ref_appl = [None] * T

    for i, ym in enumerate(months):
        pool_mask = (df["observable_at"] <= ym).values
        avail = sorted(df.loc[pool_mask, "ym"].unique())[-TRAIN_MONTHS:]
        tr_mask = pool_mask & df["ym"].isin(avail).values

        Xtr = Xall[tr_mask]
        med = Xtr.median().fillna(0)
        sc = StandardScaler().fit(Xtr.fillna(med))
        mdl = LogisticRegression(max_iter=3000, C=0.1).fit(sc.transform(Xtr.fillna(med)), y12[tr_mask])

        p_tr = mdl.predict_proba(sc.transform(Xtr.fillna(med)))[:, 1]
        edges[i] = np.concatenate([[-np.inf], np.quantile(p_tr, np.linspace(0.1, 0.9, 9)), [np.inf]])
        ref_score[i] = np.histogram(p_tr, bins=edges[i])[0]
        ref_appl[i] = np.histogram(rej.loc[rej["ym"].isin(avail), "Amount Requested"].dropna(),
                                   bins=bins_appl)[0]
        # квантили для порога отсечения при разных долях одобрения
        if i == 0:
            qgrid = {}
        qgrid[i] = {ar: float(np.quantile(p_tr, ar)) for ar in (0.6, 0.7, 0.8, 0.9)}

        for j in range(i, T):
            Xj = Xall.iloc[idx_month[months[j]]]
            pred[i, j] = mdl.predict_proba(sc.transform(Xj.fillna(med)))[:, 1]
            # калибровочный разрыв на наблюдаемой выборке месяца j
            pm = (df["observable_at"] <= months[j]).values
            last6 = sorted(df.loc[pm, "ym"].unique())[-6:]
            om = pm & df["ym"].isin(last6).values
            Xo = Xall[om]
            po = mdl.predict_proba(sc.transform(Xo.fillna(med)))[:, 1]
            calib[i, j] = abs(y12[om].mean() - po.mean()) / max(po.mean(), 1e-6)
        print(f"  модель {ym}: обучение {tr_mask.sum():,} строк ({avail[0]}…{avail[-1]})", flush=True)

    cur_appl = np.array([np.histogram(rej.loc[rej["ym"] == m, "Amount Requested"].dropna(),
                                      bins=bins_appl)[0] for m in months])

    np.savez_compressed(
        cache(TRAIN_MONTHS),
        months=np.array(months), calib=calib, cur_appl=cur_appl,
        ref_appl=np.array(ref_appl), ref_score=np.array(ref_score),
        edges=np.array(edges),
        qgrid=np.array([[qgrid[i][ar] for ar in (0.6, 0.7, 0.8, 0.9)] for i in range(T)]),
        pred=np.array([[pred[i, j] if j >= i else np.array([]) for j in range(T)]
                       for i in range(T)], dtype=object),
        cash_by_month=np.array([cash[idx_month[m]] for m in months], dtype=object),
        allow_pickle=True,
    )
    print(f"кэш сохранён: cache_tw{TRAIN_MONTHS}.npz")


if __name__ == "__main__":
    main()
