"""
Этап 4: сравнение стратегий переобучения на данных Lending Club.

Стратегии:
  S1  - без переобучения
  S2  - календарное, каждые 3 месяца
  S3  - календарное, каждые 6 месяцев
  S4a - триггер: PSI по распределению скора
  S4b - триггер: PSI по входящему потоку заявок (отклонённые + одобренные)
  S4c - триггер: расхождение калибровки на наблюдаемых когортах

Реализм:
  - целевая переменная - 12-месячный дефолт (определение Базеля);
  - исход когорты наблюдаем только через LAG месяцев после выдачи,
    поэтому переобучение использует лишь когорты с issue <= t - LAG;
  - порог отсечения фиксирует долю одобрения, пересчитывается при переобучении;
  - прибыль - фактический денежный результат одобренных кредитов.

Вход:  panel_ext.parquet, rej.pkl
Выход: strategies_monthly.csv, strategies_summary.csv
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from paths import PANEL as PANEL_PATH, REJECTED as REJ_PATH, RESULTS, require

PANEL = require(PANEL_PATH)
REJECTED = require(REJ_PATH)

TEST_START, TEST_END = "2014-01", "2015-12"
LAG = 18                 # месяцев до наблюдаемости исхода
TRAIN_MONTHS = 24        # длина обучающего окна
APPROVAL_RATE = 0.80     # доля одобрения, задаёт порог отсечения
RETRAIN_COST = 100_000   # базовое значение, варьируется в анализе чувствительности

PSI_SCORE_TH = 0.10
PSI_APPL_TH = 0.10
CALIB_TH = 0.15          # относительное расхождение предсказанной и фактической PD

SPARSE = ["mths_since_last_record", "mths_since_last_major_derog", "mths_since_last_delinq"]
CATS = ["emp_length", "home_ownership", "verification_status", "purpose",
        "addr_state", "application_type"]
DROP = ["funded_amnt", "int_rate", "total_rec_prncp", "total_rec_int", "total_pymnt",
        "recoveries", "collection_recovery_fee", "default", "default_12m", "observable_at",
        "ead", "lgd", "cash_result", "id", "issue_d", "loan_status", "term",
        "earliest_cr_line", "last_pymnt_d", "issue", "ym"]


class Scorecard:
    BINS = (0, 12, 36, np.inf)
    LABELS = ["<1 года", "1-3 года", ">3 лет"]

    def _prep(self, df):
        X = df.drop(columns=[c for c in DROP if c in df.columns])
        for col in SPARSE:
            b = pd.cut(X[col], bins=self.BINS, labels=self.LABELS)
            X[col] = b.cat.add_categories("не было").fillna("не было").astype(str)
        return pd.get_dummies(X, columns=CATS + SPARSE, dummy_na=True)

    def fit(self, df):
        X = self._prep(df)
        self.columns_ = X.columns
        self.median_ = X.median().fillna(0)   # поля, появившиеся позже, пусты в ранних окнах
        X = X.fillna(self.median_)
        self.scaler_ = StandardScaler().fit(X)
        self.model_ = LogisticRegression(max_iter=3000, C=0.1).fit(
            self.scaler_.transform(X), df["default_12m"].values)
        p = self.predict(df)
        self.cutoff_ = float(np.quantile(p, APPROVAL_RATE))
        self.score_ref_ = p
        self.edges_ = np.concatenate(
            [[-np.inf], np.quantile(p, np.linspace(0.1, 0.9, 9)), [np.inf]])
        return self

    def predict(self, df):
        X = self._prep(df).reindex(columns=self.columns_, fill_value=0)
        X = X.fillna(self.median_).fillna(0)
        return self.model_.predict_proba(self.scaler_.transform(X))[:, 1]


def psi(ref_counts, cur_counts):
    eps = 1e-6
    e = np.clip(ref_counts / ref_counts.sum(), eps, None)
    a = np.clip(cur_counts / cur_counts.sum(), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def applicant_dist(rej, months, bins):
    sub = rej[rej["ym"].isin(months)]
    return np.histogram(sub["Amount Requested"].dropna(), bins=bins)[0]


def run(strategy, df, rej, test_months, appl_bins):
    """Прогон одной стратегии по тестовым месяцам."""
    card, last_fit, n_retrain = None, None, 0
    rows = []

    for i, ym in enumerate(test_months):
        # Данные, доступные для обучения на момент ym.
        pool = df[df["observable_at"] <= ym]
        avail = sorted(pool["ym"].unique())[-TRAIN_MONTHS:]
        train = pool[pool["ym"].isin(avail)]

        need_fit = card is None
        if not need_fit:
            if strategy == "S2":
                need_fit = i % 3 == 0
            elif strategy == "S3":
                need_fit = i % 6 == 0
            elif strategy == "S4a":
                need_fit = rows[-1]["psi_score"] > PSI_SCORE_TH
            elif strategy == "S4b":
                need_fit = rows[-1]["psi_appl"] > PSI_APPL_TH
            elif strategy == "S4c":
                need_fit = rows[-1]["calib_gap"] > CALIB_TH

        if need_fit:
            card = Scorecard().fit(train)
            last_fit = avail
            if i > 0:
                n_retrain += 1
            ref_appl = applicant_dist(rej, avail, appl_bins)
            ref_appl = ref_appl if ref_appl.sum() else None

        m = df[df["ym"] == ym]
        p = card.predict(m)
        appr = p <= card.cutoff_
        y12, ylife = m["default_12m"].values, m["default"].values
        cash = m["cash_result"].values

        cur_score = np.histogram(p, bins=card.edges_)[0]
        ref_score = np.histogram(card.score_ref_, bins=card.edges_)[0]
        cur_appl = applicant_dist(rej, [ym], appl_bins)

        # Калибровка оценивается только на наблюдаемых когортах.
        obs = df[df["observable_at"] <= ym]
        obs = obs[obs["ym"].isin(sorted(obs["ym"].unique())[-6:])]
        gap = np.nan
        if len(obs) > 1000:
            p_obs = card.predict(obs)
            gap = abs(obs["default_12m"].mean() - p_obs.mean()) / max(p_obs.mean(), 1e-6)

        rows.append({
            "strategy": strategy, "ym": ym, "retrained": int(need_fit and i > 0),
            "train_from": last_fit[0], "train_to": last_fit[-1],
            "approval_rate": round(float(appr.mean()), 4),
            "auc": round(roc_auc_score(y12, p), 4) if y12.sum() else np.nan,
            "pd_pred": round(float(p[appr].mean()), 4),
            "pd_actual_12m": round(float(y12[appr].mean()), 4),
            "pd_actual_life": round(float(ylife[appr].mean()), 4),
            "psi_score": round(psi(ref_score, cur_score), 4),
            "psi_appl": round(psi(ref_appl, cur_appl), 4) if ref_appl is not None else np.nan,
            "calib_gap": round(float(gap), 4) if gap == gap else np.nan,
            "profit": round(float(cash[appr].sum()), 0),
            "n_approved": int(appr.sum()),
        })

    res = pd.DataFrame(rows)
    res["n_retrain_total"] = n_retrain
    return res


def main():
    df = pd.read_parquet(PANEL)
    rej = pd.read_pickle(REJECTED)
    test_months = [m for m in sorted(df["ym"].unique()) if TEST_START <= m <= TEST_END]
    appl_bins = np.histogram_bin_edges(
        rej["Amount Requested"].dropna().clip(0, 40000), bins=10)

    print(f"Тест: {test_months[0]}…{test_months[-1]} ({len(test_months)} мес.), "
          f"лаг наблюдаемости {LAG} мес., окно обучения {TRAIN_MONTHS} мес., "
          f"доля одобрения {APPROVAL_RATE:.0%}")

    out = pd.concat([run(s, df, rej, test_months, appl_bins)
                     for s in ["S1", "S2", "S3", "S4a", "S4b", "S4c"]], ignore_index=True)
    out.to_csv(RESULTS / "strategies_monthly.csv", index=False)

    g = out.groupby("strategy")
    summary = pd.DataFrame({
        "переобучений": g["retrained"].sum(),
        "прибыль_млн": (g["profit"].sum() / 1e6).round(1),
        "одобрено": g["n_approved"].sum(),
        "ср_AUC": g["auc"].mean().round(4),
        "ошибка_PD": (g["pd_actual_12m"].mean() - g["pd_pred"].mean()).round(4),
    })
    summary["затраты_млн"] = (summary["переобучений"] * RETRAIN_COST / 1e6).round(2)
    summary["итог_млн"] = (summary["прибыль_млн"] - summary["затраты_млн"]).round(1)
    summary["vs_S1_млн"] = (summary["итог_млн"] - summary.loc["S1", "итог_млн"]).round(1)
    summary.to_csv(RESULTS / "strategies_summary.csv")

    print("\n" + summary.to_string())

    print("\nСигналы триггеров (максимум за период):")
    for col, th in [("psi_score", PSI_SCORE_TH), ("psi_appl", PSI_APPL_TH), ("calib_gap", CALIB_TH)]:
        v = out[out["strategy"] == "S1"][col]
        print(f"  {col:11s} макс {v.max():.3f}  порог {th}  "
              f"превышений {int((v > th).sum())} из {len(v)}")


if __name__ == "__main__":
    main()
