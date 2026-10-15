"""
Этап 3: базовая модель, порог отсечения и walk-forward для стратегии S1.

Вход:  panel.parquet (результат build_panel.py)
Выход: s1_walkforward.csv - помесячные метрики стратегии "без переобучения"

Решения:
  - разреженные поля (mths_since_*) разбиваются на интервалы, отдельная категория "не было";
  - окно обучения 24 месяца;
  - модель - логистическая регрессия (классическая скоринговая карта).

Всё, что зависит от данных (границы интервалов, медианы, порог отсечения),
подбирается ТОЛЬКО на обучающем окне.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from paths import PANEL as PANEL_PATH, RESULTS, require

PANEL = require(PANEL_PATH)
OUT = RESULTS / "s1_walkforward.csv"

TRAIN_MONTHS = 24
SPARSE = ["mths_since_last_record", "mths_since_last_major_derog", "mths_since_last_delinq"]
CATS = ["emp_length", "home_ownership", "verification_status", "purpose",
        "addr_state", "application_type"]
ECON = ["funded_amnt", "int_rate", "total_rec_prncp", "total_rec_int",
        "total_pymnt", "recoveries", "collection_recovery_fee"]
DROP = ECON + ["default", "ead", "lgd", "cash_result", "id", "issue_d",
               "loan_status", "term", "earliest_cr_line", "issue", "ym"]


class Scorecard:
    """Препроцессинг и модель. Все параметры подбираются на обучающем окне."""

    def __init__(self, sparse_bins=(0, 12, 36, np.inf)):
        self.sparse_bins = sparse_bins

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.drop(columns=[c for c in DROP if c in df.columns])

        # Вариант C: интервалы. Пропуск = "события не было" и становится отдельной категорией.
        for col in SPARSE:
            binned = pd.cut(X[col], bins=self.sparse_bins,
                            labels=["<1 года", "1-3 года", ">3 лет"])
            X[col] = binned.cat.add_categories("не было").fillna("не было").astype(str)

        X = pd.get_dummies(X, columns=CATS + SPARSE, dummy_na=True)
        return X

    def fit(self, df: pd.DataFrame) -> "Scorecard":
        X = self._transform(df)
        self.columns_ = X.columns
        self.median_ = X.median()
        X = X.fillna(self.median_)
        self.scaler_ = StandardScaler().fit(X)
        self.model_ = LogisticRegression(max_iter=3000, C=0.1).fit(
            self.scaler_.transform(X), df["default"].values
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._transform(df).reindex(columns=self.columns_, fill_value=0)
        X = X.fillna(self.median_)
        return self.model_.predict_proba(self.scaler_.transform(X))[:, 1]


def pick_cutoff(pd_pred: np.ndarray, profit_paid: float, loss_default: float) -> float:
    """
    Порог отсечения, максимизирующий ожидаемую прибыль на обучающем окне.
    Заявка одобряется, если ожидаемая прибыль неотрицательна.
    """
    grid = np.quantile(pd_pred, np.linspace(0.05, 0.99, 95))
    best, best_profit = grid[0], -np.inf
    for c in grid:
        appr = pd_pred <= c
        if appr.sum() == 0:
            continue
        p = pd_pred[appr].mean()
        profit = appr.sum() * ((1 - p) * profit_paid - p * loss_default)
        if profit > best_profit:
            best, best_profit = c, profit
    return float(best)


def psi(expected: np.ndarray, actual: np.ndarray, edges: np.ndarray) -> float:
    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    eps = 1e-6
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def main() -> None:
    df = pd.read_parquet(PANEL).sort_values("ym").reset_index(drop=True)
    months = sorted(df["ym"].unique())
    train_months, test_months = months[:TRAIN_MONTHS], months[TRAIN_MONTHS:]

    train = df[df["ym"].isin(train_months)]
    print(f"Обучение: {train_months[0]}-{train_months[-1]}, {len(train):,} кредитов")
    print(f"Тест:     {test_months[0]}-{test_months[-1]}, {len(df) - len(train):,} кредитов")

    # Экономика на обучающем окне: средний результат исправного и дефолтного кредита.
    profit_paid = train.loc[train["default"] == 0, "cash_result"].mean()
    loss_default = -train.loc[train["default"] == 1, "cash_result"].mean()
    print(f"\nСредний результат: исправный +${profit_paid:,.0f}, дефолтный -${loss_default:,.0f}")

    card = Scorecard().fit(train)
    p_train = card.predict(train)
    print(f"AUC на обучающем окне: {roc_auc_score(train['default'], p_train):.4f}")

    cutoff = pick_cutoff(p_train, profit_paid, loss_default)
    edges = np.concatenate([[-np.inf], np.quantile(p_train, np.linspace(0.1, 0.9, 9)), [np.inf]])
    print(f"Порог отсечения: {cutoff:.4f} (одобрено на обучении: {(p_train <= cutoff).mean():.1%})")

    rows = []
    for ym in test_months:
        m = df[df["ym"] == ym]
        p = card.predict(m)
        appr = p <= cutoff
        n_appr = int(appr.sum())
        if n_appr == 0:
            continue
        y = m["default"].values
        rows.append({
            "ym": ym,
            "n_loans": len(m),
            "approval_rate": round(appr.mean(), 4),
            "auc": round(roc_auc_score(y, p), 4) if y.sum() else np.nan,
            "pd_pred": round(float(p[appr].mean()), 4),
            "pd_actual": round(float(y[appr].mean()), 4),
            "psi": round(psi(p_train, p, edges), 4),
            # фактический денежный результат одобренных кредитов
            "profit_actual": round(float(m.loc[appr, "cash_result"].sum()), 0),
            # ожидаемый результат по модели на момент выдачи
            "profit_expected": round(
                n_appr * ((1 - p[appr].mean()) * profit_paid - p[appr].mean() * loss_default), 0),
        })

    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)

    print("\nS1 (без переобучения), первые и последние 6 месяцев:")
    print(pd.concat([res.head(6), res.tail(6)]).to_string(index=False))

    print("\nИтог за тестовый период:")
    print(f"  AUC: {res['auc'].iloc[0]:.4f} -> {res['auc'].iloc[-1]:.4f}")
    print(f"  Предсказанная PD: {res['pd_pred'].iloc[0]:.4f} -> {res['pd_pred'].iloc[-1]:.4f}")
    print(f"  Фактическая PD:   {res['pd_actual'].iloc[0]:.4f} -> {res['pd_actual'].iloc[-1]:.4f}")
    print(f"  PSI: {res['psi'].iloc[0]:.4f} -> {res['psi'].iloc[-1]:.4f}, максимум {res['psi'].max():.4f}")
    print(f"  Прибыль фактическая: ${res['profit_actual'].sum():,.0f}")
    print(f"  Прибыль ожидаемая:   ${res['profit_expected'].sum():,.0f}")
    err = res["profit_expected"].sum() / res["profit_actual"].sum() - 1
    print(f"  Расхождение: {err:+.1%}")


if __name__ == "__main__":
    main()
