"""
Этап 2: построение месячной панели из набора Lending Club.

Вход:  accepted_2007_to_2018Q4_csv.gz
Выход: panel.parquet      - кредиты окна наблюдения с признаками и исходом
       monthly_stats.csv  - помесячная сводка

Запуск: python build_panel.py
"""

import numpy as np
import pandas as pd

from paths import ACCEPTED, PANEL, RESULTS, require

SRC = require(ACCEPTED)
OUT_PANEL = PANEL
OUT_STATS = RESULTS / "monthly_stats.csv"

START, END = "2009-01", "2015-12"   # окно: достаточно дефолтов и исходы зрелые
TERM = " 36 months"

DEFAULT = ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"]
PAID = ["Fully Paid", "Does not meet the credit policy. Status:Fully Paid"]

# Признаки, известные на момент выдачи.
# Исключены grade/sub_grade/int_rate/installment - это оценка риска самого кредитора,
# и все поля, заполняемые после выдачи.
FEATURES_NUM = [
    "loan_amnt", "annual_inc", "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc", "collections_12_mths_ex_med",
    "mths_since_last_major_derog", "acc_now_delinq", "tot_coll_amt", "tot_cur_bal",
    "total_rev_hi_lim", "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy", "bc_util",
    "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc", "mths_since_recent_bc",
    "mths_since_recent_inq", "num_accts_ever_120_pd", "num_actv_bc_tl", "num_actv_rev_tl",
    "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl", "num_rev_accts",
    "num_rev_tl_bal_gt_0", "num_sats", "num_tl_90g_dpd_24m", "num_tl_op_past_12m",
    "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies", "tax_liens",
    "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit",
]
FEATURES_CAT = [
    "emp_length", "home_ownership", "verification_status", "purpose",
    "addr_state", "application_type",
]
# Экономика: нужна для расчёта прибыли и LGD, но НЕ для обучения модели.
ECON = [
    "funded_amnt", "int_rate", "total_rec_prncp", "total_rec_int",
    "total_pymnt", "recoveries", "collection_recovery_fee",
]
KEYS = ["id", "issue_d", "loan_status", "term", "earliest_cr_line", "last_pymnt_d"]

USECOLS = KEYS + FEATURES_NUM + FEATURES_CAT + ECON


def load() -> pd.DataFrame:
    """Читает файл частями и оставляет только кредиты окна наблюдения."""
    parts = []
    for chunk in pd.read_csv(SRC, usecols=USECOLS, chunksize=200_000, low_memory=False):
        chunk = chunk[chunk["term"] == TERM]
        chunk = chunk[chunk["loan_status"].isin(DEFAULT + PAID)]
        chunk = chunk[chunk["issue_d"].notna()]
        issue = pd.to_datetime(chunk["issue_d"], format="%b-%Y", errors="coerce")
        mask = (issue >= START) & (issue <= END + "-31")
        chunk = chunk[mask].copy()
        chunk["issue"] = issue[mask]
        if len(chunk):
            parts.append(chunk)
    return pd.concat(parts, ignore_index=True)


def add_target_and_economics(df: pd.DataFrame) -> pd.DataFrame:
    df["default"] = df["loan_status"].isin(DEFAULT).astype("int8")

    # 12-месячный дефолт (определение Базеля): платежи прекратились в первые 12 месяцев.
    last = pd.to_datetime(df["last_pymnt_d"], format="%b-%Y", errors="coerce")
    mob = ((last - df["issue"]).dt.days / 30.44).round()
    df["default_12m"] = ((df["default"] == 1) & (mob <= 12)).astype("int8")
    # Момент, когда исход становится наблюдаемым: списание признаётся примерно через 5 мес.
    df["observable_at"] = (df["issue"] + pd.DateOffset(months=18)).dt.to_period("M").astype(str)
    df["ym"] = df["issue"].dt.to_period("M").astype(str)

    # Длина кредитной истории на момент выдачи — признак, а не утечка.
    hist = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df["credit_hist_months"] = ((df["issue"] - hist).dt.days / 30.44).round()

    # EAD - непогашенный основной долг на момент списания; возвраты за вычетом комиссии.
    ead = (df["funded_amnt"] - df["total_rec_prncp"]).clip(lower=0)
    rec = (df["recoveries"] - df["collection_recovery_fee"]).clip(lower=0)
    df["ead"] = np.where(df["default"] == 1, ead, np.nan)
    df["lgd"] = np.where(df["default"] == 1, (1 - rec / ead.where(ead > 0)).clip(0, 1), np.nan)

    # Фактический денежный результат: всё полученное минус выданное.
    df["cash_result"] = df["total_pymnt"] + rec - df["funded_amnt"]
    return df


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("ym")
    out = pd.DataFrame({
        "n_loans": g.size(),
        "n_defaults": g["default"].sum(),
        "default_rate": g["default"].mean().round(4),
        "mean_funded": g["funded_amnt"].mean().round(0),
        "mean_rate": g["int_rate"].mean().round(2),
        "mean_fico": g["fico_range_low"].mean().round(1),
        "lgd_ead_weighted": g.apply(
            lambda x: 1 - (x.loc[x["default"] == 1, "recoveries"]
                           - x.loc[x["default"] == 1, "collection_recovery_fee"]).sum()
            / x.loc[x["default"] == 1, "ead"].sum()
        ).round(3),
        "cash_result_total": g["cash_result"].sum().round(0),
    })
    return out


def feature_report(df: pd.DataFrame) -> pd.DataFrame:
    """Доля пропусков по признакам — чтобы выбросить неинформативные."""
    feats = FEATURES_NUM + FEATURES_CAT + ["credit_hist_months"]
    na = df[feats].isna().mean().sort_values(ascending=False)
    return na[na > 0].round(3)


def main() -> None:
    df = load()
    df = add_target_and_economics(df)

    stats = monthly_summary(df)
    stats.to_csv(OUT_STATS)

    keep = KEYS + FEATURES_NUM + FEATURES_CAT + ECON + [
        "issue", "ym", "default", "default_12m", "observable_at",
        "credit_hist_months", "ead", "lgd", "cash_result",
    ]
    out = df[keep].copy()
    out["id"] = out["id"].astype(str)
    out.to_parquet(OUT_PANEL, index=False)

    print(f"Кредитов в панели: {len(df):,}")
    print(f"Месяцев: {df['ym'].nunique()}  ({stats.index[0]} … {stats.index[-1]})")
    print(f"Доля дефолтов (весь срок): {df['default'].mean():.4f}")
    print(f"Доля дефолтов (12 мес.):   {df['default_12m'].mean():.4f}")
    print(f"Дефолтов в месяц: {stats['n_defaults'].min()}–{stats['n_defaults'].max()}")
    print("\nПропуски в признаках (>0):")
    print(feature_report(df).to_string())
    print("\nПомесячная сводка (первые и последние 6 месяцев):")
    print(pd.concat([stats.head(6), stats.tail(6)]).to_string())


if __name__ == "__main__":
    main()
