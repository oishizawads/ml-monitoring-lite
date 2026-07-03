"""データドリフト検出のロジック（PSI / KS 検定）。

UI から切り離した純粋関数のみ。すべて合成データ（seed 固定で再現可能）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

FEATURES = ["feature_a", "feature_b", "feature_c", "feature_d"]


def _clean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[~np.isnan(x)]


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index。基準 expected の分位でビン化して算出。

    同一分布なら 0 付近、ズレが大きいほど増加する。空・定数列でも落ちない（0 を返す）。
    """
    expected = _clean(expected)
    actual = _clean(actual)
    if expected.size == 0 or actual.size == 0:
        return 0.0
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    cuts = np.unique(np.quantile(expected, quantiles))
    if cuts.size < 2:  # 定数列
        return 0.0
    cuts = cuts.astype(float)
    cuts[0], cuts[-1] = -np.inf, np.inf
    e_counts = np.histogram(expected, bins=cuts)[0].astype(float)
    a_counts = np.histogram(actual, bins=cuts)[0].astype(float)
    e_pct = e_counts / e_counts.sum()
    a_pct = a_counts / a_counts.sum()
    eps = 1e-6
    e_pct = np.clip(e_pct, eps, None)
    a_pct = np.clip(a_pct, eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def ks_test(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """KS 統計量と p 値を返す。空・定数で計算不能なら (0.0, 1.0)。"""
    expected = _clean(expected)
    actual = _clean(actual)
    if expected.size == 0 or actual.size == 0:
        return (0.0, 1.0)
    try:
        res = ks_2samp(expected, actual)
        return (float(res.statistic), float(res.pvalue))
    except Exception:  # pragma: no cover - scipy 側の保険
        return (0.0, 1.0)


def psi_level(value: float) -> str:
    """PSI を業界慣行の3段階に分類する。"""
    if value < 0.1:
        return "stable"
    if value < 0.25:
        return "moderate"
    return "significant"


@dataclass(frozen=True)
class DriftRow:
    feature: str
    psi: float
    ks_stat: float
    ks_pvalue: float
    level: str
    alert: bool


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    psi_threshold: float = 0.25,
) -> pd.DataFrame:
    """特徴量ごとの PSI・KS・アラート判定をまとめて返す。"""
    features = [c for c in reference.columns if c in current.columns]
    rows: list[dict] = []
    for f in features:
        p = psi(reference[f].to_numpy(), current[f].to_numpy())
        ks_stat, ks_p = ks_test(reference[f].to_numpy(), current[f].to_numpy())
        # アラートは PSI（閾値スライダー連動）を主判定にする。
        # KS は大サンプルで偽陽性を出しやすいため参考情報に留め、
        # KS が極めて有意（p<0.001）な場合のみ補助的にアラートへ加える。
        alert = (p >= psi_threshold) or (ks_p < 0.001 and p >= psi_threshold * 0.5)
        rows.append(
            {
                "feature": f, "psi": p, "ks_stat": ks_stat,
                "ks_pvalue": ks_p, "level": psi_level(p), "alert": alert,
            }
        )
    return pd.DataFrame(rows, columns=["feature", "psi", "ks_stat", "ks_pvalue", "level", "alert"])


def generate_reference_and_current(
    n: int = 2000, drift: float = 0.0, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """基準データと現在データを合成する。drift（0〜）が大きいほど分布がずれる。"""
    rng = np.random.default_rng(seed)
    ref = pd.DataFrame(
        {
            "feature_a": rng.normal(0.0, 1.0, n),
            "feature_b": rng.gamma(2.0, 2.0, n),
            "feature_c": rng.normal(50.0, 10.0, n),
            "feature_d": rng.beta(2.0, 5.0, n),
        }
    )
    cur = pd.DataFrame(
        {
            # 平均シフト・分散拡大でドリフトを注入
            "feature_a": rng.normal(0.0 + drift * 1.0, 1.0 + drift * 0.3, n),
            "feature_b": rng.gamma(2.0 + drift * 1.5, 2.0, n),
            "feature_c": rng.normal(50.0 + drift * 8.0, 10.0, n),
            "feature_d": rng.beta(2.0, max(0.3, 5.0 - drift * 2.0), n),
        }
    )
    return ref, cur
