"""ML Monitoring Lite — Streamlit エントリポイント。

UI は薄く保ち、ドリフト計算は src/ に委譲する。
実行: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.brand import apply_brand, hero
from src.drift import (
    FEATURES,
    drift_report,
    generate_reference_and_current,
)

st.set_page_config(page_title="ML Monitoring Lite", page_icon="📡", layout="wide")
apply_brand(st)


@st.cache_data(show_spinner=False)
def _data(n: int, drift: float, seed: int):
    return generate_reference_and_current(n=n, drift=drift, seed=seed)


def main() -> None:
    st.sidebar.title("ML Monitoring Lite")
    st.sidebar.caption("合成データのドリフト監視デモ")
    n = st.sidebar.slider("サンプル数 / データセット", 200, 5000, 2000, 100)
    drift = st.sidebar.slider("ドリフト量", 0.0, 2.0, 0.0, 0.1)
    seed = st.sidebar.number_input("シード", 0, 9999, 42, 1)
    psi_th = st.sidebar.slider("PSI アラート閾値", 0.1, 0.5, 0.25, 0.05)

    hero(
        st,
        "Model Monitoring",
        "ML Monitoring Lite",
        "学習時と推論時の分布差を PSI と KS 検定で検出し、データドリフト・品質監視・アラートの概念を示します。",
    )

    ref, cur = _data(int(n), float(drift), int(seed))
    report = drift_report(ref, cur, psi_threshold=float(psi_th))

    n_alert = int(report["alert"].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("監視特徴量数", len(report))
    c2.metric("アラート数", n_alert)
    c3.metric("最大 PSI", f"{report['psi'].max():.3f}")
    c4.metric("ドリフト量", f"{drift:.1f}")

    if n_alert > 0:
        st.warning(
            f"⚠️ {n_alert} 件の特徴量でドリフトを検出しました（PSI≥{psi_th}。KS 検定は参考指標）。"
        )
    else:
        st.success("現在ドリフトは検出されていません。サイドバーの『ドリフト量』を上げると変化します。")

    st.divider()
    st.subheader("ドリフト指標（特徴量別）")
    view = report.copy()
    view["PSI"] = view["psi"].map(lambda x: f"{x:.3f}")
    view["KS統計量"] = view["ks_stat"].map(lambda x: f"{x:.3f}")
    view["KS p値"] = view["ks_pvalue"].map(lambda x: f"{x:.4f}")
    view["判定"] = view["level"].map(
        {"stable": "🟢 stable", "moderate": "🟡 moderate", "significant": "🔴 significant"}
    )
    view["アラート"] = view["alert"].map(lambda b: "🚨" if b else "")
    view = view.rename(columns={"feature": "特徴量"})[
        ["特徴量", "PSI", "KS統計量", "KS p値", "判定", "アラート"]
    ]
    st.dataframe(view, width="stretch", hide_index=True)

    st.subheader("PSI（特徴量別）")
    st.bar_chart(report[["feature", "psi"]].set_index("feature"), height=260)

    st.subheader("分布比較")
    pick = st.selectbox("特徴量を選択", FEATURES)
    hist_df = pd.DataFrame(
        {
            "value": np.concatenate([ref[pick].to_numpy(), cur[pick].to_numpy()]),
            "dataset": (["基準"] * len(ref)) + (["現在"] * len(cur)),
        }
    )
    # 簡易ヒストグラム（ビンごとの件数を基準/現在で比較）
    bins = np.linspace(hist_df["value"].min(), hist_df["value"].max(), 30)
    ref_h, _ = np.histogram(ref[pick].to_numpy(), bins=bins)
    cur_h, _ = np.histogram(cur[pick].to_numpy(), bins=bins)
    centers = (bins[:-1] + bins[1:]) / 2
    chart_df = pd.DataFrame({"基準": ref_h, "現在": cur_h}, index=np.round(centers, 2))
    st.line_chart(chart_df, height=280)

    st.divider()
    st.caption(
        "本アプリは合成データのデモであり、実運用の監視精度を保証するものではありません。"
    )


if __name__ == "__main__":
    main()
