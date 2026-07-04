"""ML Monitoring Lite — Streamlit エントリポイント。

UI は薄く保ち、ドリフト計算は src/ に委譲する。
実行: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.brand import (
    apply_brand,
    footer_backlink,
    hero,
    section,
    sidebar_header,
    themed_altair,
    PALETTE,
    show_table,
)
from src.drift import (
    FEATURES,
    drift_report,
    generate_reference_and_current,
)

themed_altair(alt)

st.set_page_config(page_title="ML Monitoring Lite", page_icon="📡", layout="wide")
apply_brand(st)


@st.cache_data(show_spinner=False)
def _data(n: int, drift: float, seed: int):
    return generate_reference_and_current(n=n, drift=drift, seed=seed)


def main() -> None:
    sidebar_header(st, "ML Monitoring Lite")
    n = st.sidebar.slider("サンプル数 / データセット", 200, 5000, 2000, 100)
    drift = st.sidebar.slider("ドリフト量", 0.0, 2.0, 0.0, 0.1)
    seed = st.sidebar.number_input("シード", 0, 9999, 42, 1)
    psi_th = st.sidebar.slider("PSI アラート閾値", 0.1, 0.5, 0.25, 0.05)

    hero(
        st,
        "ML MONITORING",
        "ML Monitoring Lite",
        "学習時と推論時の分布差を PSI と KS 検定で検出し、データドリフト・品質監視・アラートの概念を示します。",
        chips=["Python", "SciPy", "Altair", "stlite"],
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
    section(st, "DRIFT METRICS", "ドリフト指標（特徴量別）")
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
    show_table(st, view)

    # --- PSI 棒グラフ（Altair）
    section(st, "PSI CHART", "PSI（特徴量別）")
    bar_df = report[["feature", "psi"]].copy()
    bar_df["alert"] = bar_df["psi"] >= float(psi_th)
    psi_chart = (
        alt.Chart(bar_df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("psi:Q", title="PSI", scale=alt.Scale(zero=True)),
            y=alt.Y("feature:N", title="特徴量", sort="-x"),
            color=alt.condition(
                alt.datum.psi >= float(psi_th),
                alt.value(PALETTE[2]),   # amber: アラート
                alt.value(PALETTE[0]),   # teal: 正常
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="特徴量"),
                alt.Tooltip("psi:Q", title="PSI", format=".3f"),
            ],
        )
        .properties(height=220)
    )
    threshold_rule = (
        alt.Chart(pd.DataFrame({"threshold": [float(psi_th)]}))
        .mark_rule(strokeDash=[4, 4])
        .encode(x=alt.X("threshold:Q"))
    )
    st.altair_chart(psi_chart + threshold_rule, use_container_width=True)

    # --- 分布比較（Altair 折れ線）
    section(st, "DISTRIBUTION", "分布比較")
    pick = st.selectbox("特徴量を選択", FEATURES)
    bins = np.linspace(
        min(ref[pick].min(), cur[pick].min()),
        max(ref[pick].max(), cur[pick].max()),
        30,
    )
    ref_h, _ = np.histogram(ref[pick].to_numpy(), bins=bins)
    cur_h, _ = np.histogram(cur[pick].to_numpy(), bins=bins)
    centers = (bins[:-1] + bins[1:]) / 2
    dist_df = (
        pd.DataFrame(
            {"値": np.round(centers, 2), "基準": ref_h, "現在": cur_h}
        )
        .melt(id_vars="値", var_name="データセット", value_name="件数")
    )
    dist_chart = (
        alt.Chart(dist_df)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("値:Q", title="値"),
            y=alt.Y("件数:Q", title="件数", scale=alt.Scale(zero=True)),
            color=alt.Color(
                "データセット:N",
                scale=alt.Scale(domain=["基準", "現在"], range=[PALETTE[0], PALETTE[1]]),
            ),
            tooltip=[
                "データセット:N",
                alt.Tooltip("値:Q", format=".2f"),
                "件数:Q",
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(dist_chart, use_container_width=True)

    st.divider()
    footer_backlink(st, repo="ml-monitoring-lite")


if __name__ == "__main__":
    main()
