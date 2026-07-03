# ML Monitoring Lite

学習時（基準）と推論時（現在）の **分布差を PSI と KS 検定で検出** し、データドリフト・品質監視・アラートの概念を示す小型 Streamlit アプリです。

> **注意:** 本アプリはすべて **合成データ**（seed 固定で再現可能）であり、実運用の監視精度を保証するものではありません。

## 主要機能

- 基準データと現在データの **PSI（Population Stability Index）** 算出
- **KS 検定**（統計量・p 値）による分布差の検定
- 閾値を超えた特徴量の **アラート表示**
- 特徴量ごとの **分布比較** の可視化
- ドリフト量スライダーで分布のズレを操作

## 使用技術

- Python 3.11+ / Streamlit / NumPy / SciPy / pandas
- ドリフト計算は `src/drift.py`、UI は `app/streamlit_app.py`（薄く保つ）

## データの出所

外部データは使いません。基準・現在の2つの分布を seed 固定で合成します。

## ローカル実行手順

```bash
uv sync
uv run streamlit run app/streamlit_app.py
uv run pytest
```

`uv` を使わない場合は `pip install streamlit numpy scipy pandas pytest` でも動きます。
`app/streamlit_app.py` は `sys.path` を補正しているため、editable install なしでも起動します。

## ディレクトリ構成

```
ml-monitoring-lite/
├── app/streamlit_app.py   # Streamlit エントリ（UI）
├── src/drift.py           # PSI / KS 検定 / ドリフトレポート / 合成データ
├── src/brand.py           # 共通ブランドテーマ
├── tests/test_drift.py    # 同一分布≈0 / シフトで検出 / 境界のテスト
├── .streamlit/config.toml
└── pyproject.toml
```

## スクリーンショット

`assets/` に配置してください（現在は未配置）。

## 制限事項

- MVP であり、認証・DB・本番運用機能・課金は含みません
- PSI と KS 検定に基づく単純な監視デモで、実運用の監視基盤（時系列追跡・自動再学習・根本原因分析）の代替ではありません
- **合成データのデモであり、実在のモデル・データの品質を示すものではありません**

## セキュリティ

- API キー等の秘密情報は使用しません（`.env` 不要で動作）
