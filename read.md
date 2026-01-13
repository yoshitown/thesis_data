thesis_data/
├── [README.md](http://readme.md/)                    # 全体構成・実験条件一覧
├── raw_data/                    # 生データ（読み取り専用）
│   ├── 2025-11-15_condition_A/
│   │   ├── run_001.csv
│   │   └── metadata.json
│   └── 2025-11-20_condition_B/
├── scripts/                     # 解析スクリプト
│   ├── 01_preprocess.py
│   ├── 02_calc_metrics.py       # Tanaka指標・分離度・理論段数
│   ├── 03_plot_chromatograms.py
│   └── requirements.txt
├── processed/                   # 中間データ・計算結果
│   ├── summary_table.csv
│   └── metrics_by_condition.csv
├── figures/                     # 図表（卒論用）
│   ├── candidates/              # 候補図（検討中）
│   │   ├── fig_chromatogram_v1.png
│   │   └── fig_chromatogram_v2.png
│   └── final/                   # 確定版
│       ├── fig1_chromatogram.png
│       ├── fig2_separation.png
│       └── table1_metrics.csv
└── archive/                     # 不採用データ・旧版

# 卒論実験データ管理

## 概要
第4章「結果および考察」用の実験データ・図表を管理。

## 実験条件一覧

### iSEC測定（4.1-4.3節）
- 環境: s = 0.60, 0.65, 0.69（良溶媒比）
- プローブ: PS標準 7種（MW 580-377,400）
- 測定項目: K_av, Δ²K_av, 差分比R
- 判定基準: CV < 10%（命題2.1検証）

### Tanaka指標測定（4.4-4.6節）
- Run数: 10条件（ベイズ最適化18イテレーション）
- 変数: s（溶媒比 0.60-0.73）、r（架橋剤比 0.19-0.42）
- 指標: N, Rs, H, S*, A, B, C
- 目標: N>5000, Rs>1.5, H>3.0, S*≈1.0, A≈2.5
- 損失関数: L(x) = ‖W·(y-y*)‖² + λ·R(x)

## ディレクトリ構成

- `raw_data/isec/`: iSEC測定CSV（環境別フォルダ）
- `raw_data/tanaka/`: Tanaka指標測定CSV + metadata.json
- `processed/`: 計算済み中間データ（K_av曲線、Table 4.1、最適化ログ）
- `figures/chapter4/`: 第4章用図表11点
- `scripts/`: 解析・図表生成スクリプト

## スクリプト実行順序

# 環境構築

pip install -r scripts/requirements.txt

# データ処理パイプライン

python scripts/01_[preprocess.py](http://preprocess.py)

python scripts/02_calc_isec_[metrics.py](http://metrics.py)

python scripts/03_calc_tanaka_[metrics.py](http://metrics.py)

python scripts/04_plot_[chapter4.py](http://chapter4.py)

## 図表対応

| Figure | ファイル名 | 節 | 内容 |
|--------|-----------|-----|------|
| 4.1 | fig4-1_isec_series_crossing.png | 4.1 | iSEC曲線の系列交差 |
| 4.2 | fig4-2_delta2_kav.png | 4.2 | Δ²K_avの環境依存性 |
| 4.3 | fig4-3_sr_landscape.png | 4.4 | s-r空間性能ランドスケープ |
| Table 4.1 | table4-1_tanaka_correlation.csv | 4.4 | 合成条件とTanaka指標 |
| 4.4 | fig4-4_chromatogram_comparison.png | 4.5 | 最適化前後クロマトグラム |
| 4.5 | fig4-5_convergence.png | 4.5 | 最適化収束性能比較 |
| 4.1B | fig4-1b_isec_anomaly.png | 4.3 | iSEC異常値証拠 |
| 4.2B | fig4-2b_ratio_test.png | 4.3 | 差分比不変性検定 |
| 4.4B | fig4-4b_radar_chart.png | 4.5 | Tanaka指標レーダーチャート |
| 4.5B | fig4-5b_bayesian_trajectory.png | 4.5 | ベイズ最適化探索軌跡 |
| 4.6A | fig4-6a_decision_flow.png | 4.6 | アプローチ決定フロー |

## 注意事項

- `raw_data/`は読み取り専用（再現性確保）
- 再解析時は`processed/`を削除してから実行
- Figure番号は卒論本文と対応