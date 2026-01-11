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

ベイズ最適化ソースコード