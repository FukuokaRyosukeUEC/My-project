"""ui — 最適化 UI(Streamlit)。

既存の `src/` `scripts/` には手を入れず、この配下だけで完結する追加レイヤ。
- run_store.py       : 実行ディレクトリ・設定 YAML 生成・スナップショット入出力
- optimize_worker.py : NSGA-II を回すサブプロセス(スナップショットを書き出す)
- render.py          : 既存 src/visualization.py を使った描画ラッパ
- app.py             : Streamlit 画面本体
"""
