# イベント会場ブース配置最適化シミュレーション
67
仕様書「イベント会場ブース配置最適化シミュレーション 詳細設計仕様書(改訂版 v2.0)」に基づく Python 実装。
2 次元グリッド上に 6 ブースを配置し、来場者エージェントをシミュレーションして、
NSGA-II により「移動負担 f_move」と「混雑度 f_congestion」の 2 目的を同時最適化する。

既定の空間・ブース条件は `20260629/` の独立版を移植している。会場は 20×16、
入口は下辺中央 2 セル、出口は上辺中央 2 セル。ブースは 2×2 が 3 基、
3×2 が 3 基で、受付は左下、体験後退出は左上外側、待機列は下辺から右辺へ
回り込む L 字形（受付とは分離、1 セル 5 人）である。来場者生成・経路探索・目的関数は
`src/` の現行実装を使用する。

- リポジトリ: https://github.com/catdogpenguin/U_PoC2026_YMFS
- 想定 Python: 3.10 以上(開発・検証は 3.12)

---

## 目次

1. [ディレクトリ構成](#ディレクトリ構成)
2. [セットアップ(git clone と仮想環境)](#セットアップgit-clone-と仮想環境)
3. [使い方](#使い方)
4. [GitHub での開発フロー](#github-での開発フロー)
5. [検証結果](#検証結果この実装での実測)
6. [仕様書からの主要な実装対応](#仕様書からの主要な実装対応)
7. [実装上の設計判断](#実装上の設計判断仕様書が明示しない部分)

---

## ディレクトリ構成

```text
U_PoC2026_YMFS/                     # リポジトリルート
├── README.md                       # 本ファイル
├── requirements.txt                # 依存ライブラリ
├── .gitignore                      # 生成物・仮想環境・キャッシュの除外設定
├── .github/
│   └── PULL_REQUEST_TEMPLATE.md    # PR 作成時に自動で挿入される雛形
│
├── config/                         # 設定の一元管理(YAML。§26)
│   ├── default_config.yaml         #   会場・入退場辺・待機列形状・時間・容量・効用重み等
│   ├── booths.yaml                 #   6 ブースの仕様(サイズ・カテゴリ・魅力度・収容・体験時間・行列長)
│   └── visitor_profiles.yaml       #   来場者プロファイル(速度・混雑回避・待ち耐性・興味傾向)
│
├── src/                            # ライブラリ本体
│   ├── __init__.py
│   ├── models.py                   #   セル種別/状態などの定数、dataclass 群、YAML ローダ(§2,3,5,6)
│   ├── venue.py                    #   会場グリッド構築・入口出口・配置適用(§4,5)
│   ├── rng.py                      #   乱数 3 系統分離ヘルパと softmax 抽選(§2.3.2)
│   ├── routing.py                  #   BFS 静的距離場・混雑考慮 A*・勾配降下(§9)
│   ├── constraints.py              #   配置制約チェックと違反量 g の算出(§14)
│   ├── visitors.py                 #   来場者テーブルの決定的生成(配置非依存。§7)
│   ├── simulator.py                #   メインループ・サブステップ移動・一括競合解決・ブース体験(§3,8,10,11)
│   ├── metrics.py                  #   目的関数 f_move / f_congestion と補助 KPI・幸福度(§12,13)
│   ├── parallel.py                 #   1 配置評価の入口・プロセス並列コンテキスト(§16)
│   ├── optimization.py             #   pymoo Problem・NSGA-II・有効初期集団・代表解抽出・結果保存(§15,19)
│   └── visualization.py            #   パレート図・ヒートマップ・比較表・アニメーション(§20)
│
├── scripts/                        # 実行エントリポイント(CLI)
│   ├── benchmark.py                #   1 秒評価目標のベンチマーク(§18)
│   ├── run_optimization.py         #   NSGA-II 最適化の実行と成果物出力
│   └── visualize_solution.py       #   配置 JSON の可視化(ヒートマップ・GIF)
│
├── tests/                          # テスト(§23)
│   ├── __init__.py
│   ├── conftest.py                 #   20×16 会場で有効な手作り配置
│   ├── test_constraints.py         #   §23.1 配置制約
│   ├── test_movement.py            #   §23.2 セル容量・競合解決 / §23.5 目的関数の直交性
│   ├── test_queue.py               #   §23.3 ブース体験・待ち行列・諦め
│   └── test_reproducibility.py     #   再現性 / 配置ごとの目的値識別
│
└── results/                        # 最適化の出力先(生成物は .gitignore 対象)
    └── .gitkeep                    #   空ディレクトリを Git 管理下に残すための placeholder
```

> 仕様書 §21 の推奨構成に対し、来場者テーブル生成を `src/visitors.py` として分離した。
> 仕様書が挙げる `test_objectives.py` / `test_metrics.py` の内容は、
> `test_movement.py`(目的関数の直交性)と `test_reproducibility.py`(目的関数一致・トレードオフ)に統合している。

---

## セットアップ(git clone と仮想環境)

### 1. クローン

```bash
git clone https://github.com/catdogpenguin/U_PoC2026_YMFS.git
cd U_PoC2026_YMFS
```

### 2. 仮想環境(`.venv`)の作成と有効化

プロジェクト直下に `.venv` を作る。`.gitignore` で除外済みなのでコミットされない。

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows(PowerShell)**
```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
```

有効化されるとプロンプト先頭に `(.venv)` が表示される。抜けるときは `deactivate`。

### 3. 依存ライブラリのインストール

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. 動作確認

```bash
python -m pytest tests/ -q       # 全テストが通れば環境 OK
python scripts/benchmark.py      # 1 評価あたりの計算時間を測定(§18)
```

---

## 使い方

```bash
# 1 秒評価目標のベンチマーク(§18)
python scripts/benchmark.py

# 最適化(フル設定: pop=100, gen=100)
python scripts/run_optimization.py

# 縮小設定での動作確認(個体数・世代数・並列数を指定)
python scripts/run_optimization.py --pop 16 --gen 8 --workers 4

# 配置 JSON の可視化(ヒートマップ・アニメーション GIF)
python scripts/visualize_solution.py results/knee_layout.json --animate

# テスト(§23)
python -m pytest tests/ -q
```

成果物は `results/` に出力される(§19.1):
`pareto_solutions.csv`, `pareto_front.png`, `best_movement_layout.json`,
`best_congestion_layout.json`, `knee_layout.json`, `optimization_log.csv`,
代表 3 案のヒートマップ、比較表 CSV。
これらは `.gitignore` 対象なので、共有したい場合は Issue や PR に個別添付するか、`git add -f` で明示的に追加する。

パラメータ(会場サイズ・来場者数・ブース仕様・最適化設定など)はすべて `config/` の YAML で変更する。
コードへ直書きしないこと(§26)。

---

## GitHub での開発フロー

標準的な **GitHub Flow**(`main` を常にリリース可能に保ち、作業ごとにブランチを切って PR でマージ)を採用する。

### 0. 初回のみ: 認証と署名設定

HTTPS + [Personal Access Token](https://github.com/settings/tokens) または SSH 鍵を設定しておく。
コミットに使う名前とメールが未設定なら:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

### 1. main を最新化して作業ブランチを切る

ブランチ名は `種別/内容` の形式を推奨(例: `feature/queue-refactor`, `fix/astar-deadlock`, `docs/readme`)。

```bash
git switch main
git pull origin main
git switch -c feature/your-topic
```

### 2. 変更してコミット

```bash
git add <変更したファイル>          # 例: git add src/simulator.py tests/test_queue.py
git status                          # 生成物(results/ など)が混ざっていないか確認
git commit -m "待ち行列の前詰め処理を修正"
```

コミットメッセージは「何をしたか」が一目で分かる命令形で簡潔に。1 コミット 1 論点を心がける。

### 3. リモートへ push

初回はブランチの upstream を設定する(`-u`):

```bash
git push -u origin feature/your-topic
```

2 回目以降は `git push` だけでよい。

### 4. プルリクエスト(PR)を Web の GitHub 上で作成

1. ブラウザで https://github.com/catdogpenguin/U_PoC2026_YMFS を開く。
2. push 直後なら上部に表示される **「Compare & pull request」** をクリック。
   表示されない場合は **「Pull requests」タブ → 「New pull request」** を押し、
   base = `main`、compare = 自分のブランチ を選択する。
3. `.github/PULL_REQUEST_TEMPLATE.md` の内容が本文に自動で入るので、
   概要・変更内容・仕様書との対応・テスト結果を記入し、チェックリストを埋める。
4. 必要ならレビュアーやラベルを指定して **「Create pull request」** をクリック。
5. レビュー指摘があれば同じブランチに追加コミットして `git push`。PR は自動更新される。
6. 承認後、Web 上で **「Merge pull request」** を実行(`main` はスカッシュマージ推奨)。

### 5. マージ後の後片付け

```bash
git switch main
git pull origin main               # マージ結果を取り込む
git branch -d feature/your-topic   # ローカルの作業ブランチを削除
```

リモート側のブランチは、PR マージ時に GitHub の「Delete branch」ボタンで削除できる。

### コンフリクトが起きたら

作業中に `main` が進んで衝突した場合:

```bash
git switch feature/your-topic
git fetch origin
git merge origin/main              # 衝突ファイルを編集して解決
git add <解決したファイル>
git commit                         # マージコミットを作成
git push
```

---

## 検証

`python -m pytest tests/ -q` で、20260629 の空間形状、配置制約、L字待機列の
セル容量と前詰め、体験後の専用退出セル、移動競合、再現性を検査する。

---

## 仕様書からの主要な実装対応

- **乱数 3 系統分離(§2.3.2)**: `src/rng.py`。系統1=来場者生成 `[seed,0]`、
  系統2=来場者別意思決定 `[seed,1,vid]`、系統3=競合タイブレーク `[seed,2]`、
  初期集団サンプリング `[seed,3]`。softmax 抽選は累積和方式で 1 選択 =
  乱数 1 個消費に固定(`Generator.choice` の内部消費数に依存しない)。
- **f_move の直交性(§12.1)**: `moved_cells`(競合解決で実際に移動したセル数)
  のみから自由歩行換算時間を算出。ブロック待機・行列待ち・入場遅延は含まない
  (`tests/test_movement.py::test_f_move_not_polluted_by_blocking` で検証)。
- **一括競合解決(§10)**: サブステップ開始時点の占有スナップショットで空きを
  判定するため、同一サブステップ内の退出は次サブステップから有効。優先順位は
  ブロック回数 → 残り滞在時間 → 系統3乱数。
- **待ち行列の実セル占有(§5.2)**: フェーズ3 要件どおり実装。待機者は先頭詰めで
  行列セルを占有し、先頭入場・離脱時に前詰めされる。

---

## 実装上の設計判断(仕様書が明示しない部分)

1. **行列参加のテレポート簡略化**: 受付セルに到達した来場者が行列に入るとき、
   受付セルから該当スロットセルへ直接移動する(この移動は `moved_cells` に
   数えない)。行列末尾まで歩く挙動は再現しない。
2. **L字待機列**: 待機列セルは通路として使わず、各セルに先頭から最大5人を
   割り当てる。受付セルは到着・入場処理専用で、待機者は配置しない。
3. **通路幅チェック(§14-8)**: 「各通行可能セルが少なくとも 1 つの
   w×w 全通行可能ブロックに属する」ことで判定し、違反セル数を幾何量
   (重み 1)として g に加算する(到達性の ×100 とは区別)。
4. **move_credit の累積上限**: ブロック中の残高蓄積による後追い高速移動を
   防ぐため、上限を 1 ステップの最大移動量とした。
5. **行列参加時の諦め判定(§11.3)**: 満杯、または残り時間 <(推定待ち時間+
   平均体験時間)のときに諦める。待機中は許容時間の厳密超過(>)で諦める。
6. **タイムアウト・無効配置のペナルティ**: 目的関数値 1.0e6 の定数
   (§17.6 の「観測最大値の 10 倍」の代替。制約付き NSGA-II では違反量 g で
   劣後させるため実質影響なし)。
7. **体験成功率**: 総体験回数 / (総体験回数 + 総諦め回数) と定義。
8. **入場スポーン**: 下辺中央の入口 2 セルを x 昇順に探索し空きセルへ出現。全セル満杯なら
   FIFO で次ステップへ繰り延べ、繰り延べ時間を入場遅延 KPI に計上(§4.5)。

---

## 縮小設定での実行例

まず `python scripts/run_optimization.py --pop 16 --gen 8 --workers 4` で
設定と実行環境を確認し、必要に応じて `config/default_config.yaml` の
個体数・世代数を調整する。
