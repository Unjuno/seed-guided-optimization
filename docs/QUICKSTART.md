# まず読む・検算する・学習を再実行する

[日本語README](../README.md) · [English README](../README.en.md) · [実験目次](../experiments/README.md)

## 1. 読むだけならインストール不要

[研究状況](RESEARCH_STATUS.md)で主張の範囲を確認し、[結果の目次](../results/README.md)からCSVを開けます。最初から学習実験や大きな成果物の取得を行う必要はありません。

## 2. 公開CSVだけを検算する

### macOS / Linux

Python 3.12、Git、依存パッケージ取得時のネット接続を使います。

```bash
git clone https://github.com/Unjuno/seed-guided-optimization.git
cd seed-guided-optimization
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-check.txt
python scripts/check_public_results.py
```

### Windows PowerShell

```powershell
git clone https://github.com/Unjuno/seed-guided-optimization.git
cd seed-guided-optimization
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-check.txt
.venv\Scripts\python.exe scripts/check_public_results.py
```

Windowsでは仮想環境のPythonを直接呼ぶので、PowerShellの実行ポリシーを変更する手順は不要です。パッケージの導入後、検算スクリプトはネットにアクセスしません。必要なのはNumPy・pandas・SciPyだけで、PyTorch、GPU、画像、checkpointは使いません。

### 何を確認するのか

[30ペアの公開値](../results/translation_matched_primary30.csv)から、二つの高novelty対低novelty比較について、平均、標準誤差、95%両側t区間、事前指定の片側p値、正の差の件数を再計算します。[保存済み集計](../results/translation_matched_summary30.csv)との一致も検査します。

出力例（丸めた表示）:

```text
Reference loss + total diversity: +3.070818 pp; 95% CI [+2.040397, +4.101239]; one-sided p=6.13693e-07; positive=26/30
Also translation-balanced on supported steps: +2.040585 pp; 95% CI [+0.861808, +3.219362]; one-sided p=0.000685123; positive=22/30
PUBLIC STATISTICS VERIFIED (not a full experiment audit)
```

これは**新しい学習実験ではありません**。matchingの全ステップ検査やcheckpointの真正性確認も行いません。元の凍結判定を差し替えず、公開された二つの主要性能集計を確認する入口です。[比較対象と限界](OVERLAP_TRANSLATION_RESULT.md)も読んでください。

CSVの正答率差は割合で保存されています。`0.01`が1パーセントポイントです。標準誤差・信頼区間は固定画像・環境群に条件づけた反復間変動で、全データセットや全backendの不確かさを表しません。

### 他の軽量チェック

同じ依存環境で、既存の検算コードも動かせます。前二つは過去の公開データの確認、最後は代数の合成テストです。

```bash
python experiments/check_parameter_matched_evidence.py
python experiments/check_spatial_allocation_evidence.py
python experiments/check_novelty_coherence_identity.py
```

単体テストと入口文書の相対リンク検査:

```bash
python -m unittest discover -s scripts -p 'test_public_results.py'
python scripts/check_public_docs.py
```

## 3. 元の成果物を監査する場合

こちらは別の作業です。GitHub Actionsの対象runからZIPを取得して展開します。取得時にGitHubへのログインが必要になる場合があります。成果物には保存期限があるため、**CSVの閲覧・上の検算はActions成果物に依存させていません**。

| 対象 | 実行記録 | 成果物名 |
|---|---|---|
| 全ステップmatchingに失敗した校正 | [run 34250390284](https://github.com/Unjuno/seed-guided-optimization/actions/runs/34250390284) | `translation-calibration-evidence` |
| 比較可能なステップに限定した確認試験 | [run 34251284833](https://github.com/Unjuno/seed-guided-optimization/actions/runs/34251284833) | `overlap-complete-evidence` |

ZIPをそれぞれ `calibration_extracted/` と `confirmation_extracted/` に展開した場合のコマンドです。実行前に、必要なPyTorch等を含む対象workflowの依存環境を確認してください。

```bash
python experiments/audit_translation_support.py --evidence-dir calibration_extracted --output-dir calibration_audit
python experiments/check_overlap_translation_evidence.py --evidence-dir confirmation_extracted --output-dir confirmation_audit
```

必要な成果物が取得できない場合、完全監査ができたとは扱わないでください。公開CSVの統計検算と、モデル・選択履歴を含む完全監査は異なります。

## 4. 学習そのものを再実行する場合

[experiments/README.md](../experiments/README.md)から対象実験のworkflowを確認してください。モデルや学習時間は実験によって異なり、単一の万能コマンドはありません。

元の学習再現では、Python、CPU版PyTorch、データ分割、seed、thread数、optimizer、画像変換、判定条件を、その実験の定義に合わせてください。CIFARとDigitsではthread設定も異なります。`requirements-check.txt`は学習用ではなく、`requirements.txt`だけではCIFAR用のtorchvision等が揃わない場合があります。**対象workflowを実行環境の定義として使います。**

既存の事前登録実験を変更する場合は、新しい試験として区別してください。元の公開CSVを新しい結果で上書きする必要はありません。依存バージョンを変えた実行は、その変更も報告してください。

## 困ったとき

`ModuleNotFoundError`は、上で作った仮想環境のPythonで`requirements-check.txt`を導入したか確認してください。公開検算は不足列・反復の欠落や重複・非有限値・集計不一致を検出すると、エラーメッセージと非ゼロの終了コードを返します。

[Issue](https://github.com/Unjuno/seed-guided-optimization/issues)には、実行したcommit、コマンド、Pythonと依存パッケージのバージョン、エラーを添えてください。OSごとの手順の記載は、全OSでの動作検証を意味しません。
