# 実験コード / Experiments

[トップへ](../README.md) · [公開CSVの検算](../docs/QUICKSTART.md) · [結果CSV](../results/README.md)

実験ごとの再現スクリプトです。単一の製品用optimizerパッケージではありません。まず結果を見るだけなら、学習や画像データの取得は不要です。

## 再学習しない入口

リポジトリのrootで実行します。仮想環境の作り方は [QUICKSTART](../docs/QUICKSTART.md) を参照してください。

```bash
python -m pip install -r requirements-check.txt
python scripts/check_public_results.py
python experiments/check_parameter_matched_evidence.py
python experiments/check_spatial_allocation_evidence.py
python experiments/check_novelty_coherence_identity.py
```

最初の検算はIssue #91の二つの主要性能集計、次の二つは過去の公開ペア統計、最後は測度に関する合成テストです。新しい反復実験としては数えません。完全な成果物監査は [別手順](../docs/QUICKSTART.md) です。

## 最近の介入実験

| 内容 | 実装 | 実行条件を定義するworkflow |
|---|---|---|
| 平行移動調整・比較可能な場面のみ介入 | [overlap_translation_novelty.py](overlap_translation_novelty.py) | [workflow](../.github/workflows/overlap_translation_novelty.yml) |
| 全ステップ平行移動調整の校正 | [translation_matched_calibration.py](translation_matched_calibration.py) | [workflow](../.github/workflows/translation_matched_calibration.yml) |
| 平行移動スコアによる代替採点 | [spatial_allocation_control.py](spatial_allocation_control.py) | [workflow](../.github/workflows/spatial_allocation_control.yml) |
| 総パラメータ多様性を揃える校正 | [parameter_matched_novelty_calibration.py](parameter_matched_novelty_calibration.py) | [workflow](../.github/workflows/parameter_matched_novelty_calibration.yml) |
| 総パラメータ多様性を揃えた確認試験 | [parameter_matched_novelty_confirmatory.py](parameter_matched_novelty_confirmatory.py) | [workflow](../.github/workflows/parameter_matched_novelty_confirmatory.yml) |
| 損失層別の高低介入 | [loss_stratified_nonredundancy.py](loss_stratified_nonredundancy.py) | [workflow](../.github/workflows/loss_stratified_nonredundancy.yml) |

これらは共通の参照モデルで選択履歴を作る機構実験です。オンラインgradnov対loss-hardの効果量と混ぜないでください。

## 採用する環境数の比較

[CNN](cnn_budget_scaling.py) / [workflow](../.github/workflows/cnn_budget_scaling.yml) · [Fashion Transformer](fashion_transformer_budget_scaling.py) / [workflow](../.github/workflows/fashion_transformer_budget_scaling.yml) · [CIFAR ResNet](cifar_resnet_budget_scaling.py) / [workflow](../.github/workflows/cifar_resnet_budget_scaling.yml)

## 初期の手法・optimizer比較

[mlp_geometric.py](mlp_geometric.py) · [gradient_vs_parameter_novelty.py](gradient_vs_parameter_novelty.py) · [cnn_replication.py](cnn_replication.py) · [optimizer_ablation.py](optimizer_ablation.py) · [sgd_lr_sweep.py](sgd_lr_sweep.py)

共通モデル・変換・署名の定義は [common.py](common.py)。CIFARの主要手法比較は [cifar_resnet_primary.py](cifar_resnet_primary.py) と [workflow](../.github/workflows/cifar_resnet_primary.yml) を参照してください。

## その他の過去の実験

| 系列 | 主な実装 |
|---|---|
| RNG候補圧縮・関連座標 | [rng_compression_sweep.py](rng_compression_sweep.py)、[learned_rng_fingerprint.py](learned_rng_fingerprint.py)、[learned_rng_cross_generator.py](learned_rng_cross_generator.py) |
| novelty係数の制御 | [gradient_novelty_beta_adaptive.py](gradient_novelty_beta_adaptive.py)、[gradient_novelty_relative_control.py](gradient_novelty_relative_control.py)、[relative_control_breast_cancer.py](relative_control_breast_cancer.py) |
| trajectory / rank診断 | [trajectory_mechanism_pilot.py](trajectory_mechanism_pilot.py)、[prospective_rep_rank_validation.py](prospective_rep_rank_validation.py)；[解釈の更新](../docs/RESEARCH_STATUS.md)も参照 |
| CPU時間 | [wallclock_seed_count.py](wallclock_seed_count.py)；GPU効率を示すものではない |

## 再学習するときの注意

対象workflowのPython・依存版・thread数・データ分割・seed・optimizer条件を使ってください。[requirements.txt](../requirements.txt)は基本依存であり、実験固有の追加依存やCPU wheelの指定を置き換えるものではありません。

参照履歴の作成、全履歴の封印、介入学習、全stateの封印、held-out評価を分ける試験があります。順序を変えたり、一部の結果だけを見て設定・反復数を調整したりしないでください。既存workflowを再実行する行為は、計算資源の使用を伴います。公開結果の閲覧や検算のためには不要です。

成果物や元CSVを上書きせず、変更した試験は別の出力先と実験名にします。反復の単位・検定族・補正は [METHODS](../docs/METHODS.md) と各事前登録を確認し、現在の主張は [RESEARCH_STATUS](../docs/RESEARCH_STATUS.md) を参照してください。
