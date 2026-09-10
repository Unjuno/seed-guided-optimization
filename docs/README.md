# 文書の目次 / Documentation index

[日本語README](../README.md) · [English README](../README.en.md) · [検算の入口 / Quickstart](QUICKSTART.md)

初めて読む場合はREADMEからで十分です。ここは実験の比較対象・結果・失敗まで確認したい人向けの索引です。旧文書は当時の仮説を含むため、現在の解釈は [RESEARCH_STATUS](RESEARCH_STATUS.md) と [THEORETICAL_FRAMEWORK](THEORETICAL_FRAMEWORK.md) を優先してください。

## 現在地と手順

| 内容 | 文書 |
|---|---|
| 何が支持され、何が未確認か | [RESEARCH_STATUS](RESEARCH_STATUS.md) |
| 作業仮説と、測度・因果解釈の限界 | [THEORETICAL_FRAMEWORK](THEORETICAL_FRAMEWORK.md) |
| 読むだけ / 公開CSVの検算 / 完全監査 / 再学習 | [QUICKSTART](QUICKSTART.md) |
| 結果CSVの索引 | [results](../results/README.md) |
| 過去の設計・比較・統計ルール | [METHODS](METHODS.md) |
| X用の告知文 | [SHARE_ON_X](SHARE_ON_X.md) |

## 最近の機構実験：高低の選択履歴を比較

ここでの大きな高低差は、通常のオンラインgradnov対loss-hardの改善値とは別です。

| 試験 | 結果・論点 |
|---|---|
| [Issue #91：比較可能な場面で平行移動も調整](OVERLAP_TRANSLATION_RESULT.md) | 調整後の高低差+2.041 pp。比較不能なステップでは両群に同じ候補を採用。残る要因差あり。 |
| [Issue #89：全ステップmatchingの校正](TRANSLATION_MATCH_CALIBRATION_RESULT.md) | FAIL。性能評価をせずに停止した記録。 |
| [Issue #86：平行移動だけで採点する対照](SPATIAL_ALLOCATION_RESULT.md) | 別の採点でも高低差。ただし勾配幾何も同時に変わる。 |
| [Issue #83：損失・総多様性を調整](PARAMETER_MATCHED_RESULT.md) | 正の高低差と、座標別の配分差の監査。 |
| [Issue #80：損失層別の高低介入](LOSS_STRATIFIED_NONREDUNDANCY_RESULT.md) | 正の高低差。ただし総パラメータ多様性も動く。 |
| [平均noveltyの測度を点検](NOVELTY_COHERENCE_IDENTITY.md) | 方向の整合・相殺と、独立な方向数は同じではない。性能定理ではない。 |

## 通常の手法比較・採用環境数の実験

| 対象 | 文書 |
|---|---|
| CIFAR-10 / ResNet-20、40ペアのオンライン手法比較 | [CIFAR_RESNET_PRIMARY](CIFAR_RESNET_PRIMARY.md) |
| CIFAR-10 / ResNet-20の採用数比較 | [CIFAR_BUDGET_SCALING_RESULT](CIFAR_BUDGET_SCALING_RESULT.md) |
| FashionMNIST / Tiny Transformerの採用数比較 | [FASHION_BUDGET_SCALING_RESULT](FASHION_BUDGET_SCALING_RESULT.md) |
| Digits / SmallCNNの採用数比較 | [CNN_BUDGET_SCALING_RESULT](CNN_BUDGET_SCALING_RESULT.md) |
| 以前の結果のまとめ | [RESULTS](RESULTS.md)（歴史的なまとめ。最新結果は上記を参照） |

全候補を使うと手法が同じ更新になることは実装上の対照です。それだけで、小さい採用数で改善する原因が証明されるわけではありません。

## 失敗・再現性・過去の仮説も残す

[CPU再現性監査](CIFAR_CPU_REPRO_AUDIT.md)はcross-run driftを記録しています。[研究状況](RESEARCH_STATUS.md)にはrank介入、standardized-rank、structured/nuisance校正等の否定的結果を含めています。[LIMITATIONS](LIMITATIONS.md)は以前の限界整理です。

[PROSPECTIVE_REPRESENTATION_RANK](PROSPECTIVE_REPRESENTATION_RANK.md)、[TRAJECTORY_MECHANISM](TRAJECTORY_MECHANISM.md)、[GRADIENT_DIRECTION_AUDIT](GRADIENT_DIRECTION_AUDIT.md)は過去の診断・仮説の記録です。rankの予測記録を、rankが原因である証拠や各runの信頼できる判定器として扱わないでください。

## そのほかの過去の記録

[Fashion初期rank試験](FASHION_TRANSFORMER_REP_RANK.md) · [Fashion独立extension](FASHION_TRANSFORMER_EXTENSION.md) · [CIFAR rank試験](CIFAR_RESNET_REP_RANK.md) · [適応的novelty係数](ADAPTIVE_BETA.md) · [相対的な冗長性制御](RELATIVE_REDUNDANCY_CONTROL.md) · [RNG座標の転移](RNG_CROSS_GENERATOR.md)

詳細な実行条件は、それぞれの文書、対応する実装、事前登録Issue、workflowを組にして確認してください。
