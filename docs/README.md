# 文書の目次 / Documentation index

[日本語README](../README.md) · [English README](../README.en.md) · [検算の入口 / Quickstart](QUICKSTART.md)

初めて読む場合はREADMEからで十分です。ここは実験の比較対象・結果・失敗まで確認したい人向けの索引です。旧文書は当時の仮説を含むため、現在の解釈は [RESEARCH_STATUS](RESEARCH_STATUS.md) を優先してください。実験の完了と、原因の解明は区別しています。

## 現在地と手順

| 内容 | 文書 |
|---|---|
| 最新の支持・非再現・未確認 | [RESEARCH_STATUS](RESEARCH_STATUS.md) |
| 完了した試験と残る因果問題 | [RESEARCH_ROADMAP](RESEARCH_ROADMAP.md) |
| 自律的な継続の範囲と停止条件 | [AUTONOMOUS_RESEARCH_SCOPE](AUTONOMOUS_RESEARCH_SCOPE.md) |
| 作業仮説と測度の限界 | [THEORETICAL_FRAMEWORK](THEORETICAL_FRAMEWORK.md)（以前の整理。最新実験は研究状況を参照） |
| 検算・完全監査・再学習の違い | [QUICKSTART](QUICKSTART.md) |
| 結果CSVの索引 | [results](../results/README.md) |
| 過去の設計・統計ルール | [METHODS](METHODS.md) |
| X用告知の草案 | [SHARE_ON_X](SHARE_ON_X.md)（現在、告知は保留） |

## 2026年9月25日に確定した結果

| 試験 | 結果・論点 |
|---|---|
| [Issue121：オンラインのランダム・シャッフル対照](ONLINE_CONTROL_RESULT.md) | 60反復・3環境群。gradnov対loss-hard +2.672 pp、対勾配シャッフル +2.070 pp。3比較の片側Holm補正は合格。ただし対ランダムの差は小さく、両側95%区間はゼロを含む。 |
| [Issue115：平均と最低性能の新規60反復](RANK_TAIL_TRADEOFF_RESULT.md) | 平均改善と最低性能悪化のどちらも、事前登録の主判定を満たさず。clean・ばらつきの差は探索結果として分離。 |
| [Issue100：平均noveltyを調整した勾配の広がり](CENTERED_SPAN_CONFIRMATION_RESULT.md) | 操作・matchingは成立したが、性能改善の主仮説は支持されず。 |

Issue121の公開CSVだけの検算は [check_online_controls.py](../scripts/check_online_controls.py)。モデル・画像は不要です。Issue115の完全な成果物監査は [audit_rank_tail_evidence.py](../experiments/audit_rank_tail_evidence.py)。既存ZIPが必要で、再学習や生の画像推論は行いません。

## 以前の機構実験：高低の選択履歴を比較

ここでの高低差は、通常のオンラインgradnov対loss-hardの改善値とは別です。

| 試験 | 結果・論点 |
|---|---|
| [Issue91：比較可能な場面で平行移動も調整](OVERLAP_TRANSLATION_RESULT.md) | 調整後の高低差+2.041 pp。比較不能なステップでは同じ候補を採用。残る要因差あり。 |
| [Issue89：全ステップmatchingの校正](TRANSLATION_MATCH_CALIBRATION_RESULT.md) | FAIL。性能評価をせず停止した記録。 |
| [Issue86：平行移動だけで採点する対照](SPATIAL_ALLOCATION_RESULT.md) | 別の採点でも高低差。ただし勾配幾何も同時に変化。 |
| [Issue83：損失・総多様性を調整](PARAMETER_MATCHED_RESULT.md) | 正の高低差と、座標別の配分差の監査。 |
| [Issue80：損失層別の高低介入](LOSS_STRATIFIED_NONREDUNDANCY_RESULT.md) | 正の高低差。ただし総パラメータ多様性も変化。 |
| [平均noveltyの測度を点検](NOVELTY_COHERENCE_IDENTITY.md) | 方向の整合・相殺と、独立な方向数は同じではない。性能定理ではない。 |

## 通常の手法比較・採用環境数の実験

| 対象 | 文書 |
|---|---|
| CIFAR-10 / ResNet-20、40ペア | [CIFAR_RESNET_PRIMARY](CIFAR_RESNET_PRIMARY.md) |
| CIFAR-10 / ResNet-20の採用数比較 | [CIFAR_BUDGET_SCALING_RESULT](CIFAR_BUDGET_SCALING_RESULT.md) |
| FashionMNIST / Tiny Transformerの採用数比較 | [FASHION_BUDGET_SCALING_RESULT](FASHION_BUDGET_SCALING_RESULT.md) |
| Digits / SmallCNNの採用数比較 | [CNN_BUDGET_SCALING_RESULT](CNN_BUDGET_SCALING_RESULT.md) |
| 以前の結果のまとめ | [RESULTS](RESULTS.md)（歴史的なまとめ。最新結果は上記を参照） |

全候補を使うと同じ更新になることは実装上の対照です。それだけでは、少数採用で改善する原因の証明になりません。

## 失敗・再現性・過去の仮説も残す

[CPU再現性監査](CIFAR_CPU_REPRO_AUDIT.md)はcross-run driftを記録しています。[研究状況](RESEARCH_STATUS.md)には反証・校正失敗・非再現を含めています。[LIMITATIONS](LIMITATIONS.md)は以前の限界整理です。

[PROSPECTIVE_REPRESENTATION_RANK](PROSPECTIVE_REPRESENTATION_RANK.md)、[TRAJECTORY_MECHANISM](TRAJECTORY_MECHANISM.md)、[GRADIENT_DIRECTION_AUDIT](GRADIENT_DIRECTION_AUDIT.md)は過去の診断・仮説です。rankの予測記録を、原因の同定や信頼できる各run判定器として扱わないでください。

## そのほかの過去の記録

[Fashion初期rank試験](FASHION_TRANSFORMER_REP_RANK.md) · [Fashion独立extension](FASHION_TRANSFORMER_EXTENSION.md) · [CIFAR rank試験](CIFAR_RESNET_REP_RANK.md) · [適応的novelty係数](ADAPTIVE_BETA.md) · [相対的冗長性制御](RELATIVE_REDUNDANCY_CONTROL.md) · [RNG座標の転移](RNG_CROSS_GENERATOR.md)

各文書・実装・事前登録Issue・workflowを組にして確認してください。
