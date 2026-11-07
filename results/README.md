# 結果CSV / Evidence index

[トップへ](../README.md) · [文書目次](../docs/README.md) · [検算手順](../docs/QUICKSTART.md)

これは実験の証拠を保管する場所です。探索的な集計、副次評価、事前登録した主要評価を区別してください。**同じ「改善値」でも比較対象が違う表を、手法のランキングとして並べないでください。**

## 2026年11月8日：早期切替Issue133の公開集計

[登録3検定](switch_early_20261108_primary.csv)、[12環境群×3差](switch_early_20261108_pools.csv)、[clean非劣性の12差](switch_early_20261108_clean_guardrail_pools.csv)、[4方式の平均指標](switch_early_20261108_policy_means.csv)。60切替の40/80間のピークは未支持、clean非劣性も未支持。\`python scripts/check_switch_early_20261108.py\` はCSVの統計だけ再計算する。全状態・raw画像を検査したとの主張ではない。[結果の全文](../docs/SWITCH_TIME_EARLY_RESULT_20261108.md)。

## 2026年11月7日：切替時刻の登録比較（Issue131）

[12プール・3主要検定](switch_time_20261107_primary.csv)、[12プール×3主要差](switch_time_20261107_pool_differences.csv)、[4方策のclean/shifted平均](switch_time_20261107_policy_means.csv)。**T100対T80は逆向きの差**であり、登録仮説としてはNOT SUPPORTED。clean/shiftedの目的混合w=0.562/0.624は**結果後の探索閾値**で、実用閾値ではない。CSVの統計のみ再計算：python scripts/check_switch_time_boundary_20261107.py。全192モデルの生重み監査には、チャットで別途配布した完全証拠ZIPが必要。[結果・限界](../docs/SWITCH_TIME_BOUNDARY_RESULT_20261107.md)。

## 新しいOptimizer切替・初期Seed実験の公開統計（2026年10月31日、外部事前登録なし）

[主な5検定](switch_seed_20261031_primary_decisions.csv) と [12プール×5比較の差](switch_seed_20261031_pool_differences.csv)。自然対数のCE（無次元）で、正の差は対象方式の損失が低いことを意味する。**0.01 CEは正答率の1 ppではない**。統計単位は12環境群。`python scripts/check_switch_seed_results.py` によりHolm5・95%点別区間・符号反転感度解析を**記録済みペア差だけから**検算可能。checkpointと推論の再現はこのCSVのみではできない。[手法・陰性結果・事後解析](../docs/OPTIMIZER_SWITCH_SEED_RESULT_20261031.md)。

## 単位と反復

正答率・正答率差のCSVは割合です。`0.01`は1パーセントポイント。`rep`は学習反復の識別子で、同じ画像・環境群を評価した複数行を独立反復として数えません。信頼区間は各実験で固定した画像・環境群に条件づけたものです。標準化したmatchingスコアは無次元であり、正答率とは別の量です。

## 新しい6環境群の検算用CSV（Issue #124/#125）

| 試験 | 主なCSV | 比較の意味 |
|---|---|---|
| [PR #126：登録済み12検定](roadmap_validation_20260926_primary.csv) | [6環境群の72主要差分](roadmap_validation_20260926_pool_contrasts.csv) | Digitsのclean–minimumトレードオフ、少数Fashion画像での転用、objective×estimator×optimizer構成。区間は6環境群t5の点ごとの区間。確認は `python scripts/check_roadmap_validation_results.py`。 |
 
## 最近の確認試験と失敗

| 試験 | 主なCSV | 比較の意味 |
|---|---|---|
| Issue #91 平行移動を追加調整 | [primary30](translation_matched_primary30.csv)、[summary30](translation_matched_summary30.csv)、[balance30](translation_matched_balance30.csv)、[decision30](translation_matched_decision30.csv) | 高対低の参照選択履歴。対照条件と追加調整条件の両方を含む。 |
| Issue #91 座標別監査 | [coordinate secondary30](translation_coordinate_secondary30.csv) | 事前指定の副次的記述。要因ごとの分布まで一致したという主張ではない。 |
| Issue #89 厳密matchingの校正 | [calibration summary](translation_calibration_summary.csv) | 校正FAIL。held-out性能の比較ではない。 |
| Issue #86 物理的スコアによる代替採点 | [primary paired30](spatial_primary_paired30.csv)、[summary30](spatial_summary30.csv)、[balance30](spatial_balance30.csv)、[decision30](spatial_decision30.csv) | 勾配採点の高低と、平行移動採点の高低。参照モデルとの差は副次評価。 |
| Issue #83 損失・総多様性調整 | [paired30](parameter_matched_paired30.csv)、[座標別事後解析](parameter_composition_summary_posthoc.csv) | 高低差と、残る要因配分の探索的記述。 |

`translation_matched_primary30.csv`の`u_`は参照損失・総多様性のみを揃えた対照、`m_`は平行移動も揃える比較です。いずれも高novelty対低noveltyで、オンラインgradnov対loss-hardではありません。

小さいCSVだけで確認できる範囲は [check_public_results.py](../scripts/check_public_results.py) に明記しています。全matchingゲート・選択履歴・checkpoint・環境別再集計までの監査には [完全成果物](../docs/QUICKSTART.md) が別途必要です。検算PASSと科学的な仮説のPASSは異なります。

## CIFARの通常手法比較

[cifar_resnet_primary_all40.csv](cifar_resnet_primary_all40.csv) · [paired40](cifar_resnet_primary_paired40.csv) · [summary40](cifar_resnet_primary_summary40.csv)

40ペアのオンラインgradnov対loss-hardです。平均差+0.1206 ppはHolm補正後に支持されましたが、p10/minimumは未確認です。[元のプロトコル](../docs/CIFAR_RESNET_PRIMARY.md)を参照してください。

## 採用環境数を変える試験

| 対象 | 反復ごとの主要contrast | 採用数別集計 |
|---|---|---|
| SmallCNN | [cnn_budget_attenuation30.csv](cnn_budget_attenuation30.csv) | [cnn_budget_q_summary30.csv](cnn_budget_q_summary30.csv) |
| CIFAR / ResNet-20 | [cifar_budget_paired30.csv](cifar_budget_paired30.csv) | [cifar_budget_q_summary30.csv](cifar_budget_q_summary30.csv) |

Fashion・MLPを含む比較は [研究状況](../docs/RESEARCH_STATUS.md) と [Fashionの結果](../docs/FASHION_BUDGET_SCALING_RESULT.md) を参照してください。全候補使用時の自動的一致を除く事後解析は、元の事前登録した主要検定とは区別します。

## 過去の証拠も保持

[初期のkey findings](public_key_findings.csv)、[幾何変換のペア比較](geometric_paired.csv)、[勾配採点対パラメータ採点](gradient_vs_parameter_paired.csv)、[CNN初期比較](cnn_replication_paired20.csv)、[optimizer比較](optimizer_ablation_paired20.csv)は過去の系列です。詳細は [実験目次](../experiments/README.md) と [過去の結果まとめ](../docs/RESULTS.md) を参照してください。

representation-rankの予測記録、RNG圧縮、係数制御などの既存CSVも削除していません。rankの予測記録を、rankが性能の原因だという証拠にはしません。後続の否定的結果を含む [現在の解釈](../docs/RESEARCH_STATUS.md) を優先してください。

[CPU再現性の判定](cifar_cpu_repro_decision.csv)と[CPU条件](cifar_cpu_repro_runtime.csv)も保持しています。異なる実行の科学的な行を都合よく混ぜず、元の成果物と対応させてください。
