# Xで紹介するための文章

[公開リポジトリ](https://github.com/Unjuno/seed-guided-optimization) · [日本語README](../README.md) · [現在の研究状況](RESEARCH_STATUS.md)

このページは公開時の表現を固定するためのメモです。投稿は自動では行いません。

## 推奨する最初の投稿

```text
機械学習実験「Seed-Guided Optimization (SGO)」を公開しました。

学習時のランダムな画像変換を、損失と勾配方向を見ながら有限候補から選ぶと何が変わるかを調べています。

効いた条件だけでなく、外れた仮説・失敗した事前登録試験・結果CSV・検算コードも公開しています。
https://github.com/Unjuno/seed-guided-optimization
```

この文面は、現在の証拠が支持する「training-environment selection policy の条件付き効果」に限定しています。seed整数そのものの普遍的な良し悪しを主張していません。

## 数値を紹介する場合

異なる比較を一つの性能ランキングとして混ぜないでください。特に次の二つは別の問いです。

```text
通常のオンライン比較では、CIFAR-10 / ResNet-20 の40ペアで gradnov が loss-hard より平均 +0.1206 pp。

別の機構介入では、Digits / SmallCNN の30ペアで、損失・総多様性・平行移動を許容差内に調整した高novelty履歴と低novelty履歴の差が +2.041 pp（95% CI +0.862〜+3.219 pp）。

後者は通常のSGO対loss-hard比較ではなく、原因を切り分けるための別実験です。
```

数値を投稿するときは、必ず「何対何の差か」「task/model」「反復数」を併記してください。最新の解釈は [RESEARCH_STATUS](RESEARCH_STATUS.md) を優先します。

## 現時点で避ける表現

次の表現は現在の証拠を超えます。

- 「普遍的な最適化法則を発見した」
- 「良いseedと悪いseedを整数だけから判定できる」
- 「gradient novelty が唯一の原因だと証明した」
- 「worst-case robustness を保証する」
- 「同じ精度をより少ない総計算量・wall-clockで達成する」
- 「GPUで高速化する」
- 「既存optimizerへそのまま差し込める完成ライブラリである」

## GitHub公開メタデータ

GitHubのAbout欄はREADMEより先に読まれる場合があるため、本文と同じ主張境界にしてください。

推奨Description:

```text
Experimental study of loss- and gradient-aware selection among stochastic training environments, with preregistered tests, failures, public result CSVs, and reproducibility checks.
```

推奨Topics:

```text
machine-learning
deep-learning
stochastic-optimization
gradient-diversity
data-augmentation
pytorch
reproducibility
experimental-design
cifar10
```

`robust-learning`、`speedup`、`efficient-training` のように、未確認の性能保証を連想させるtopicは現時点では付けません。

## 公開時の参照先

実験の入口は [README](../README.md)、最新の結果と未解明点は [研究状況](RESEARCH_STATUS.md)、理論上の区別は [THEORETICAL_FRAMEWORK](THEORETICAL_FRAMEWORK.md)、失敗した試験は [厳密matchingの校正](TRANSLATION_MATCH_CALIBRATION_RESULT.md)、公開値は [results](../results/README.md)、軽量な再集計は [QUICKSTART](QUICKSTART.md) に対応しています。
