# Seed-Guided Optimization（SGO）

**学習中のランダムな画像変換を、損失と勾配方向を見て選んだら何が変わる？**

個人の趣味として進めている、機械学習の実験ノートです。うまくいった結果だけでなく、外れた仮説、実験前に決めた判定条件、結果CSV、検算コードも公開しています。

[English](README.en.md) · [実験と結果を読む](docs/README.md) · [再学習せず検算する](docs/QUICKSTART.md) · [現在の研究状況](docs/RESEARCH_STATUS.md)

## 何をしているのか

例えば、16種類の画像変換が候補にあり、4種類だけをモデルの更新に使うとします。

**loss-hard** は、現在のモデルが苦手な変換を優先します。**gradnov** は、難しさに加えて「モデルを修正しようとする勾配方向が似すぎていないか」も見て選びます。比較するのは、この選び方が学習結果を変えるかどうかです。

seedは画像変換を再現するための番号です。**seedの整数そのものに、普遍的な当たり・外れがあるという研究ではありません。** また、更新に使う環境数を揃えることと、実行時間・総計算量を揃えることは別です。

## 今のところ分かったこと

[Digits・FashionMNIST・CIFAR-10の実験](docs/RESEARCH_STATUS.md)では、選び方によって平均正答率が改善する条件が見つかっています。ただし、すべてのデータやモデルで効くわけではなく、**なぜ効くかは引き続き調べています。**

| どの実験か | 何と何を比較したか | 結果と読める範囲 |
|---|---|---|
| [CIFAR-10 / ResNet-20、40ペア](docs/CIFAR_RESNET_PRIMARY.md) | 学習中に選ぶgradnov 対 loss-hard | 平均正答率 **+0.1206 pp**。小さい改善で、最悪条件の性能改善は未確認。 |
| [Digits / SmallCNN、30ペア](docs/OVERLAP_TRANSLATION_RESULT.md) | 損失・総多様性・平行移動のばらつきを許容差内に揃えた、高novelty履歴 対 低novelty履歴 | 平均正答率 **+2.041 pp**、95%区間 **+0.862〜+3.219 pp**。通常のSGO対loss-hardとは別の介入実験。 |
| [厳密な平行移動matchingの校正](docs/TRANSLATION_MATCH_CALIBRATION_RESULT.md) | 全ステップで比較条件を揃えられるか | **失敗**。条件不足のまま性能評価へ進まず、失敗として保存。 |

**ppはパーセントポイント**です。正答率80%から81%への変化が+1 ppで、相対改善率とは異なります。モデル、学習枚数、optimizer、CPU条件、検定方法は各結果文書に記載しています。異なる比較の数値を並べて手法の優劣を決めることはできません。

最近は「多様性の総量」だけでなく、**平行移動・回転・ノイズなど、どの変動要因に学習機会を配分するか**を調べています。[平行移動だけで採点する実験](docs/SPATIAL_ALLOCATION_RESULT.md)でも高低差が出ましたが、勾配の性質も同時に変わるため、原因の切り分けはまだ途中です。

## 最初に試す：公開CSVの検算

GPU、画像データ、学習済みモデルのダウンロードは不要です。下の手順は、既に公開した30ペアの数値を再集計するだけで、学習は実行しません。

```bash
git clone https://github.com/Unjuno/seed-guided-optimization.git
cd seed-guided-optimization
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-check.txt
python scripts/check_public_results.py
```

Windowsでの環境作成・有効化、出力の読み方、他の検算、学習実験への入口は [QUICKSTART](docs/QUICKSTART.md) にまとめています。

> `PUBLIC STATISTICS VERIFIED` は、公開CSVの平均・区間・p値などが一致したという意味です。画像からの再学習、成果物全体のハッシュ検査、研究仮説の新たな確認を行うコマンドではありません。

## コードと記録の場所

| 読みたいもの | 入口 |
|---|---|
| 現在分かっていること・分からないこと | [研究状況](docs/RESEARCH_STATUS.md) / [理論と限界](docs/THEORETICAL_FRAMEWORK.md) |
| 実験別の結果と失敗の記録 | [文書目次](docs/README.md) |
| 実装と実行手順 | [experiments](experiments/README.md) |
| 結果CSVと比較の定義 | [results](results/README.md) |
| 感想・疑問・再現報告 | [Issues](https://github.com/Unjuno/seed-guided-optimization/issues) / [参加方法](CONTRIBUTING.md) |

## まだ言えないこと

**普遍的な最適化法則、勾配noveltyだけが原因という説明、最悪条件での性能保証、GPUでの高速化は示していません。** 後半の機構実験はDigitsの画像集合を再利用しています。新しいseedでの反復と、新しいデータセットでの再現は区別しています。CPUやbackendが変わると、同じseedでも完全に同じ数値にならない場合があります。

実験用スクリプトの集まりであり、既存の学習へそのまま差し込める完成済みoptimizerではありません。詳細は [研究状況](docs/RESEARCH_STATUS.md) を参照してください。

## ライセンス

コードは [Apache-2.0](LICENSE)。引用情報は [CITATION.cff](CITATION.cff)。外部データセットの利用条件は各配布元に従ってください。
