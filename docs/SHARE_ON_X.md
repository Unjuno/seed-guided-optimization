# Xで紹介するための文章

[公開リポジトリ](https://github.com/Unjuno/seed-guided-optimization) · [日本語README](../README.md)

以下は投稿用の下書きです。自動投稿は行いません。

## そのまま使う文面

```text
趣味で進めている機械学習実験「SGO」の記録をまとめました。
学習時のランダムな画像変換を、損失と勾配方向を見て選ぶ試みです。
効いた条件も、外れた仮説も、コード・結果CSVと一緒に残しています。
https://github.com/Unjuno/seed-guided-optimization
```

## 紹介文の根拠

実験の説明は [README](../README.md)、最新の結果と未解明点は [研究状況](RESEARCH_STATUS.md)、失敗した試験は [厳密matchingの校正](TRANSLATION_MATCH_CALIBRATION_RESULT.md)、公開値は [results](../results/README.md) に対応しています。

数値を追加して紹介するときは、「何対何の差か」を添えてください。CIFARの通常手法比較と、Digitsの高novelty対低noveltyの選択履歴実験は違う比較です。「普遍的な最適化法則を発見」「勾配noveltyが唯一の原因」「GPUで高速化」といった説明は、現在の証拠を超えます。
