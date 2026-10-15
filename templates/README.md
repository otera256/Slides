# KMC Typstスライドテーマ

`kmc.typ` は shinkan/Apr24.typ のMetropolisテーマと配色を共通化したものです。

- アクセント：オレンジ `#ce6636`
- 見出し：濃紺 `#0e0b31`
- 本文：`#26383e`
- 背景：`#fafafa`
- フォント：`UD Digi Kyokasho NP`

## 使用例

各発表フォルダから読み込む例です。

```typst
#import "@preview/touying:0.7.0": *
#import themes.metropolis: title-slide, slide
#import "../templates/kmc.typ": *

#show: kmc-theme.with(
  title: [発表タイトル],
  subtitle: [サブタイトル],
  author: [49th furakuta],
  date: [2026-10-15],
  institution: image("../images/KMClogo_trans.png", height: 1.4em),
)

#title-slide()

#slide(title: [スライドの見出し])[
  - 説明する内容
  #takeaway[特に伝えたいこと]
]
```

共通関数：`emphasis`（強調）、`caption`（本文サイズの補足）、`reference-caption`（11ptの出典・参考文献）、`takeaway`（罫線を使わない太字の要点）、`columns`（2列）、`plain-table`（表）、`image-slide`（画像専用の独立スライド）。`colors` はCeTZ等からもインポートできます。

`image-slide` は画像領域を12.5emの高さで確保し、その下に本文サイズの説明と出典を配置します。画像や複数画像のグリッドを本文に渡せます。写真の縦横比は `fit: "contain"` で維持してください。出典の書式は呼び出し側で指定します。

```typst
#image-slide(
  title: [実際の製品],
  description: [画像から確認してほしい特徴。],
  credit: text(size: 11pt, fill: colors.muted)[
    画像引用：掲載者「#link("https://example.com/")[資料名]」（閲覧日）
  ],
)[
  #image("product.png", width: 100%, height: 12em, fit: "contain")
]
```

`columns` の幅は `widths: (2fr, 1fr)` のように指定します。本文には20pt、表には18pt、図の主要ラベルには16〜22pt程度を用い、数式・図の詳細は別ファイルへ分ける想定です。

新しい発表でこのテーマを使う場合も、プロジェクトルートを `typst compile --root . ...` で指定してください。
