#import "@preview/touying:0.6.1": *
#import themes.metropolis: *
#import "@preview/numbly:0.1.0": numbly
#import "@preview/algorithmic:1.0.7"
#import algorithmic: algorithm-figure, style-algorithm
#show: style-algorithm
#import "@preview/theorion:0.4.1": *

// --- 図解用ライブラリのインポート ---
#import "@preview/fletcher:0.5.8": diagram, node, edge, shapes
#import "@preview/lilaq:0.4.0" as lq

// --- 設定 ---
#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  footer: self => self.info.institution,
  config-info(
    title: [Gitの知っておくと便利な機能],
    author: [49th furakuta],
    date: [2026-01-12],
    institution: [KMC],
  )
)

#set text(lang: "ja", font: "UD Digi Kyokasho NP")
#set heading(numbering: numbly("{1}.", default: "1.1"))

// 数式ブロックのスタイル
#show math.equation.where(block: true): it => block(inset: 10pt, radius: 5pt, fill: rgb("#f0f0f0"), width: 100%, it)

#let image-slide(body, background: none) = touying-slide-wrapper(self => {
  self = utils.merge-dicts(
    self,
    config-page(
      background: place(right, background),
      margin: 0em,
    ),
  )
  set image(width: auto, height: auto)
  touying-slide(self: self, align(bottom, body))
})

#title-slide()
