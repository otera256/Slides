#import "@preview/touying:0.6.1": *
#import themes.metropolis: *
#import "@preview/numbly:0.1.0": numbly

// --- 図解用ライブラリのインポート ---
#import "@preview/fletcher:0.5.8": diagram, node, edge, shapes
#import "@preview/lilaq:0.4.0" as lq

// --- 設定 ---
#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  footer: self => self.info.institution,
  config-info(
    title: [機械学習勉強会 第3.5回],
    subtitle: [数学的準備],
    author: [furakuta],
    date: datetime.today(),
    institution: [KMC],
  )
)

#set text(lang: "ja", font: "UD Digi Kyokasho NP")
#set heading(numbering: numbly("{1}.", default: "1.1"))

// 数式ブロックのスタイル
#show math.equation.where(block: true): it => block(inset: 10pt, radius: 5pt, fill: rgb("#f0f0f0"), width: 100%, it)

#title-slide()

== はじめに

このスライドは今後の機械学習勉強会で用いる「道具としての数学」について確認し、学部ごとの記法・用語の揺らぎを吸収することを目的としています。

期末考査もちょうど終わったところのはずなので、大半の内容はすでに知っているはずです。

= 線形代数

== ベクトルと行列

=== 体

