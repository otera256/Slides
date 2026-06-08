#import "@preview/touying:0.7.0": *
#import themes.metropolis: *
#import "@preview/numbly:0.1.0": numbly
#import "@preview/algorithmic:1.0.7"
#import algorithmic: algorithm-figure, style-algorithm
#show: style-algorithm
#import "@preview/theorion:0.4.1": *
#import "@preview/cetz:0.4.2"
#import "@preview/physica:0.9.3": *


// --- 図解用ライブラリのインポート ---
#import "@preview/fletcher:0.5.8": diagram, node, edge, shapes
#import "@preview/lilaq:0.4.0" as lq

// --- 色の定義 ---
#let color-primary = rgb("#ce6636")
#let color-primary-light = rgb("#dc997a")
#let color-secondary = rgb("#0e0b31")
#let color-secondary-lighter = rgb("#2c2a6c")
#let color-secondary-light = rgb("#4a4780")
#let color-secondary-lightest = rgb("#6a66a0")
#let color-def-bg = rgb("#e3f2fd")
#let color-thm-bg = rgb("#fff3e0")
#let color-note-bg = rgb("#f1f8e9")
#let color-math-bg = rgb("#f0f0f0")

// --- 設定 ---
#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  footer: self => self.info.institution,
  config-info(
    title: [難解言語Brainfuckで遊ぼう],
    subtitle: [+++++++++\[>++++++++>+++++++++++>+++>+<<<\<-\]>.>++.+++++++..+++.
>+++++.<<+++++++++++++++.>.+++.------.--------.>+.>+.],
    author: [49th furakuta],
    date: [2026-04-24],
    institution: [#image("../images/KMClogo_trans.png", height: 2em)],
  ),
  config-colors(
    primary: color-primary,
    primary-light: color-primary-light,
    secondary: color-secondary,
    secondary-lighter: color-secondary-lighter,
    secondary-light: color-secondary-light,
    secondary-lightest: color-secondary-lightest,
    
  )
)

#set text(lang: "ja", font: "UD Digi Kyokasho NP")
#set heading(numbering: numbly("{1}.", default: "1.1"))

#let slide-block-render(color) = (prefix: none, title: "", full-title: auto, body) => {
  block(
    fill: color,
    inset: 0.6em,
    radius: 0.2em,
    width: 100%,
    stroke: (left: 4pt + color.darken(20%)),
    below: 1em,
    [
      #strong[#full-title.] #h(0.5em) #body
    ]
  )
}

// 定義 (Definition): 青系
#let (def-counter, def-box, definition, show-definition) = make-frame(
  "definition", "Definition",
  render: slide-block-render(color-def-bg)
)
#show: show-definition

// 定理 (Theorem): オレンジ系
#let (thm-counter, thm-box, theorem, show-theorem) = make-frame(
  "theorem", "Theorem",
  render: slide-block-render(color-thm-bg)
)
#show: show-theorem

// 重要・注釈 (Note/Alignment): グレー/緑系
#let (note-counter, note-box, note, show-note) = make-frame(
  "note", "Note",
  render: slide-block-render(color-note-bg)
)
#show: show-note


// 数式ブロックのスタイル
#show math.equation.where(block: true): it => block(inset: 10pt, radius: 5pt, fill: color-math-bg, width: 100%, it)

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

= Preliminary

== KMCってどんなサークル?

- 正式名称: *京都大学マイコンクラブ*
#pause
- 設立は1977年で、来年で50周年を迎える
#pause
- 活動場所
  - 出町柳駅付近の部室（ぼろい）
  - Slack: オンラインでの主な活動場所
    - 月間アクティブユーザーは約200人
  - DiscordやCosenseなどのサービスも併用
#pause
- 活動内容
  - ソフトウェア: Webサービス開発、機械学習、競技プログラミングなど
  - ハードウェア: 電子工作、サーバー管理、インフラ構築
  - パソコンを用いた創作活動: ゲーム開発、イラスト制作、音楽制作
#pagebreak()
#note-box(title: "マイコン(Mincrocomputer)とは")[
  
  創部当時（1970年代）は、コンピュータと言えば部屋を丸ごと占領するようなものや大きな冷蔵庫のようなサイズのものが主流だった

  #pause
  そんな中、個人が所有できる小型のコンピュータ（マイコン）が登場し、このコンピュータで遊び始めたのがマイコンクラブの始まり\
  意味合い的には現代のパソコンに近い

  #pause
  Minro controller（マイクロコントローラー）とは別物なので注意
]

#slide[
  #align(center)[
    #image("../images/bushitu.jpg")
  ]
][
  #align(center)[
    #image("../images/bushitu2.jpg")
  ]
]

== 自己紹介
#slide(composer: (2fr, 1fr))[
  - kmc_id: furakuta
  - 所属: 京都大学工学部情報学科数理工学コース2回生
  - KMC49代副会長
  - KMCでの活動分野
    - 機械学習: 応用よりも基礎理論中心
    - ゲーム開発: みんゲーや個人でちょっと作ってる
    - 競技プログラミング: 週1でABCに出ているくらい(大学入ってから始めたのでまだまだ初心者)
  - ひとこと: 超かぐや姫良かった
][
  #align(center)[
    #image("../images/furakuta.png", height: 10em)
  ]
]

== 他の方の自己紹介
当日出席している部員の自己紹介を簡単に

参加者の人数を見てできれば参加者の自己紹介も
- 所属
- 興味のある分野
- ひとこと

= Brainfuckとは?

#focus-slide[
  プログラミング言語は覚えるのが大変！！
]

#image-slide[
  #align(center)[
    #image("images/difficult_python.png")
  ]
]

== Brainfuckのすべての文法
#align(center)[
  #set text(size: 1em)
  #table(
    columns: (1.3em, auto),
    `>`, [ポインタを右に移動],
    `<`, [ポインタを左に移動],
    `+`, [ポインタが指すセルの値を1増やす],
    `-`, [ポインタが指すセルの値を1減らす],
    `.`, [ポインタが指すセルの値をASCIIコードとみなして出力],
    `,`, [入力から1文字読み取り、ポインタが指すセルにそのASCIIコードを格納],
    `[`, [ポインタが指すセルの値が0なら、対応する`]`の次の命令にジャンプ],
    `]`, [ポインタが指すセルの値が0でないなら、対応する`[`の次の命令にジャンプ]
  )
  #pause
  #text(size: 1.25em)[
    *たった8つの命令で構成される、超シンプルなプログラミング言語*
    #pause
    それでいて他の高級な言語と同じくチューリング完全であるため、理論上はどんな計算も行うことができる
  ]
]

== 実行環境
Brainfuckの実行環境は、通常以下のような構成になっている
- メモリ: 無限に続くセルの列。各セルは0で初期化されている(通常は30000セル程度が用意されることが多い)
- ポインタ: メモリ上のセルを指すポインタが1つ存在し、最初は最初のセルを指している
- 入出力: ポインタが指すセルの値をASCIIコードとみなして入出力を行う
- セルの値は通常0から255の範囲で、256を超えると0に戻り、0未満になると255に戻る（オーバーフロー/アンダーフロー）これは処理系によって異なる場合が多い

今日は#link("https://poyo.me/bf/webbf/")[*ブラウザ上で動くBrainfuckインタプリタ*]を使用して実行する

== コード例
=== "A"を出力するコード
愚直な方法:
"A"のASCIIコードは65(16進法で41)なので、ポインタが指すセルの値を65にしてから出力すれば良い
```
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++.
```

掛け算を利用する方法:
```
++++++++       // セル0を8にする
[
  >++++++++    // セル1に8を加える（ループが8回繰り返されるので、最終的にセル1は64になる）
  <-           // セル0を1減らす
]              // セル0が0になるまでループする
>+             // セル1を1増やす（セル1は65になる）
.              // セル1の値を出力（"A"が出力される）
```

#pagebreak()
=== "Hello, World!"を出力するコード
```
+++++++++      // セル0を9にする
[
  >++++++++    // セル1に8を加える（ループが9回繰り返されるので、最終的にセル1は72になる）
  >+++++++++++ // セル2に12を加える（ループが9回繰り返されるので、最終的にセル2は108になる）
  >+++         // セル3に3を加える（ループが9回繰り返されるので、最終的にセル3は27になる）
  >+           // セル4に1を加える（ループが9回繰り返されるので、最終的にセル4は9になる）
  <<<<-        // セル0を1減らす
]
>.>++.+++++++..+++.>+++++.
<<+++++++++++++++.>.+++.------.--------.>+.>+.
```

== ASCIIコード表
#align(center)[
  #image("images/ascii_table.png", height: 90%)
]
#align(right)[
  #text(size: 0.8em, fill: luma(80))[
    https://e-words.jp/p/r-ascii.html より引用
  ]
]

#focus-slide[
  自分の好きな文字を出力するコードを書いてみよう！
]

== ちなみに私のidだと...
f(102), u(117), r(114), a(97), k(107), u(117), t(116), a(97) となり、dama(DNEK)氏による現在最短のコード(66文字)だと以下のようになる。すごい
```
>->+>+>--
[
  >+++
  [-<++++++>]
  <
  [->++++++<]
  <
]
>>.>+++.>.>+.<++<<+++++[.>]
```
`[-1, 1, 1, -2].map(i => (i + 3 * 6) * 6)`みたいな処理をしている感じらしい。\
すごい

= 応用編
== 入力
`,`命令を使うと、ユーザーからの入力を受け取ることができる。

=== 入力された文字をそのまま出力するコード
```
+[>,.<]
```

=== 入力された1桁の数字の回数だけ"Hello"を出力するコード
1. 入力された文字を数字に変換するために、ASCIIコードの"0"（48）を引く
2. ループを使って、変換された数字の回数だけ"Hello"を出力する
```
++++++[>++++++++<-] // セル1を48にする（ASCIIコードの'0'）
,                   // ユーザーからの入力を受け取る（セル0に格納される）
>[-<->]             // セル0の値からセル1の値を引く
>
```