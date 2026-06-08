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
#let color-def-bg = rgb("#e3f2fd")
#let color-thm-bg = rgb("#fff3e0")
#let color-note-bg = rgb("#f1f8e9")
#let color-math-bg = rgb("#f0f0f0")

// --- 設定 ---
#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  footer: self => self.info.institution,
  config-info(
    title: [チャットAIの仕組み],
    subtitle: [2026年　機械学習分野　新勧講座],
    author: [49th furakuta],
    date: [2026-04-17],
    institution: [#image("../images/KMClogo_trans.png", height: 2em)],
  ),
  config-colors(
    primary: color-primary,
    primary-light: color-primary-light,
    secondary: color-secondary,
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


== ロボットと喋りたい！ (歴史)
古くから、人間は「機械と言葉を交わすこと」を夢見てきました。

- *ELIZA (1966年)*: 
  - 世界初のチャットボット。
  - 基本は「キーワードに反応して言い換えるだけ」の*ルールベース*。
  - 例：ユーザー「悲しいです」 → ELIZA「なぜ悲しいと思うのですか？」
- *手抜きチャットボットのデモ*:
  - 自作の「超手抜きチャットボット」をお見せします。
  - 特定の単語にしか反応できない不自由さを体感！

---

== ルールベースの限界
#grid(columns: (1fr, 1fr), column-gutter: 1em)[
  - *柔軟性がない*:
    - 登録されていない言葉には「分かりません」しか言えない。
  - *文脈が読めない*:
    - 前の会話を覚えるのが非常に困難。
  - *メンテナンス不能*:
    - あらゆる会話のパターンを人間が手書きするのは不可能。
][
  #note-box(title: "限界の例")[
    ユーザー：
    「昨日のリンゴは美味しかったけど、今日のはちょっと...」
    
    AI：
    「リンゴ」というキーワードしか見ていないので
    「リンゴの産地を教えましょうか？」と返してしまう。
  ]
]

---

== チャットAIの正体：「次トークン予測」
現代のAI（ChatGPTなど）は、実は驚くほど単純なタスクを解いています。

#align(center)[
  #block(stroke: 2pt + color-primary, inset: 20pt, radius: 10pt)[
    *「これまでの文脈を読んで、次に来る最も確率の高い単語を当てる」*
  ]
]

#pause

=== 皆さんも予測してみてください：
「むかしむかし、あるところに、おじいさんとおばあさんが #text(fill: color-primary)[ *[？]* ] 」

#pause
- 「住んでいました」？ 「いました」？ 「暮らしていました」？
- 確率は違えど、どれも正解候補です。

---

== 統計的な言語モデル (n-gram)
最も古典的な「予測」の方法は、単なる「カウント」です。

- *n-gramモデル*: 
  - 過去の $n-1$ 個の単語を見て、次に来る単語の確率を計算する。
  - 例：2-gram (bigram)
    - 「吾輩は」の後に「猫」が来る回数を、全テキストから数えるだけ。

#def-box(title: "デモ：夏目漱石/太宰治モデル")[
  - $n=2$ のとき：文法はめちゃくちゃだが、なんとなく言葉が繋がる。
  - $n=5$ のとき：元の文章の「コピペ」になってしまう。
]

---

== カウントベースの限界
1. *「未知の組み合わせ」に弱い*:
   - 一度も見たことがない単語の並びの確率は 0 になってしまう。
2. *記憶力が短い*:
   - $n$ を大きくすると、計算量が爆発し、かつ単なる暗記になる。
3. *意味を理解していない*:
   - 「猫」と「子猫」が似ているという知識を一切持っていない。

---

== 単語を「数字（ベクトル）」にする (word2vec)
ニューラルネットワークは数字しか扱えません。そこで、単語を多次元空間の点（座標）で表します。

word2vec(王) $= [1.7, -2.1, 0.1, -0.3, ..., 0.5]$\
word2vec(女) $= [1.5, -1.8, 0.2, -0.4, ..., 0.6]$\
word2vec(男) $= [1.6, -2.0, 0.0, -0.2, ..., 0.4]$


#align(center)[*王 - 男 + 女 = 女王様* ！？]

---

== 学習：確率を「改善」し続ける
AI（ニューラルネットワーク）は巨大な微分可能な関数です。

1. *予測*: 前の単語から、次の単語の確率分布 $y$ を出す。
2. *ショック (損失)*: 実際に出現した単語とのズレを前回習った*Cross-Entropy*で計算。
3. *改善*: 勾配降下法を使って、予測を少しずつ正解に近づける。

#note-box(title: "ここがポイント")[
  n-gram（ただのカウント）と違い、ニューラルネットワークは*「似た意味の単語」を活用して、見たことがない文脈でも予測できる*ようになります。
]

---

== 自作モデルの「鑑賞会」
実際に私が作成した、少し小さな言語モデルが生成した文章を見てみましょう。


=== 考えてみよう
- この文章は「自然」ですか？
- どこかに「不自然さ」や「コピペ感」はありますか？

---

== まとめ
- チャットAIの基本は*「次に来る言葉の予測」*。
- 昔は「ルール（人間が書く）」、今は「学習（データから勝手に見つける）」。
- ニューラルネットワークは、単語を「意味のある数字」として扱い、文脈を捉える。

=== 次回予告
*「実際のロボットアームを動かしてみよう」5/8 (木)*
- 画面の中の知識が、どうやって現実世界（物理）の動きに繋がるのか。
- 数学とハードウェアが交差する瞬間をお楽しみに！

---

== 質疑応答
#align(center + horizon)[
  質問は何でも歓迎です！
  
  例：
  - 「ChatGPTと今回のモデルは何が違うの？」
  - 「どうやってプログラミングの勉強を始めたらいい？」
  - 「GPUってやっぱり必要？」
]