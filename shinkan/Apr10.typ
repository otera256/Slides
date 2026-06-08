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
    title: [手書き文字認識],
    subtitle: [2026年　機械学習分野　新勧講座],
    author: [49th furakuta],
    date: [2026-04-10],
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

== 注意
- 高校数学レベルの知識を前提としているため、詳しい人にとっては物足りなかったり正確さに欠ける内容になるかも
- そもそもあまり時間がないので、雰囲気だけわかってもらえれば数式がよくわからなくても大丈夫
- どうしても内容がないようなので大学の講義みたいになってしまうのでちょっと眠たくなるかも
- 途中で質問があれば気軽にどうぞ

= What is Machine Learning?
== 定義
なんか猫も杓子もAIとか機械学習とか言ってるけどそもそも何？
#pause

#block(
  fill: luma(230),
  inset: 12pt,
  radius: 8pt,
)[
  人工知能は、「計算（computation）」という概念と「コンピュータ（computer）」という道具を用いて「知能」を研究する計算機科学（computer science）の一分野である。誤解を恐れず平易にいいかえるならば、「これまで人間にしかできなかった知的な行為（認識、推論、言語運用、創造など）を、どのような手順（アルゴリズム）とどのようなデータ（事前情報や知識）を準備すれば、それを機械的に実行できるか」を研究する分野である。
  #align(right)[
    #text(size: 0.8em, fill: luma(70))[
      ［佐藤理史　2018年6月19日　日本大百科全書（ニッポニカ）］より引用
    ]
  ]
]

#pagebreak()

#align(center)[
  #cetz.canvas({
    import cetz.draw: *
    
    // 配色の定義 (グローバルの色定義を利用)
    let color-ai = color-def-bg
    let color-ml = color-thm-bg
    let color-dl = color-primary-light
    let color-text-dark = color-secondary

    // 全体のサイズ感を (14, 10) に調整して「縦」を確保
    
    // --- 人工知能 (AI) 領域 ---
    rect(
      (0, 0), (20, 12), 
      fill: color-ai, 
      stroke: color-text-dark + 1.5pt, 
      radius: 5pt, 
      name: "ai"
    )
    content(
      "ai.north-west", 
      anchor: "north-west", 
      padding: 0.4, 
      text(fill: color-text-dark, weight: "bold", size: 1.3em)[人工知能 (AI)]
    )
    
    // AIキーワード：上部に配置
    content((13.5, 10.5), text(size: 1em, fill: color-text-dark.lighten(20%))[エキスパートシステム])
    content((13.5, 9.5), text(size: 1em, fill: color-text-dark.lighten(20%))[A\* アルゴリズム])

    // --- 機械学習 (ML) 領域 ---
    rect(
      (0.5, 0.5), (19.5, 8.5), // 高さを ML の文字が見える位置まで確保
      fill: color-ml, 
      stroke: (paint: color-primary, thickness: 1pt, dash: "dashed"), 
      radius: 5pt, 
      name: "ml"
    )
    content(
      "ml.north-west", 
      anchor: "north-west", 
      padding: 0.4, 
      text(fill: color-primary, weight: "bold", size: 1.3em)[機械学習 (ML)]
    )

    // MLキーワード：DLの箱の上に配置
    content((13.5, 7.5), text(size: 1em, fill: color-primary)[パーセプトロン])
    content((13.5, 6.7), text(size: 1em, fill: color-primary)[決定木])

    // --- 深層学習 (DL) 領域 ---
    rect(
      (1, 1), (19, 5.8), // ここに十分な高さを与える
      fill: color-dl, 
      stroke: white + 1.2pt, 
      radius: 5pt, 
      name: "dl"
    )
    content(
      "dl.north-west", 
      anchor: "north-west", 
      padding: 0.4, 
      text(fill: white, weight: "bold", size: 1.3em)[深層学習 (Deep Learning)]
    )

    // DLキーワード：中央付近にゆったり配置
    content((10, 3.0), text(fill: white, weight: "bold", size: 1.1em)[
      CNN #h(1em)
      RNN #h(1em)
      Transformer
    ])
  })
]

#pagebreak()
#align(center)[#text(1.5em)[
  機械学習を学ぶモチベーション
  #pause

  $arrow.b$

  ブラックボックスの中身を知れる・応用できる
  ]
  
  #pause
  (情報学科の人は予習にもなる)
]

== アイスクリームおいしい
機械学習はモデリングの１種であるから、まずは月平均気温とアイスクリームの売上の関係をモデル化してみよう！！
#align(center)[
  #table(
    columns: (1fr, 1fr, 1fr),
    [モデルに必要なもの], [アイスクリームの例], [手書き文字認識の場合],
    [入力], [月平均気温], [各ピクセルの値\
    ($28 times 28 = 784$次元)],
    [出力], [アイスクリームの売上\ (連続的・回帰タスク)], [認識した文字の確率分布\ (離散的・分類タスク)],
    [モデル], [線形回帰・重回帰], [ニューラルネットワーク],
    [評価基準], [平均二乗誤差 (MSE)], [クロスエントロピー損失]
  )
]

#pagebreak()
#align(center)[
  #image("images/icecream-plot.png")
]

#pagebreak()
#align(center)[
  #text(1.2em)[
    入力に対して出力を予測してくれる関数（モデル）を作りたい\
    #pause
    $arrow.b$\
    どれだけ出力がうまく行っているかを評価する基準が必要
  ]
]

#def-box(title: "平均二乗誤差")[
  $ "MSE" = 1 /N sum_(i=1)^N (y_i - t_i)^2 $
  $y_i$: モデルの予測値 
  $t_i$: 正解ラベル
]

#pagebreak()
MSEを最小化するようなモデルが良い

モデルがただの直線($y = w x + b$)だとすると、
$
  "MSE"(w, b) = 1 /N sum_(i=1)^N (w x_i + b - t_i)^2
$
#pause
ややこしく見えるがこれはただの$w$と$b$の二次関数である

平方完成でもできるんですが...
#pause
凸であることがわかっているので微分が0になる点を求めれば最小値がわかる

#pagebreak()
$
  (partial "MSE") / (partial w) &= 1 /N sum_(i=1)^N 2 x_i (w x_i + b - t_i) = 0 \
  (partial "MSE") / (partial b) &= 1 /N sum_(i=1)^N 2 (w x_i + b - t_i) = 0
$
この連立方程式を解けば最適な$w$と$b$が求まる
$
  w = (N sum_(i=1)^N x_i t_i - sum_(i=1)^N x_i sum_(i=1)^N t_i) / (N sum_(i=1)^N x_i^2 - (sum_(i=1)^N x_i)^2) \
  b = (sum_(i=1)^N t_i - w sum_(i=1)^N x_i) / N
$

#pagebreak()
これは*1変数*の入力と*1変数*の出力であったが、*多変数*の入力と*多変数*の出力の場合はどうなるだろうか？
#pause

入力を $vb(x) in RR^n$ 、出力を $vb(y) in RR^m$ とすると、モデルは $vb(y) = W vb(x) + vb(b)$ という形になる（重回帰）
このとき、MSEは以下のようになる
$  "MSE"(W, vb(b)) = 1 /N sum_(i=1)^N || W vb(x_i) + vb(b) - vb(t_i) ||^2 $
このMSEを最小化するような$W$と$vb(b)$を求めることができる（解析的に解ける）

重みの行列、学習データの入力、出力をそれぞれ$W in M_(m times n) (RR), X in M_(N times n) (RR), T in M_(N times m) (RR)$とおくと、最適な$W$と$vb(b)$は以下のように求まる（1回生の方は理解できなくても大丈夫です）
#pagebreak()
$
  (partial "MSE") / (partial W) &= 1 /N sum_(i=1)^N 2 (W vb(x_i) + vb(b) - vb(t_i)) vb(x_i)^T = 0 \
  (partial "MSE") / (partial vb(b)) &= 1 /N sum_(i=1)^N 2 (W vb(x_i) + vb(b) - vb(t_i)) = 0 \
  hat(W) &= (X^T X)^(-1) X^T T \
  hat(vb(b)) &= 1/N sum_(i=1)^N (vb(t_i) - W vb(x_i))
$


ちなみにこの重回帰の面白い使い方として、入力の次元を増やす（特徴量エンジニアリング）ことで非線形な関係も表現できるようになる
例えば、入力に $x_1^2$ や $x_1 x_2$ といった項を追加すれば、非線形な関係も表現できるようになる

#align(center)[
  #image("images/icecream-double.png")
]

= MNIST
== MNISTってなぁに
手書き数字の画像データセット。機械学習の「Hello World」。

#grid(
  columns: (1fr, 1.5fr),
  [
    - *内容*: 0から9までの手書き数字画像
    - *画像サイズ*: $28 times 28$ ピクセル (グレースケール)
    - *データ数*:
      - 訓練データ: 60,000枚
      - テストデータ: 10,000枚
    - *入力*: 784次元のベクトル ($28 times 28$を平坦化)
    - *出力*: 10クラスの確率
  ],
  align(center + horizon)[
    // 簡易的なMNISTのイメージ図
    #image(
      "images/MnistExamples.png",
      width: 480pt,
      alt: "MNIST Dataset Examples",
    )
  ]
)

== Softmaxとクロスエントロピー損失

これまでの線形回帰では出力が連続的な値だったが、MNISTのような分類タスクでは出力は離散的なクラスの確率分布になる

このとき、出力層の活性化関数としてSoftmaxを使い、損失関数としてクロスエントロピー損失を使うのが一般的。分類タスクの分類の数（今回なら10クラス）を$C$とすると、Softmaxは以下のように定義される
$
  "Softmax"(z_i) &= exp(z_i) / (sum_(j=1)^C exp(z_j)) \
  "CrossEntropy"(y, t) &= - sum_(i=1)^C t_i log(y_i)
$

=== Softmaxの仕組み
- まず、出力層の線形変換の結果を $z$ とする
- Softmaxは $z$ の各要素を指数関数で変換し、すべて正の値にする
- さらに、すべての要素の合計で割ることで、出力が確率分布になるようにする
  - すべての出力が0以上で、合計が1になる
- もとの $z_i$ が大きいほど、対応するクラスの確率 $y_i$ が高くなる
$
  "Softmax"(z_i) &= exp(z_i) / (sum_(j=1)^C exp(z_j)) \
$

=== クロスエントロピー損失の仕組み
- まず確率 $p$ のことが起こることには $- log p$ の"驚き"(損失) があると考える
- 正解クラスの確率が高いほど損失が小さくなり、正解クラスの確率が低いほど損失が大きくなる
- さらに、すべてのクラスに対して損失を合計することで、全体の損失を計算する

#note-box(title: "One-hot表現について")[
\
クロスエントロピー損失を計算するためには、正解ラベルをOne-hot表現に変換する必要がある
例えば、正解クラスが3であれば、$t$ は $[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]$ のようなベクトルになる
]
$
  "CrossEntropy"(y, t) &= - sum_(i=1)^C t_i log(y_i)
$

=== Softmaxとクロスエントロピー損失の組み合わせのメリット
- 突然正の値にするためだけに指数関数を使ったり、驚きが対数で定義されると言ったりするのは一見不自然に見えるが、Softmaxとクロスエントロピー損失を組み合わせることで、出力層の誤差が非常にシンプルな形になる
- 具体的には、Softmaxとクロスエントロピー損失を組み合わせると、出力層の誤差が $delta = y - t$ という非常にシンプルな形になる
$
  L(y, t) &= - sum_(i=1)^C t_i log(y_i) \
  (partial L) / (partial y_i) &= - t_i / y_i \
  (partial y_i) / (partial z_i) &= y_i (1 - y_i) \
  delta_i &= (partial L) / (partial z_i) = (partial L) / (partial y_i) dot (partial y_i) / (partial z_i) = y_i - t_i
$

#pagebreak()

#note-box(title: "勾配")[
  \
  高校まではスカラーをスカラーで微分して、スカラーの微分係数を求めていた

  スカラーをベクトルで微分すると、ベクトルの微分係数（勾配）が得られる

  $
    nabla L = (partial L) / (partial Theta) = ((partial L) / (partial theta_1), (partial L) / (partial theta_2), ..., (partial L) / (partial theta_n))^T
  $

  勾配は、関数の値が最も急激に増加する方向を示すベクトルである
]

== 勾配降下法
#slide[
  *なぜ勾配の話が出てくるのか*
  - 線形回帰では解析的に最適なパラメータを求めることができた
  - しかし、モデルが複雑になると解析的に解くことができなくなる
  - そこで、最適なパラメータを数値的に求める方法が必要になる

  $
    W <- W - eta times (partial L) / (partial W) \
  $
][
  #align(center)[
    #image("images/gradient-descent.png", width: 400pt)
  ]
  #align(right)[
    #text(size: 0.8em, fill: luma(70))[
      https://dx-consultant-fast-evolving.com/gradient-descent/ より引用
    ]
  ]
]

== モデルも複雑にする
線形回帰のような単純なモデルでは、複雑なデータをうまく表現できないことがある

#link("https://otera256.github.io/shinkan2026/contents/mlp.html")[*この教材*]で中間層のニューロン数を1にすると、線形な分離しかできないことがわかる\
ニューロン数を増やすと、非線形な分離もできるようになる！！！

$
  hat(vb(h_1)) &= W_1 vb(x) + vb(b_1) \
  vb(h_1) &= sigma(hat(vb(h_1))) \
  hat(vb(h_2)) &= W_2 vb(h_1) + vb(b_2) \
  vb(h_2) &= sigma(hat(vb(h_2))) \
  vb(z) &= W_3 vb(h_2) + vb(b_3) \
  vb(y) &= "Softmax"(vb(z))
$

=== 活性化関数
- 活性化関数は、ニューロンの出力を非線形に変換する関数である

#let sigmoid(x) = 1 / (1 + calc.exp(-x))
#let xs = lq.linspace(-8, 8, num: 100)

#grid(
  columns: (1fr, 1fr),
  align(center + horizon)[
    #lq.diagram(
      height: 200pt, width: 250pt,
      xlabel: $x$, ylabel: $y$,
      lq.plot(xs, xs.map(x => sigmoid(x)), mark: none, color: blue),
    )
  ],
  align(left + horizon)[
    $
      "Sigmoid"(x) = 1 / (1 + exp(-x))
    $
    なぜ非線形な活性化関数が必要なのか？\
    - 活性化関数が線形だと、複数の層を重ねても結局は線形な関数しか表現できないから
    なぜ連続な関数なのか？\
    - 活性化関数が非連続だと、微分が定義できないから（勾配降下法が使えない）
  ]
)

#note-box(title: "勾配の連鎖律")[
  \
  高校でも習った以下のような関係を用いることで、モデルのパラメータに対する損失の勾配を効率的に計算することができる
  $
    y = f(x) space.quad z = g(y) \
    (partial z) / (partial x) = (partial z) / (partial y) dot (partial y) / (partial x)
  $
]
