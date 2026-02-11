#import "@preview/touying:0.6.1": *
#import themes.metropolis: *
#import "@preview/numbly:0.1.0": numbly

// --- 数式用ライブラリ (physica) ---
#import "@preview/physica:0.9.3": *

// --- 定理環境用ライブラリ (theorion) ---
#import "@preview/theorion:0.4.1": *

// --- 図解用ライブラリ ---
#import "@preview/fletcher:0.5.8": diagram, node, edge, shapes
#import "@preview/lilaq:0.4.0" as lq

// --- 設定 ---
#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  footer: self => self.info.institution,
  config-info(
    title: [機械学習勉強会 第3.5回],
    subtitle: [数学的準備 - 定義と記法のすり合わせ],
    author: [furakuta],
    date: datetime.today(),
    institution: [KMC],
  )
)

#set text(lang: "ja", font: "UD Digi Kyokasho NP")
#set heading(numbering: numbly("{1}.", default: "1.1"))

// --- Theorion 環境定義 ---
#show: show-theorion

// スライド用のブロックスタイル定義
// 修正点: titleの二重表示を防ぐため、full-titleのみを表示するように変更
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
  render: slide-block-render(rgb("#e3f2fd"))
)
#show: show-definition

// 定理 (Theorem): オレンジ系
#let (thm-counter, thm-box, theorem, show-theorem) = make-frame(
  "theorem", "Theorem",
  render: slide-block-render(rgb("#fff3e0"))
)
#show: show-theorem

// 重要・注釈 (Note/Alignment): グレー/緑系
#let (note-counter, note-box, note, show-note) = make-frame(
  "note", "Note",
  render: slide-block-render(rgb("#f1f8e9"))
)
#show: show-note


// 数式ブロックのスタイル
#show math.equation.where(block: true): it => block(inset: 10pt, radius: 5pt, fill: rgb("#f0f0f0"), width: 100%, it)

#title-slide()

== はじめに

このスライドは今後の機械学習勉強会で用いる「道具としての数学」について確認し、学部ごとの記法・用語の揺らぎを吸収することを目的としています。

- *目的*: 記号の定義確認、線形代数・微積分・確率統計・情報理論の基礎復習
- *注意*: 証明は省略するので、詳しくは参考書を参照してください。

= 線形代数

== ベクトルと行列

=== 体 (Field)

#definition(title: "環・可換環の定義")[
  - $(R, +)$は可換群（アーベル群）である。すなわち,
    - $forall a, b, c in R$ について、$a + (b + c) = (a + b) + c$（結合法則）
    - $forall a, b in R$ について、$a + b = b + a$（可換法則）
    - 単位元 $exists 0 in R "s.t." forall a in R$ について、$a + 0 = a$（単位元の存在）
    - $forall a in R$ について、逆元 $exists -a in R "s.t." a + (-a) = 0$（逆元の存在）
  - $(R, times)$ は単位元 $1$ を持つモノイドである。すなわち,
    - $forall a, b, c in R$ について、$a times (b times c) = (a times b) times c$（結合法則）
    - 単位元 $exists 1 in R "s.t." forall a in R$ について、$a times 1 = a$（単位元の存在）
  - 分配法則: $forall a, b, c in R$ について、
    - $a times (b + c) = a times b + a times c$
    - $(a + b) times c = a times c + b times c$
  が成り立つとき、$R$ を*環*と呼ぶ。
  さらに、$forall a, b in R$ について $a times b = b times a$ が成り立つとき、$R$ を*可換環*と呼ぶ。
]

#definition(title: "体の定義")[
  空でない集合$K$が体であるとは、
  - $(K, +, times)$が単位元を持つ可換環であること
  - $forall a in K \\ {0}$ に対して、逆元 $exists a^(-1) in K "s.t." a times a^(-1) = 1$ が存在すること
]
- 機械学習では主に*実数体* $RR$ を扱う。
  - 複素数体 $CC$ も信号処理（フーリエ変換など）で使うことがあるが、基本は $RR$。
- スカラー（scalar）: 体の元。単なる数値。

=== ベクトル空間 (Vector Space)

#definition(title: "ベクトル空間")[
  集合$V$が体$K$上のベクトル空間であるとは、
  - $(V, +)$が可換群であること
  - $forall a, b in K$、$forall vb(x), vb(y) in V$ について、
    - $a (vb(x) + vb(y)) = a vb(x) + a vb(y)$
    - $(a + b) vb(x) = a vb(x) + b vb(x)$
    - $(a b) vb(x) = a (b vb(x))$
    - $1 vb(x) = vb(x)$
]

#definition(title: "n次元実ベクトル空間")[
  $ RR^n = { vb(x) = (x_1, x_2, dots, x_n)^top | x_i in RR } $
]

#pagebreak()

#note(title: "すり合わせ: ベクトルの向き")[
  本勉強会では、ベクトル $vb(x)$ は特に断りがない限り*列ベクトル (Column Vector)* とします。
  $ vb(x) = (x_1, x_2, dots, x_n)^top = vec(x_1, x_2, dots.v, x_n) in RR^n $
]

=== 線形写像 (Linear Map)

#definition(title: "線形写像")[

  ベクトル空間からベクトル空間への写像 $f: V -> W$ で、線形性を持つもの。
  $ f(a vb(x) + b vb(y)) = a f(vb(x)) + b f(vb(y)) $
]

- 線形代数の主役。行列と密接に関係する。

=== 基底と次元

#definition(title: "基底と次元")[
  - *基底* (Basis): ベクトル空間$V$の線形独立な生成系。すなわち、$V$の任意の元が基底の線形結合で一意的に表現できる。
  - *次元* (Dimension): 基底の元の個数。有限次元ベクトル空間では、すべての基底は同じ個数の元を持つ。
]

=== 線形写像の基本定理

#theorem[
  - $V, W$ を有限次元ベクトル空間、$f: V -> W$ を線形写像とする。
  - $"dim"V = n$, $"dim"W = m$ とする。
  - $"Ker"f = { vb(x) in V | f(vb(x)) = vb(0) }$ を核空間 (Kernel)、$"Im" f = { vb(y) in W | exists vb(x) in V "s.t." f(vb(x)) = vb(y) }$ を像空間 (Image) とする。
  - このとき、次が成り立つ:
    $ "dim"V = "dim"("ker"f) + "dim"("Im"f) $
]

=== 行列 (Matrix)

#definition(title: "行列")[

  数を長方形に並べたもの。$m times n$ 行列は $m$ 行 $n$ 列を持つ。
  $ A = mat(
    a_(11), a_(12), dots, a_(1n);
    a_(21), a_(22), dots, a_(2n);
    dots.v, dots.v, dots.down, dots.v;
    a_(m 1), a_(m 2), dots, a_(m n)
  )$
]

並べる数のことを*成分* (Element) または*要素* (Entry) と呼ぶ。成分 $a_(i j)$ は $i$ 行 $j$ 列目の数。

$m$行$n$列の実数行列全体の集合は $RR^(m times n)$または $M_(m times n)(RR)$ と表す。

$m$行1列の行列は*列ベクトル*、1行$n$列の行列は*行ベクトル*と呼ぶ。

==== 線形写像の行列表現

- ベクトル $vb(x) in RR^n$ を $vb(y) in RR^m$ に移す線形写像は、$A in RR^(m times n)$ を用いて以下のように書ける。
  $ vb(y) = A vb(x) $
- ニューラルネットワークの「全結合層」は、この線形変換（＋バイアス）そのもの。

==== 行列の演算

- *和・差*: 同じ型の行列同士でのみ定義。要素ごとの和・差。
- *スカラー倍*: 全要素を $k$ 倍。
- *行列積*: $A$ ($l times m$) と $B$ ($m times n$) の積 $C = A B$ ($l times n$)。
  $ c_(i j) = sum_(k=1)^m a_(i k) b_(k j) $
  #note[一般に $A B != B A$（非可換）である。]
- *アダマール積* ($A circle.small B$): 要素ごとの積。機械学習（特にゲート機構）で頻出。
  $ (A circle.small B)_(i j) = a_(i, j) b_(i j) $

==== 行列式 (Determinant)

- 正方行列 $A$ に対して定義されるスカラー値 $det(A)$ または $|A|$。
- 幾何学的意味: 線形変換による*体積の拡大率*。
- $det(A) != 0 <=> A$ は正則（逆行列を持つ）。

==== 行列の階数 (Rank)

#theorem(title: "行列の階数")[
  $m times n$行列$A$に対して、以下の数は等しい
  1. $rank A$
  2. $A$の列ベクトルの中から選びうる線形独立なベクトルの最大数
  3. $A$の行ベクトルの中から選びうる線形独立なベクトルの最大数
  4. $A$の小行列の中で小行列式が0でないものの最大次元
  5. $A$の行ベクトルが生成する部分空間の次元
  6. $A$の階段標準形における単位行列の次数
]

==== 逆行列 (Inverse Matrix)

#definition(title: "逆行列")[
  正方行列 $A in M_(n times n)$ に対して、$det A != 0$のとき$exists A^(-1) in M_(n times n) "s.t." A A^(-1) = A^(-1) A = I$
]

- 連立一次方程式 $A vb(x) = vb(b)$ の解は $vb(x) = A^(-1) vb(b)$。
- 数値計算上は、逆行列を直接求めることは計算コストが高く（$O(n^3)$）不安定なため避ける（LU分解などを使う）。

==== 転置行列 (Transpose)

#definition(title: "実転置行列")[
  行と列を入れ替えた行列 $A^T$。定義は $(A^T)_(i j) = A_(j i)$。
]

- 性質: $(A B)^T = B^T A^T$
- ベクトルの内積: $vb(x)^T vb(y)$ （行ベクトル $times$ 列ベクトル）

#definition(title: "複素エルミート転置行列")[
  複素共役をとってから転置した行列 $A^*$。定義は $(A^*)_(i j) = dash(A_(j i))$。
]

=== 特別な行列

==== 単位行列 (Identity Matrix)

- 対角成分が1、それ以外が0の正方行列 $I$ または $E$。
- 行列積の単位元: $A I = I A = A$。

==== ゼロ行列 (Zero Matrix)

- 全成分が0の行列 $O$。

==== 対角行列 (Diagonal Matrix)

- 対角成分以外が0の行列。
  $ "diag"(d_1, dots, d_n) = mat(d_1, 0, dots; 0, d_2, dots; dots, dots, dots.down) $
- 計算が容易（逆行列は各要素の逆数、積は要素ごとの積）。

==== 実対称行列 (Symmetric Matrix)

- $A^T = A$ を満たす行列。
- 固有値がすべて実数になる（重要）。
- つまり常に直交行列で対角化可能。
- 分散共分散行列、ヘッセ行列などは対称行列。

==== エルミート行列 (Hermitian Matrix)

- $A^* = A$ を満たす複素行列。
- 固有値がすべて実数になる（重要）。
- つまり常に直交行列で対角化可能。（本当に重要）。

==== 歪対称行列 (Skew-Symmetric Matrix)

- $A^T = -A$ を満たす行列。
- 対角成分はすべて0。

==== 歪エルミート行列 (Skew-Hermitian Matrix)

- $A^* = -A$ を満たす複素行列。
- 対角成分はすべて純虚数。

==== 正規行列 (Normal Matrix)
- $A A^* = A^* A$ を満たす複素行列。
- エルミート行列、ユニタリ行列、対称行列、直交行列はすべて正規行列。
- ユニタリ行列によって対角化可能。

==== 直交行列 (Orthogonal Matrix)

#definition(title: "直交行列")[
  $A^T A = A A^T = I$ を満たす実正方行列。
]
- $A^(-1) = A^T$。
- 幾何学的には「回転」や「鏡映」を表す（長さや角度を変えない）。
- 行列の$(i, j)$成分は$i$行目の行ベクトルと$j$行目の行ベクトルの内積であるため、上の条件は「行ベクトル同士が直交し、かつ長さ1である」ことを意味する。このような基底を*正規直交基底*と呼ぶ。

==== ユニタリ行列 (Unitary Matrix)
#definition(title: "ユニタリ行列")[
  $A^* A = A A^* = I$ を満たす複素正方行列。
]
- $A^(-1) = A^*$。
- 直交行列の複素版。量子力学などで重要。

==== 正定値行列 (Positive Definite Matrix)
#definition(title: "正定値行列")[
  任意の非ゼロベクトル $vb(x) in RR^n$ に対して、$ vb(x)^T A vb(x) > 0 $ を満たす実対称行列 $A$。
]

- 分散共分散行列、ヘッセ行列などは正定値行列（または半正定値行列）。

== 内積とノルム

=== 内積 (Inner Product)

#definition(title: "内積空間")[
  ベクトル空間$V$に対して、内積 $iprod: V times V -> K$ が定義されており、以下を満たすとき、$V$ を*内積空間*と呼ぶ。
  1. 正定値性: $forall vb(x) in V \\ {vb(0)}$, $iprod(vb(x), vb(x)) > 0$。また、$iprod(vb(0), vb(0)) = 0$。
  2. 共役対称性: $forall vb(x), vb(y) in V$, $iprod(vb(x), vb(y)) = dash(iprod(vb(y), vb(x)))$。
  3. 線形性: $forall a, b in K$, $forall vb(x), vb(y), vb(z) in V$, $iprod(a vb(x) + b vb(y), vb(z)) = a iprod(vb(x), vb(z)) + b iprod(vb(y), vb(z))$。
]

#definition(title: "標準内積")[
  $RR^n$ における内積 (Dot Product):
  $ iprod(vb(x), vb(y)) = vb(x) dot vb(y) = vb(x)^T vb(y) = sum_(i=1)^n x_i y_i $
]

- 幾何学的意味:
  $ iprod(vb(x), vb(y)) = norm(vb(x)) norm(vb(y)) cos theta $

=== エルミート内積

- 複素ベクトル空間における内積。
- 片方について共役線形。
  $ iprod(vb(x), vb(y)) = vb(x)^* vb(y) = sum_(i=1)^n overline(x_i) y_i $
- 量子力学などで使われるが、今回は標準内積が主。

=== ノルム (Norm)

#definition(title: "ノルム")[
  ベクトルの大きさを測る関数 $norm: V -> RR$ で、以下を満たすもの。
  1. 正定値性: $forall vb(x) in V \\ {vb(0)}$, $norm(vb(x)) > 0$。また、$norm(vb(0)) = 0$。
  2. 齢数倍性: $forall a in K$, $forall vb(x) in V$, $norm(a vb(x)) = |a| norm(vb(x))$。
  3. 三角不等式: $forall vb(x), vb(y) in V$, $norm(vb(x) + vb(y)) <= norm(vb(x)) + norm(vb(y))$。
]

#definition(title: "Lpノルム")[
  ベクトルの「大きさ」を測る尺度。
  $ norm(vb(x))_p = (sum_(i=1)^n |x_i|^p)^(1/p) $
]

=== ノルムの種類

- *$L_2$ ノルム* (ユークリッドノルム):
  $ norm(vb(x))_2 = sqrt(sum x_i^2) $
  - 通常の距離。特に断らない限りこれを指す。
  - Ridge回帰（L2正則化）で使用。
- *$L_1$ ノルム* (マンハッタンノルム):
  $ norm(vb(x))_1 = sum |x_i| $
  - Lasso回帰（スパース推定）で使用。
- *$L_oo$ ノルム* (最大値ノルム):
  $ norm(vb(x))_oo = max_i |x_i| $

$p$に関してLpノルムは広義単調増加であり、$p$が大きくなるほど大きい成分に重みがかかる。

=== コサイン類似度 (Cosine Similarity)

- 2つのベクトルの「向き」の近さを表す。
  $ "similarity" = cos theta = (iprod(vb(x), vb(y))) / (norm(vb(x)) norm(vb(y))) $
- 自然言語処理（単語ベクトルの類似度など）で頻出。
- 値は $[-1, 1]$。1に近いほど似ている。

=== （グラム）シュミットの正規直交化法

- 線形独立なベクトル列から、正規直交基底（長さ1で互いに直交）を作るアルゴリズム。
- QR分解の基礎。
- 手順: 内積空間$V$の独立なベクトル$vb(a)_1, vb(a)_2, dots, vb(a)_n$について
  1. $vb(v)'_1 = vb(a)_1$, $vb(v)_1 = vb(v)'_1 / norm(vb(v)'_1)$
  2. $"for" k = 2$ to $n$:
     - $vb(v)'_k = vb(a)_k - sum_(j=1)^(k-1) iprod(vb(a)_k, vb(v)_j) vb(v)_j$
     - $vb(v)_k = vb(v)'_k / norm(vb(v)'_k)$
  

== 固有値と固有ベクトル

=== 固有値と固有ベクトルの定義

#definition(title: "固有値・固有ベクトル")[
  正方行列 $A$ に対して、以下を満たす非ゼロベクトル $vb(x)$ とスカラー $lambda$。
  $ A vb(x) = lambda vb(x) $
  $vb(x)$ を*固有ベクトル*、$lambda$ を*固有値*と呼ぶ。
]
- 線形変換 $A$ によって方向が変わらず、定数倍されるだけのベクトル。RNNの安定性解析などで重要。

=== 固有値問題の行列表現

固有値問題は以下のように書ける。
$ (A - lambda I) vb(x) = vb(0) $
非ゼロ解 $vb(x)$ が存在するための必要十分条件は、
$ det(A - lambda I) = 0 $
$lambda$を$t$に置き換えると、これは$t$の$n$次多項式となる。この多項式を*特性多項式*と呼ぶ。
- $n$次正方行列は高々$n$個の固有値を持つ（重複度を考慮）。

=== 固有値の性質

#theorem(title: "トレースと行列式")[
  - 固有値の和 ＝ トレース（対角成分の和）: $ sum lambda_i = tr(A) = sum_(i = 1)^n A_(i i) $
  - 固有値の積 ＝ 行列式: $ product lambda_i = det(A) $
]

=== 対角化

- $n$ 次正方行列 $A$ が $n$ 個の線形独立な固有ベクトルを持つとき、
  $ P^(-1) A P = Lambda = "diag"(lambda_1, dots, lambda_n) $
  と対角化できる。
- $P$ は固有ベクトルを並べた行列。
- 行列の $k$ 乗計算などが容易になる。

== 特異値分解 (SVD)

=== なぜSVDが必要か？

- 固有値分解は*正方行列* ($n times n$) にしか定義できない。
- しかし、機械学習で扱うデータ行列（$N$個のデータ, $D$次元）は通常*長方形* ($N != D$)。
- *任意の行列*に対して定義できる分解が欲しい $->$ *特異値分解 (SVD)*。

=== 特異値分解の定義

#definition(title: "特異値分解 (Singular Value Decomposition)")[
  任意の行列 $A in RR^(m times n)$ は以下のように分解できる。
  $ A = U Sigma V^T $
]
- $U in RR^(m times m)$: *左特異ベクトル*からなる直交行列 ($U^T U = I$)
- 左特異ベクトル $vb(u)_i$ は $A A^T$ の固有ベクトル
- $Sigma in RR^(m times n)$: *特異値* $sigma_i$ を対角に並べた行列 (非対角は0)
- $V in RR^(n times n)$: *右特異ベクトル*からなる直交行列 ($V^T V = I$)
- 右特異ベクトル $vb(v)_i$ は $A^T A$ の固有ベクトル

=== 幾何学的意味：回転・拡大縮小・回転

行列 $A$ による変換 $A vb(x)$ は、3つのステップに分解できる。

1. *$V^T$ (または $V^(-1)$) による回転*:
   入力空間の基底を「特異ベクトルの方向」に合わせる。
2. *$Sigma$ による拡大縮小*:
   軸ごとに特異値 $sigma_i$ 倍だけ引き伸ばす（次元が変わることもある）。
3. *$U$ による回転*:
   出力空間の向きに合わせる。

#note[
  「どんな複雑な行列も、適切な座標系を選べば、単なる拡大縮小として表現できる」という強力な定理。
]

=== 特異値 $sigma_i$ の意味

- $Sigma = "diag"(sigma_1, sigma_2, dots)$。順序には任意性があるため通常 $sigma_1 >= sigma_2 >= dots >= 0$ と並べる。
- 特異値 $sigma_i$ は、変換後の*広がり（分散）の大きさ*、あるいはデータの*重要度*を表す。
- $sigma_i$ が大きい成分ほど、元の行列 $A$ の情報を多く持っている。

=== 低ランク近似（情報圧縮）

行列 $A$ を特異値の大きい順に $k$ 個だけ使って再構成すると、元の行列の最も良い近似になる（エッカート・ヤングの定理）。

$ A approx A_k = sum_(i=1)^k sigma_i vb(u)_i vb(v)_i^T $

- *画像圧縮*: 画像を行列とみなし、上位の特異値だけ残すと、容量を減らしつつ見た目を保てる。
- *ノイズ除去*: 小さな特異値はノイズとみなして切り捨てる。
- *推薦システム*: ユーザー $times$ 商品行列を分解し、潜在的な特徴（ランク $k$）を抽出する。

=== ムーア・ペンローズ擬似逆行列

正方行列でない $A$ に「逆行列っぽいもの」を定義したい。

$ A^+ = V Sigma^+ U^T $

- $Sigma^+$: $Sigma$ の非ゼロ成分 $sigma_i$ を $1/sigma_i$ にして転置した行列。
- 連立方程式 $A vb(x) = vb(b)$ の最小二乗解 $ min_(vb(x)) norm(A vb(x) - vb(b))^2 $ は $ vb(x) = A^+ vb(b) $ で求まる。


= 微分積分

== 1変数関数の微分

=== 微分係数と導関数

#definition(title: "微分係数")[
  接線の傾き。
  $ dv(f, x) = f'(x) = lim_(h -> 0) (f(x+h) - f(x)) / h $
]
- 機械学習では「パラメータを少し動かしたときの損失関数の変化量」として重要。

=== 基本的な微分の公式

- $(x^n)' = n x^(n-1)$
- $(e^x)' = e^x$
- $(log x)' = 1/x$
- $(sin x)' = cos x$

=== 合成関数の微分 (Chain Rule)

#theorem(title: "Chain Rule")[
  *最重要項目*（Back Propagationの原理）。
  $ y = f(u), quad u = g(x) => dv(y, x) = dv(y, u) dv(u, x) $
]
- 外側の微分 $times$ 内側の微分。

== 多変数関数の微分

=== 偏微分 (Partial Derivative)

#definition(title: "偏微分")[
  多変数関数 $f(x_1, dots, x_n)$ において、一つの変数 $x_i$ 以外を定数とみなして微分する。
  $ pdv(f, x_i) = lim_(h -> 0) (f(dots, x_i+h, dots) - f(dots, x_i, dots)) / h $
]

=== 勾配 (Gradient)

#definition(title: "勾配")[
  偏微分係数をベクトルとして並べたもの。
  $ nabla f = grad f = vec(pdv(f, x_1), pdv(f, x_2), dots.v, pdv(f, x_n)) $
]
#note(title: "すり合わせ: 勾配の形状")[
  本勉強会では勾配を*列ベクトル*として扱います（分母レイアウトの転置相当）。
]
- 関数が最も急激に増加する方向を向く。
- 勾配降下法: $vb(x) arrow.l vb(x) - eta nabla f$

=== ヤコビ行列 (Jacobian Matrix)

#definition(title: "ヤコビ行列")[
  ベクトル値関数 $vb(f): RR^n -> RR^m$ の導関数 ($m times n$ 行列)。
  $ J = pdv(vb(f), vb(x)) = mat(
    pdv(f_1, x_1), dots, pdv(f_1, x_n);
    dots.v, , dots.v;
    pdv(f_m, x_1), dots, pdv(f_m, x_n)
  ) $
]
- 変数変換や、層ごとの微分の連鎖で使用。

=== ヘッセ行列 (Hessian Matrix)

#definition(title: "ヘッセ行列")[
  スカラー値関数の二階偏微分係数を並べた行列 ($n times n$ 対称行列)。
  $ H = nabla^2 f = mat(
    pdv(f, x_1, x_1), dots, pdv(f, x_1, x_n);
    dots.v, , dots.v;
    pdv(f, x_n, x_1), dots, pdv(f, x_n, x_n)
  ) $
]
- 関数の「曲率」を表す。

=== 凸性の判定

#theorem(title: "凸関数の判定")[
  $f$ が凸関数 $<=>$ 任意の $vb(x)$ でヘッセ行列 $H$ が*半正定値* ($vb(v)^T H vb(v) >= 0$)。
]
- 凸関数であれば、勾配降下法で局所解に陥らず大域的最適解に到達できる（ロジスティック回帰など）。

=== テイラー展開 (Taylor Expansion)

#theorem(title: "多変数関数のテイラー展開")[
  $ f(vb(x) + vb(h)) approx f(vb(x)) + nabla f(vb(x))^T vb(h) + 1/2 vb(h)^T H vb(h) $
]
- 1次近似 $->$ 勾配降下法の原理
- 2次近似 $->$ ニュートン法の原理

= 確率・統計

== 確率論の基礎

=== 条件付き確率とベイズの定理

- 条件付き確率:
  $ P(A|B) = (P(A inter B)) / (P(B)) $
#theorem(title: "ベイズの定理")[
  $ P(B|A) = (P(A|B) P(B)) / (P(A)) $
]
  - $P(B)$: 事前確率
  - $P(B|A)$: 事後確率
  - $P(A|B)$: 尤度
- 生成モデル（VAE, Diffusion）の理論的支柱。

== 確率変数と確率分布

=== 確率変数 (Random Variable)

- 値が確率的に決まる変数 $X$。
- *離散型*: 確率質量関数 (PMF) $P(X=x)$
- *連続型*: 確率密度関数 (PDF) $p(x)$
  - 特定の値をとる確率は0。積分して確率になる。
  - $integral_(-oo)^(oo) p(x) d x = 1$

=== 期待値と分散

#definition(title: "期待値・分散")[
  - *期待値* (Expectation): $ E[X] = integral x p(x) d x $
  - *分散* (Variance): $ V[X] = E[(X - E[X])^2] = E[X^2] - (E[X])^2 $
]
- 線形性: $E[a X + b Y] = a E[X] + b E[Y]$
- *共分散* (Covariance): $ "Cov"(X, Y) = E[(X-E[X])(Y-E[Y])] $

== 推定

=== 尤度 (Likelihood)

#definition(title: "尤度")[
  観測データ $D$ が得られたときの、パラメータ $theta$ のもっともらしさ。
  $ L(theta) = P(D | theta) $
]
- 確率とは変数の扱いが逆（データ固定、パラメータを変数と見る）。

=== 最尤推定法 (Maximum Likelihood Estimation)

- 尤度 $L(theta)$ を最大にする $theta$ を求める手法。
- 計算を楽にするため、*対数尤度* $log L(theta)$ を最大化することが多い（積を和に直せるため）。
- 機械学習の多くの損失関数（クロスエントロピーなど）は、負の対数尤度の最小化と等価。

= 情報理論

== 情報量

=== 自己情報量

#definition(title: "自己情報量")[
  事象 $x$ が起きたときの「驚き」の度合い。
  $ I(x) = - log P(x) $
]
- 確率が低い事象ほど情報量は大きい。
- 底が2ならビット(bit)、$e$ならナット(nat)。

=== シャノンエントロピー (Entropy)

#definition(title: "シャノンエントロピー")[
  情報量の期待値（平均情報量）。不確実性の尺度。
  $ H(P) = E_(x ~ P) [I(x)] = - sum_x P(x) log P(x) $
]
- 分布が一様であるほど値は大きく、偏りがあるほど小さい。

=== KL Divergence (Kullback-Leibler)

#definition(title: "KLダイバージェンス")[
  2つの確率分布 $P$ と $Q$ の「距離」のようなもの（非対称）。
  $ D_"KL"(P || Q) = sum_x P(x) log (P(x)) / (Q(x)) $
]
- VAEの損失関数などで、真の分布と近似分布の差を測るために使用。
- 常に $D_"KL" >= 0$ (Gibbsの不等式)。

=== 交差エントロピー (Cross Entropy)

#definition(title: "交差エントロピー")[
  $ H(P, Q) = - sum_x P(x) log Q(x) $
]
- 関係性:
  $ H(P, Q) = H(P) + D_"KL"(P || Q) $
- $P$（正解分布）を固定して $H(P, Q)$ を最小化することは、$D_"KL"$ を最小化すること（$P$ に $Q$ を近づけること）と同じ。

= まとめ

== まとめ

- *線形代数*: 行列積のサイズ確認、内積、固有値分解・SVDが重要。
- *微積分*: 勾配（Gradient）と連鎖律（Chain Rule）がDeep Learningのすべて。
- *確率統計*: 最尤推定とベイズの定理がモデル設計の基礎。
- *情報理論*: 損失関数（Loss）の意味理解に不可欠。

次回の第4回からは、これらの数学を「PyTorchでの実装」と紐付けて、実際のDeep Learningに入っていきます。