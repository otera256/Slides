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
    title: [機械学習勉強会 第4回],
    subtitle: [深層学習の基礎と自動微分],
    author: [furakuta],
    date: datetime.today(),
    institution: [KMC],
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

= 導入

== 今日の目標

*Multi-Layer Perceptron (MLP)* を構成し、正しく学習させるための要素技術を理解する。

1.  *自動微分*: どうやって複雑な数式の勾配を計算機で求めるか？
2.  *初期値*: 学習が「始まる」ための条件は？
3.  *Optimizer*: 勾配を使ってどう賢くパラメータを更新するか？
4.  *正規化*: 学習を安定させ、過学習を防ぐには？

== 前回の復習

- *Universal Approximation Theorem*:
  - 線形層と活性化関数の組み合わせで、任意の非線形関数を近似できる。
- *深層化の利点*:
  - 層を深くすることで、同じ表現力をより少ないパラメータで実現できる。
- *誤差逆伝播法 (Backpropagation)*:
  - 出力誤差を出発点として、連鎖律 (Chain Rule) を適用し、各パラメータの勾配を効率的に計算する。

= 計算グラフと自動微分

== 計算グラフ (Computational Graph)

数式を*ノード*（演算）と*エッジ*（データ）のグラフ構造で表現したもの。

例: $L = (x + y) times z$

#align(center)[
  #diagram(
    node-stroke: 1pt,
    spacing: (60pt, 40pt),
    node((0,0), $x$, name: <x>),
    node((0,1), $y$, name: <y>),
    node((1,0.5), $+$, name: <add>, shape: circle),
    node((0,2), $z$, name: <z>),
    node((2,1.25), $times$, name: <mul>, shape: circle),
    node((3,1.25), $L$, name: <L>),

    edge(<x>, <add>, "->"),
    edge(<y>, <add>, "->"),
    edge(<add>, <mul>, "->", label: $u$),
    edge(<z>, <mul>, "->"),
    edge(<mul>, <L>, "->"),
  )
]

- *順伝播 (Forward)*: 入力から出力へ値を計算。
- *逆伝播 (Backward)*: 出力から入力へ勾配を計算。

== 自動微分 (Automatic Differentiation)

微分の計算を「数値微分」でも「数式処理（シンボリック）」でもなく、*計算グラフ上の連鎖律の適用*として行う手法。

#theorem(title: "連鎖律 (Chain Rule)")[
  ある変数 $x$ が $u$ を経由して $L$ に影響を与えるとき ($L = f(u), u = g(x)$):
  $ pdv(L, x) = pdv(L, u) dot pdv(u, x) $
]

計算グラフの各ノードは「自分の入力に対する出力の局所的な微分」を知っていれば良い。
全体の微分は、それらを後ろから掛け合わせるだけで求まる。

== 逆伝播の例

先ほどの $L = u times z$ ($u = x + y$) における $pdv(L, x)$ を求める。

#grid(
  columns: (1fr, 1.5fr),
  align(center + horizon)[
    #diagram(
      node-stroke: 1pt,
      spacing: (40pt, 30pt),
      node((0,0), $x$, name: <x>),
      node((1,0), $+$, name: <add>, shape: circle),
      node((2,0), $times$, name: <mul>, shape: circle),
      node((3,0), $L$, name: <L>),

      edge(<add>, <x>, "->", stroke: red, label: $pdv(u, x)$, label-side: right, bend: 30deg),
      edge(<mul>, <add>, "->", stroke: red, label: $pdv(L, u)$, label-side: right, bend: 30deg),
    )
  ],
  [
    1. *出力層*: $pdv(L, L) = 1$
    2. *$times$ ノード*:
       - $L = u z$ なので $pdv(L, u) = z$
       - 勾配 $1 times z$ を $u$ 側に流す
    3. *$+$ ノード*:
       - $u = x + y$ なので $pdv(u, x) = 1$
       - 勾配 $z times 1$ を $x$ 側に流す
    
    $therefore pdv(L, x) = z$
  ]
)

== 実装イメージ (Python)

各演算をクラスとして定義し、`forward` と `backward` を実装する。

#text(size: 16pt)[
```python
class AddLayer:
    def forward(self, x, y):
        self.shape = x.shape  # 形状保存（ブロードキャスト対応用など）
        return x + y

    def backward(self, dout):
        # L = x + y => dL/dx = 1 * dL/du
        dx = dout * 1
        dy = dout * 1
        return dx, dy

class MulLayer:
    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout):
        # L = xy => dL/dx = y * dL/du
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy
```
]

== 自動微分の特徴

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    *利点 (Pros)*
    - *正確性*: 数値微分のような近似誤差がない。
    - *効率性*: 計算グラフを一度構築すれば、すべてのパラメータの勾配を1回のBackward passで計算できる。
    - *柔軟性*: `if` 文やループを含む複雑な計算も微分可能（Define-by-Run）。
  ],
  [
    *欠点・注意点 (Cons)*
    - *メモリ消費*: 逆伝播のために、順伝播時の中間変数をメモリに保持しておく必要がある。
    - *実装*: 循環参照によるメモリリークに注意が必要（PythonのGCで概ね解決するが）。
  ]
)

- *Define-by-Run (動的グラフ)*: PyTorch。データが流れるたびにグラフを作る。柔軟。
- *Define-and-Run (静的グラフ)*: TensorFlow(v1)。先にグラフを定義してからデータを流す。最適化しやすい。

= パラメータの初期値

== なぜ初期値が重要か？

前回の実装で、重み $W$ を「全て0」や「極端に大きな値」にすると学習が進まなかった。

- *全て0*: 全てのニューロンが同じ計算をし、同じ勾配を持つため、実質1つのニューロンしか無いのと同じになる（対称性の破壊が必要）。
- *大きすぎる*: 活性化関数（Sigmoid/Tanh）の飽和領域に入り、勾配が0になる（勾配消失）。
- *小さすぎる*: 信号が層を通るごとに減衰し、最後には0になる（情報の消失）。

*目標*: 各層のアクティベーション（出力値）と勾配の*分散*が、層を経ても変化しないようにしたい。

== アクティベーションの分散 (順伝播)

ある層の計算 $y = sum_(i=1)^n w_i x_i$ を考える（バイアスは省略）。
$x_i, w_i$ が互いに独立で、平均0と仮定すると：

$ V[y] &= V[sum_(i=1)^n w_i x_i] = sum_(i=1)^n V[w_i x_i] \
       &= sum_(i=1)^n E[w_i^2 x_i^2] - (E[w_i x_i])^2 \
       &= sum_(i=1)^n (E[w_i^2] E[x_i^2]) quad (because E[w] = E[x] = 0) \
       &= n dot V[w] dot V[x] $

*条件*: $V[y] = V[x]$ （入力と出力で分散が変わらない）ためには
$ n dot V[w] = 1 arrow.r.double V[w] = 1/n $

$n = n_"in"$として、前層のノード数と呼ぶと、重みの分散は $ V[w] = 1/n_"in"$ であるべき。

== 勾配の分散 (逆伝播)

逆伝播でも同様に、ある層の勾配 $pdv(L, x_i) = sum_(j=1)^m pdv(L, y_j) pdv(y_j, x_i)$ を考える。
$pdv(y_j, x_i) = w_(j i)$ なので、同様に計算すると：

$ V[pdv(L, x_i)] &= V[sum_(j=1)^m pdv(L, y_j) w_(j i)] \
                 &= m dot V[w] dot V[pdv(L, y_j)] $

*条件*: $V[pdv(L, x_i)] = V[pdv(L, y_j)]$ のためには
$ m dot V[w] = 1 arrow.r.double V[w] = 1/m $

$m = n_"out"$として、後層のノード数と呼ぶと、重みの分散は $ V[w] = 1/n_"out"$ であるべき。

一般に、順伝播と逆伝播の両方を満たすことは不可能。
よって、どちらを優先するかで初期化手法が分かれる。妥協案として$n_"in"$と$n_"out"$の調和平均を取って
$ 1/n = 2/(n_"in" + n_"out") , V[n] ~ 1/n $
とする方法もある。

== 初期化手法 (1/2): Xavier Initialization

前ページの条件 $V[w] = 1/n$ に従う初期化。
*Sigmoid* や *Tanh* のような「原点付近で線形とみなせる」活性化関数の場合に有効。

#definition(title: "Xavier (Glorot) Initialization")[
  前層のノード数を $n$ とするとき、
  - *一様分布*: $W ~ U(-sqrt(3/n), sqrt(3/n))$
  - *ガウス分布*: $W ~ N(0, 1/n)$
]

※ 一様分布 $U(-a, a)$ の分散は $a^2/3$ なので、$a = sqrt(3/n)$ とすれば分散 $1/n$ になる。

== 初期化手法 (2/2): He Initialization

*ReLU ($max(0, x)$)* は負の入力を0にするため、出力の分散が約半分になる。
よって、重みの分散を2倍にして補正する必要がある。
$ n dot V[w] = 2 arrow.r.double V[w] = 2/n $

#definition(title: "He (Kaiming) Initialization")[
  ReLUを用いる場合の標準。
  - *一様分布*: $W ~ U(-sqrt(6/n), sqrt(6/n))$
  - *ガウス分布*: $W ~ N(0, 2/n)$
]

最近のモデル (ResNet, Transformer等) はReLU系が多いため、基本はこれ。

= Optimizer (最適化手法)

== SGDの問題点

単純な *SGD (Stochastic Gradient Descent)*:
$ W arrow.l W - eta nabla L(W) $

#grid(
  columns: (1fr, 1.2fr),
  [
    - *問題点1*: *異方性 (Anisotropy)*。ある方向には急峻だが、別の方向には緩やかな場合（細長い谷）、ジグザグに進んでしまい収束が遅い。
    - *問題点2*: *鞍点 (Saddle Point)*。勾配が平坦な場所で止まりやすい。
    - *問題点3*: 適切な学習率 $eta$ を決めるのが難しい。
  ],
  align(center + horizon)[
    #diagram(
      node-stroke: 1pt,
      node((0,0), "Start"),
      node((2,2), "Goal"),
      edge((0,0), (1,0.2), "->"),
      edge((1,0.2), (0.5, 0.8), "->"),
      edge((0.5,0.8), (1.5, 1.0), "->"),
      edge((1.5,1.0), (1.0, 1.6), "->"),
      edge((1.0,1.6), (2,2), "->"),
    )
    (ジグザグするイメージ)
  ]
)

== Momentum SGD

物理的な「慣性」を導入する。
ボールが谷底に転がるように、勾配をそのまま更新する「速度」としてではなく「加速度」として扱う。まず離散化された運動方程式は以下のようにかける。

$
  v(t + Delta t) &= v(t) + a Delta t \
  x(t + Delta t) &= x(t) + v(t + Delta t) Delta t
$

この式に勾配降下法の考え方を組み合わせると、

$ v_t &= alpha v_(t-1) - eta nabla L(W_t) \
W_(t+1) &= W_t + v_t $

- $alpha$: モーメンタム係数（通常 0.9）。摩擦力のように速度が大きくなりすぎるのを防ぐ。
- ジグザグを抑え、加速して最適解へ向かう。
- 局所解や鞍点を勢いで突破できる可能性がある。

== AdaGrad & RMSProp

学習率 $eta$ をパラメータごとに自動調整したい。

*AdaGrad*:
これまでの勾配の二乗和 $h$ を保持し、学習率を割る。
$ h arrow.l h + (nabla L)^2, quad W arrow.l W - eta / (sqrt(h) + epsilon) nabla L $
- よく動くパラメータ（$h$大）は学習率を下げ、あまり動かないパラメータは上げる。
- *欠点*: $h$ が単調増加し続けるため、学習が進むと更新が止まってしまう。

*RMSProp*:
$h$ の計算に指数移動平均を用いる（過去の情報を徐々に忘れる）。
$ h arrow.l rho h + (1-rho)(nabla L)^2 $
- AdaGradの欠点を解消。Deep Learningでよく使われる。

== Adam (Adaptive Moment Estimation)

Momentum と RMSProp の「いいとこ取り」。現在最も標準的な手法。

#definition(title: "Adam")[
  1. *Momentum項*（勾配の平均）: $m_t = beta_1 m_(t-1) + (1-beta_1) nabla L$
  2. *RMSProp項*（勾配の分散）: $v_t = beta_2 v_(t-1) + (1-beta_2) (nabla L)^2$
  3. バイアス補正を行い、更新:
     $ W arrow.l W - eta dot hat(m)_t / (sqrt(hat(v)_t) + epsilon) $
]

- ハイパーパラメータ: $eta=0.001, beta_1=0.9, beta_2=0.999$ が一般的。
- 迷ったらとりあえず *Adam* を使えば間違いはない。

== ハイパーパラメータ

ここまでのOptimizerで学習率$eta$を始めとして、自動的にモデルのパラメータを調整する役割を持つ機械学習のアルゴリズムでも、いくつかの「手動で設定する値（ハイパーパラメータ）」が存在する。

- *学習率 (Learning Rate)*: $eta$。大きすぎると発散、小さすぎると収束が遅い。
- *バッチサイズ (Batch Size)*: 1回の更新に使うデータ数。大きいほど安定するがメモリを消費。
- *エポック数 (Epochs)*: 全データを何回学習するか。過学習に注意。

このハイパーパラメータを調整する方法には、
- *グリッドサーチ (Grid Search)*: 候補をいくつか用意し、全組み合わせを試す。
- *ランダムサーチ (Random Search)*: 候補をランダムに選んで試す。
- *ベイズ最適化 (Bayesian Optimization)*: 過去の結果を元に次の候補を賢く選ぶ。
などがある。

= 正規化 (Normalization)

== 過学習 (Overfitting)

モデルの表現力が高すぎると、訓練データのノイズまで学習してしまい、未知のデータに対する性能（汎化性能）が下がる現象。

これを防ぐためのテクニックを *正則化 (Regularization)* と呼ぶ。

== Dropout

学習時にランダムにニューロンを無効化（出力を0）する。

#grid(
  columns: (1fr, 1fr),
  align(center)[
    通常時
    #diagram(
      spacing: 20pt,
      node((0,0), shape: circle, fill: gray), node((1,0), shape: circle, fill: gray),
      node((0,1), shape: circle, fill: gray), node((1,1), shape: circle, fill: gray),
      edge((0,0),(0,1)), edge((0,0),(1,1)),
      edge((1,0),(0,1)), edge((1,0),(1,1)),
    )
  ],
  align(center)[
    Dropout時
    #diagram(
      spacing: 20pt,
      node((0,0), shape: circle, fill: gray), node((1,0), shape: circle, stroke: gray, label: $times$),
      node((0,1), shape: circle, stroke: gray, label: $times$), node((1,1), shape: circle, fill: gray),
      edge((0,0),(1,1)),
    )
  ]
)

- *アンサンブル学習*の効果：毎回異なるネットワーク構造で学習しているのと等価。
- 推論時（Test時）はDropoutせず、出力を確率 $1-p$ 倍してスケーリングする（または学習時に $1/(1-p)$ 倍しておく *Inverted Dropout* が主流）。

== Batch Normalization

「各層への入力分布が学習ごとにコロコロ変わる（内部共変量シフト）」のを防ぐため、層の入力を強制的に正規化する。

#definition(title: "Batch Norm")[
  ミニバッチ $B = {x_1, dots, x_m}$ ごとに平均 $mu_B$ と分散 $sigma_B^2$ を計算し、
  $ hat(x)_i = (x_i - mu_B) / sqrt(sigma_B^2 + epsilon) $
  さらに、学習可能なパラメータ $gamma, beta$ でスケール・シフトする。
  $ y_i = gamma hat(x)_i + beta $
]

- 学習係数を大きくできる。初期値依存性が減る。
- *注意*: 推論時は、学習全体での移動平均・分散を使用する。

== その他の正規化

- *Layer Normalization*:
  - バッチ方向ではなく、*1つのサンプルの特徴量方向*（層内の全ニューロン）で正規化。
  - バッチサイズに依存しない。RNNやTransformerで標準的に使用。

- *Weight Decay (L2正則化)*:
  - 損失関数に重みの大きさのペナルティを加える。
  - $L_"total" = L_"data" + lambda/2 norm(W)^2$
  - 勾配更新時に $W arrow.l W - eta (nabla L + lambda W) = (1 - eta lambda) W - eta nabla L$ となり、重みを毎回少しずつ減衰させる効果がある。

= 機械学習フレームワーク

== フレームワークの構造 (PyTorchの例)

実装時は以下のコンポーネントを組み合わせて記述する。

1.  *Dataset / DataLoader*: データの読み込み、前処理、ミニバッチ化。
2.  *Model (nn.Module)*: 層（`nn.Linear`, `nn.ReLU` 等）の積み重ね。`forward` を定義。
3.  *Loss Function*: 誤差関数 (`nn.CrossEntropyLoss` 等)。
4.  *Optimizer*: 更新規則 (`optim.Adam` 等)。

#note[
  フレームワークは「計算グラフの構築」と「自動微分」を裏側でやってくれている。
  我々は順伝播のロジックを書くだけで良い。
]

== 実装の流れ (疑似コード)

#text(size: 14pt)[
```python
# 1. 定義
model = MLP()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 2. 学習ループ
for inputs, targets in dataloader:
    # 勾配リセット (実装によっては必要)
    optimizer.zero_grad()
    
    # 順伝播
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # 逆伝播 (ここで計算グラフを遡る)
    loss.backward()
    
    # パラメータ更新
    optimizer.step()
```
]