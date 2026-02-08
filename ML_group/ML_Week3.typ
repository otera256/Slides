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
    title: [機械学習勉強会 第3回],
    subtitle: [万能の近似能力と確率的最適化],
    author: [ML Circle],
    date: datetime.today(),
    institution: [KMC],
  )
)

#set text(lang: "ja", font: "UD Digi Kyokasho NP")
#set heading(numbering: numbly("{1}.", default: "1.1"))

// 数式ブロックのスタイル
#show math.equation.where(block: true): it => block(inset: 10pt, radius: 5pt, fill: rgb("#f0f0f0"), width: 100%, it)

#title-slide()

= Universal Approximation Theorem\ (普遍近似定理)

== 定理の主張

*「隠れ層が1層でもあれば、任意の連続関数を任意の精度で近似できる」*

正確には：
$sigma(dot)$ を有界で単調増加な連続関数（シグモイドなど）とする。
任意の連続関数 $f: [0, 1]^n arrow.r RR$ と任意の誤差 $epsilon > 0$ に対して、ある整数 $N$ とパラメータ $v_i, w_i, b_i$ が存在し、以下が成り立つ：

$ F(x) = sum_(i=1)^N v_i sigma(w_i^T x + b_i) $
$ |F(x) - f(x)| < epsilon quad (forall x in [0, 1]^n) $

== 証明のアイディア (1/3): ステップ関数の構成

シグモイド関数 $sigma(w x + b)$ において、重み $w$ を非常に大きくすると、ステップ関数に近づく。

#let sigmoid(x) = 1 / (1 + calc.exp(-x))
#let xs = lq.linspace(-10, 10, num: 100)

#grid(
  columns: (1.5fr, 1fr),
  align(center + horizon)[
    #lq.diagram(
      height: 200pt, width: 300pt,
      xlabel: $x$, ylabel: $y$,
      lq.plot(xs, xs.map(x => sigmoid(x)), mark: none, color: blue, label: $w=1$),
      lq.plot(xs, xs.map(x => sigmoid(5*x)), mark: none, color: red, label: $w=5$),
      legend: (position: bottom + right),
    )
  ],
  align(left + horizon)[
    $ w arrow.r infinity, b = -w c\ arrow.r sigma(w(x - c)) approx cases(1 quad (x > c), 0 quad (x < c)) $
    
    重み $w$ を大きくすると、関数は急峻になり、ステップ関数（階段）のように振る舞う。
  ]
)

== 証明のアイディア (2/3): バンプ関数の構成

2つのシグモイド関数の差をとることで、「ある区間だけ $1$、それ以外 $0$」となる*矩形関数（バンプ）*を作れる。

#let bump(x) = sigmoid(10*(x - (-2))) - sigmoid(10*(x - 2))

#grid(
  columns: (1fr, 1.2fr),
  align(left + horizon)[
    $ "Bump"(x) = sigma(w(x - a)) - sigma(w(x - b)) $
    (ただし $a < b$)
    
    右図のように、2つのシグモイドの差分を取ると、局所的に値を持つ関数が作れる。
  ],
  align(center + horizon)[
    #lq.diagram(
      legend: (position: top + left, dy: -2em),
      width: 300pt, height: 200pt,
      xlabel: $x$, ylabel: $y$,
      lq.plot(xs, xs.map(x => bump(x)), mark: none, color: green, label: "Bump Function"),
    )
  ]
)

== 証明のアイディア (3/3): 関数の近似

任意の連続関数 $f(x)$ は、幅の狭い矩形関数の線形結合（リーマン積分の短冊のようなもの）で近似できる。

#align(center)[
  $ f(x) approx sum k_i dot "Bump"_i (x) $
]

// MLPの構造図
#align(center)[
  #diagram(
    node-stroke: 1pt,
    node-shape: circle,
    spacing: (30pt, 30pt),
    // Input layer
    node((0, 0), $x_1$),
    node((0, 1), $x_2$),
    node((0, 2), $x_3$),
    // Hidden layer
    node((2, -0.5), $h_1$),
    node((2, 0.5), $h_2$),
    node((2, 1.5), $dots$, stroke: none),
    node((2, 2.5), $h_N$),
    // Output layer
    node((4, 1), $y$),
    
    // Edges
    edge((0,0), (2,-0.5), "->"), edge((0,0), (2,0.5), "->"), edge((0,0), (2,2.5), "->"),
    edge((0,1), (2,-0.5), "->"), edge((0,1), (2,0.5), "->"), edge((0,1), (2,2.5), "->"),
    edge((0,2), (2,-0.5), "->"), edge((0,2), (2,0.5), "->"), edge((0,2), (2,2.5), "->"),
    
    edge((2,-0.5), (4,1), "->"), edge((2,0.5), (4,1), "->"), edge((2,2.5), (4,1), "->"),
    
    // Labels
    node((0, 3), "入力層", stroke: none),
    node((2, 3), "隠れ層 (N個)", stroke: none),
    node((4, 3), "出力層", stroke: none),
  )
]

隠れ層のニューロン数 $N$ を増やせば、この「短冊」をいくらでも細かくできるため、誤差 $epsilon$ を限りなく小さくできる。

= 多クラス分類への拡張

== Softmax関数 (1/2)

2値分類ではシグモイド関数を用いたが、$K$ クラス分類（例：MNISTの0～9）では *Softmax関数* を出力層に用いる。

$ y_k = "softmax"(z)_k = e^(z_k) / (sum_(j=1)^K e^(z_j)) quad (k=1, dots, K) $

*性質:*
1. *$0 < y_k < 1$*: 各出力は正の値。
2. *$sum_(k=1)^K y_k = 1$*: 全クラスの合計は1（確率分布として解釈可能）。
3. 入力 $z_k$ の大小関係を保ったまま確率に変換する（単調増加）。

== Softmax関数 (2/2): シグモイドとの関係

$K=2$ のとき、Softmax関数はシグモイド関数と一致する。

証明：$z_1, z_2$ について、クラス1の確率は
$ y_1 &= e^(z_1) / (e^(z_1) + e^(z_2)) \
      &= 1 / (1 + e^(z_2) / e^(z_1)) \
      &= 1 / (1 + e^(-(z_1 - z_2))) = sigma(z_1 - z_2) $

つまり、ロジスティック回帰はSoftmax回帰の特殊ケースである。

== 多クラス交差エントロピー誤差

Softmax関数の出力 $y$ と、正解のOne-hotベクトル $t$ （例: $[0, 0, 1, 0]$）との間の誤差。

$ L = - sum_(k=1)^K t_k log y_k $

正解クラス（$t_k=1$）の確率 $y_k$ を最大化すること（尤度最大化）と同義。
逆伝播時の勾配が非常にシンプルになる美しい性質を持つ。
$ (partial L) / (partial z_k) = y_k - t_k $

= 確率的勾配降下法 (SGD) の理論

== 計算グラフと誤差逆伝播法

ニューラルネットワークの学習は、出力の誤差 $L$ を入力側へ逆流させることで勾配を求める。

#align(center)[
  #diagram(
    node-stroke: 1pt,
    spacing: (75pt, 20pt),
    node((0,0), $x$, name: <x>),
    node((1,0), $times W$, name: <mul>, shape: rect),
    node((2,0), $+ b$, name: <add>, shape: rect),
    node((3,0), $sigma$, name: <act>, shape: rect),
    node((4,0), $y$, name: <y>),
    node((5,0), $L(y, t)$, name: <loss>, shape: rect, stroke: 1pt + red),
    
    // Forward
    edge(<x>, <mul>, "->"),
    edge(<mul>, <add>, "->"),
    edge(<add>, <act>, "->"),
    edge(<act>, <y>, "->"),
    edge(<y>, <loss>, "->", label: "順伝播", label-pos: 0.5, label-side: left),
    
    // Backward
    edge(<loss>, <y>, "->", stroke: 1pt + red, bend: 40deg),
    edge(<y>, <act>, "->", stroke: 1pt + red, bend: 40deg),
    edge(<act>, <add>, "->", stroke: 1pt + red, bend: 40deg),
    edge(<add>, <mul>, "->", stroke: 1pt + red, bend: 40deg),
    edge(<mul>, <x>, "->", stroke: 1pt + red, bend: 40deg, label: text(fill: red)[逆伝播], label-pos: 0.5, label-side: left),
  )
]

連鎖律 (Chain Rule):
$ (partial L) / (partial x) = (partial L) / (partial y) dot (partial y) / (partial sigma) dot (partial sigma) / (partial z) dot (partial z) / (partial x) $
局所的な微分の積として、勾配を効率的に計算できる。

== 勾配降下法 (GD) の課題

目的関数 $L(w) = 1/N sum_(i=1)^N ell_i (w)$ の最小化。

*GD (Batch Gradient Descent):*
$ w_(t+1) = w_t - eta nabla L(w_t) $

- *計算コスト:* 1回の更新に全データ $N$ 個の勾配計算が必要。
- *局所解:* 凸でない場合、近くの極小値や鞍点 (Saddle Point) に捕まると動けなくなる。

== Stochastic Gradient Descent (SGD)

ランダムに選んだ1つのデータ（またはミニバッチ） $xi_t$ の勾配を使って更新する。

$ w_(t+1) = w_t - eta_t nabla ell(w_t, xi_t) $

- $nabla ell(w_t, xi_t)$ は真の勾配 $nabla L(w_t)$ の*不偏推定量*である。
  $ EE_xi [nabla ell(w_t, xi_t)] = nabla L(w_t) $
- 勾配に「ノイズ」が乗ることで、鞍点からの脱出が期待できる。

== Robbins-Monro 条件 (1/2)

SGDが最適解 $w^*$ に収束するための、学習率 $eta_t$ に関する十分条件。

1. *$sum_(t=1)^infinity eta_t = infinity$*
   - パラメータが初期位置からどこまででも遠くへ移動できる（探索範囲を限定しない）ために必要。

2. *$sum_(t=1)^infinity eta_t^2 < infinity$*
   - ノイズによる分散を時間の経過とともに抑え込み、最終的に一点に収束させるために必要。

== Robbins-Monro 条件 (2/2)

*満たす例:*
$ eta_t = 1 / t $ や $ eta_t = 1 / (t + c) $
（調和級数は発散し、二乗和は収束する）

*満たさない例:*
$ eta_t = "const" $ （定数）
- 二乗和が無限大になる。
- 実際には、定数の学習率だと最適解の周りでずっと振動し続ける（これがDeep Learningの実務では逆に良かったりもする）。

== 収束のイメージ

#grid(
  columns: (1fr, 1fr),
  gutter: 20pt,
  [
    *GD (Gradient Descent)*
    - 等高線に対して垂直に進む。
    - 谷底に一直線（振動しなければ）。
    - 鞍点では勾配が0になり停止する。
  ],
  [
    *SGD (Stochastic GD)*
    - フラフラとランダムウォークしながら進む。
    - ノイズのおかげで鞍点でも止まらずに抜け出せる。
    - 学習率を小さくしていけば、最終的に解に近づく。
  ]
)

= 実装演習: MNIST

== MNISTデータセットとは

手書き数字の画像データセット。機械学習の「Hello World」。

#grid(
  columns: (1fr, 1fr),
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
      width: 350pt,
      alt: "MNIST Dataset Examples",
    )
    $ 28 times 28 = 784 "pixels" $
  ]
)

== 実装に役立つアルゴリズム集 (Cheat Sheet)

3層MLP（入力 $x$, 隠れ層 $h$, 出力 $y$, 正解 $t$）の実装数式。
中間変数を $z$、要素ごとの積（アダマール積）を $dot.o$ とする。
*計算途中の値もすべて保存しておくこと*。

1. *順伝播 (Forward)*
   $
   z_1 &= W_1 x + b_1, quad h = sigma(z_1)\
   z_2 &= W_2 h + b_2, quad y = "softmax"(z_2)
   $

2. *逆伝播 (Backward)*

- *出力層の誤差* ($delta equiv partial L \/ partial z$):
  $ delta_2 = y - t $ (Softmax + CrossEntropy)
- *隠れ層の誤差*:
  $ delta_1 = (W_2^T delta_2) dot.o sigma'(z_1) $
  ※ シグモイドの場合: $sigma'(z_1) = h dot.o (1 - h)$

3. *勾配の計算 (Gradient)*
  $
  nabla W_2 = delta_2 h^T, quad nabla b_2 = delta_2\
  nabla W_1 = delta_1 x^T, quad nabla b_1 = delta_1
  $

4. *パラメータ更新 (SGD)*
  $ W arrow.l W - eta nabla W, quad b arrow.l b - eta nabla b $

== データの準備 (Python)

`scikit-learn` を使用して簡単にデータを取得し、前処理を行う例。

#text(size: 16pt)[
```python
from sklearn.datasets import fetch_openml
import numpy as np

# 1. データのダウンロード (時間がかかります)
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# 2. 前処理: 正規化 (0~255 -> 0.0~1.0)
X = X / 255.0

# 3. 前処理: One-hot Encoding (ラベル -> ベクトル)
# 例: 3 -> [0, 0, 0, 1, 0, ...]
I = np.eye(10) # 単位行列
Y_onehot = I[y]

# 4. 分割
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = Y_onehot[:60000], Y_onehot[60000:]
```
]

== データの準備 (Rust)

`mnist` クレートを使用する例。`Cargo.toml` に `mnist = "0.5"` を追加。

#text(size: 16pt)[
```rust
use mnist::{Mnist, MnistBuilder};
use ndarray::Array2;

fn main() {
    // 1. データのダウンロードと展開
    let Mnist { trn_img, trn_lbl, tst_img, tst_lbl, .. } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(60_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    // 2. Normalize & Vectorize (Vec -> ndarray)
    // 28*28 = 784次元。各画素を 0.0~1.0 に正規化
    let train_x: Array2<f64> = Array2::from_shape_vec((60_000, 784), trn_img)
        .unwrap()
        .mapv(|x| x as f64 / 255.0);
    
    // ラベル等の処理は別途One-hot変換が必要
}
```
]

別に方法はいくらでもあります。お好みで。

あと凸最適化の場合はパラメータの初期値を0にしても良いですが、ニューラルネットワークの場合はランダムに小さな値で初期化しないと学習が進まないので注意。

この感じで層を増やしたり、活性化関数を変えたりして実装してみましょう。
多分このままだと特に層を増やした時にうまく動作しなくなると思いますが、どうやって解決するかは次回以降のお楽しみ。

= Appendix

== 参考文献
- Robbins, H. and Monro, S. (1951). "A Stochastic Approximation Method".
- Cybenko, G. (1989). "Approximation by superpositions of a sigmoidal function".
- ultralytics MINISTデータセット: https://docs.ultralytics.com/ja/datasets/classify/mnist/#sample-images-and-annotations