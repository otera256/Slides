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
    title: [機械学習勉強会 第5回],
    subtitle: [正規化・正則化の補足とCNN],
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

// 重要・注釈 (Note): 緑系
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

1.  *正規化・正則化*: Batch NormやDropoutの詳細な挙動と、実装時の落とし穴を理解する。
2.  *CNN (畳み込みニューラルネットワーク)*: 画像処理のデファクトスタンダードであるCNNの仕組み（畳み込み、プーリング、チャネル）を理解する。

== 先週の復習・課題確認

- *自動微分*: 計算グラフと連鎖律により、複雑な関数の勾配を機械的に求めた。
- *課題*: MNIST / Fashion-MNIST の実装とスコア比較。
  - 適切に実装できていれば、MNISTで90%以上、Fashion-MNISTでも80%以上の精度が出ているはず。
  - 出ていない場合：学習率、初期値、バッチサイズ、エポック数を見直そう。

= 正規化・正則化の補足

== 前処理としての正規化

学習を安定させるため、入力データのスケールを揃えることが重要。

- *正規化 (Normalization)*: データを $[0, 1]$ の範囲に収める。
  $ x' = (x - x_"min") / (x_"max" - x_"min") $
  - 画像データ ($0 ~ 255$) を $255$ で割るなど。

- *標準化 (Standardization)*: 平均 $0$、分散 $1$ に変換する。
  $ x' = (x - mu) / sigma $
  - 多くの機械学習アルゴリズム（特にSVMや線形回帰）で標準的。Deep Learningでもよく使われる。

== Batch Normalization (復習と詳細)

各層の入力を、ミニバッチごとの統計量を用いて正規化する。

#definition(title: "Batch Normalization")[
  ミニバッチ $B$ に対し、平均 $mu_B$、分散 $sigma_B^2$ を計算。
  $ hat(x)_i = (x_i - mu_B) / sqrt(sigma_B^2 + epsilon), quad y_i = gamma hat(x)_i + beta $
  $gamma, beta$ は学習可能なパラメータ（スケールとシフト）。
]

*利点*:
1.  学習率を大きくできる（勾配消失・爆発しにくい）。
2.  初期値への依存度が下がる。
3.  多少の正則化効果（過学習抑制）がある。

== Batch Norm の注意点 (1/2): Train vs Eval

*最重要*: 学習時と推論時で挙動が異なる。

- *学習時 (Training)*:
  - その瞬間のミニバッチの平均・分散を使って正規化する。
  - 同時に、全体の平均・分散の「移動平均 (Running Mean/Var)」を更新し続ける。

- *推論時 (Evaluation/Testing)*:
  - ミニバッチ統計量は使わない（入力が1枚かもしれないため）。
  - 学習中に蓄積した*移動平均*を使って正規化する。

#note(title: "実装上の注意")[
  PyTorchでは `model.train()` と `model.eval()` の切り替えを忘れると、推論時の精度がボロボロになる。
]

== Batch Norm の注意点 (2/2): バッチサイズと代替手法

- *バッチサイズ問題*:
  - ミニバッチサイズが小さい（例: 2, 4）と、統計量が不安定になり性能が落ちる。
- *学習率*:
  - BNを入れると損失局面が滑らかになるため、学習率を大きめに設定するのがコツ。

*代替手法 (Alternatives)*:
- *Layer Norm*: サンプル単位で正規化（RNN/Transformer向け）。
- *Instance Norm*: チャネル単位で正規化（スタイル変換向け）。
- *Group Norm*: チャネルをグループ化して正規化（バッチサイズが小さいCNN向け）。

== Dropout (詳細)

学習時にランダムにニューロンを無効化する。

#grid(
  columns: (1fr, 1fr),
  align(center)[
    // 画像: Dropoutの概念図（ノードが×になっている図）
    // week4のスライドと同じような図を想定
    #diagram(
      spacing: 20pt,
      node((0,0), shape: circle, fill: gray), node((1,0), shape: circle, stroke: gray, label: $times$),
      node((0,1), shape: circle, stroke: gray, label: $times$), node((1,1), shape: circle, fill: gray),
      edge((0,0),(1,1)),
    )
    *Training (確率 $p$ でDrop)*
  ],
  align(center)[
    // 画像: 全結合の図
    #diagram(
      spacing: 20pt,
      node((0,0), shape: circle, fill: gray), node((1,0), shape: circle, fill: gray),
      node((0,1), shape: circle, fill: gray), node((1,1), shape: circle, fill: gray),
      edge((0,0),(0,1)), edge((0,0),(1,1)),
      edge((1,0),(0,1)), edge((1,0),(1,1)),
    )
    *Inference (全ノード使用)*
  ]
)

- *アンサンブル学習効果*: 異なる部分ネットワークを多数学習させて平均を取るのと等価。
- *適用場所*: 全結合層の直前・直後が多い。畳み込み層にはあまり使わない（Spatial Dropout等は別）。

== Dropout の注意点

1.  *推論時のスケーリング*:
    - 学習時に $p$ の割合で消していたため、推論時にそのまま全ノードを使うと出力値の合計が大きくなりすぎる。
    - *Inverted Dropout* (実装の主流): 学習時に出力を $1/(1-p)$ 倍しておく。これで推論時は何もしなくて良くなる。

2.  *ドロップアウト率*:
    - 一般的に $0.5$ (50%) が多い。入力層付近では $0.2$ 程度など低めにするか、適用しない。高すぎると学習が進まない。

== 正則化 (Regularization)

モデルの複雑さにペナルティを与え、過学習を防ぐ。

*L2 正則化 (Weight Decay)*
損失関数 $L$ に、重みの二乗和を加える。
$ L_"new" = L_"old" + lambda / 2 sum w^2 $

- *直感的理解*:
  - 「重み $w$ が大きくなること」＝「特定の入力特徴量に過剰に反応すること」。
  - これを罰することで、決定境界を滑らかにする。
  - 勾配更新式で見ると、毎回 $w$ を少し $0$ に近づける（減衰させる）操作になる。

#text(size: 0.8em)[参考: ヨビノリコラボ動画 https://zero2one.jp/learningblog/yobinori-collab-regularization/]

= 畳み込みニューラルネットワーク (CNN)

== 全結合層 (MLP) の限界

画像（例: $28 times 28$）を1列のベクトル（$784$次元）として扱うと...

1.  *空間情報の欠落*: 上下左右のピクセルのつながり（形状）を無視してしまう。
2.  *パラメータ爆発*: 画像サイズが大きくなると（例: $1000 times 1000$）、重み行列が巨大になりすぎて学習困難。

$arrow.r$ *畳み込み層 (Convolutional Layer)* の導入

この概念については自分で説明するより動画を見たほうが１００倍わかりやすいのでまず下の動画を見ましょう

参考動画: 3Blue1BrownJapan|畳み込みの仕組み https://youtu.be/CHx6uHnWErY?si=PDGv5vTxuqCwOJH-

== 畳み込み層の仕組み

「フィルタ（カーネル）」をスライドさせながら、局所的な積和演算を行う。

// 画像: 3Blue1Brownの動画のような、フィルタが画像をスキャンしていくアニメーションのイメージ
// https://github.com/3b1b/manim
#align(center)[
  #[フィルタによる局所特徴の抽出イメージ]
  
  $ mat(1, 0, 1; 0, 1, 0; 1, 0, 1) times mat(1, 2, 3; 4, 5, 6; 7, 8, 9) arrow.r "積和" $
]

- *局所受容野*: あるニューロンが見る領域は画像の一部だけ。
- *重み共有*: 画面のどこにあっても「エッジ」は「エッジ」。同じフィルタを画像全体で使い回す（パラメータ削減 & 平行移動不変性）。

== 畳み込みの構成要素

1.  *フィルタ (Kernel)*: 重み行列。$3 times 3$ や $5 times 5$ が一般的。
2.  *ストライド (Stride)*: フィルタを動かす歩幅。大きくすると出力サイズが小さくなる。
3.  *パディング (Padding)*: 端の情報を維持するため、周囲を0などで埋める。

#definition(title: "出力サイズの計算式")[
  入力サイズ $H, W$、フィルタサイズ $K$、パディング $P$、ストライド $S$ のとき：
  $ H_"out" = floor((H + 2P - K) / S) + 1 $
]
パディング $P=1$、ストライド $S=1$、$3 times 3$ フィルタならサイズは変わらない。

== チャンネル (Channel) の概念

*ここが初心者の躓きポイント！*

- 入力画像は通常 *3次元* ($H times W times C$)。例: RGBなら $C=3$。
- フィルタも実は *3次元* ($K times K times C_"in"$)。
  - 入力の全チャネルにわたって積和し、1枚の出力マップを作る。
- そのフィルタが $C_"out"$ 個ある。
  - 結果、出力は ($H' times W' times C_"out"$) になる。

#align(center)[
  // 図解: (H, W, 3) * (3, 3, 3)のフィルタ => (H', W', 1)
  // これが N個ある => (H', W', N)
  $ "Input": [H, W, C_"in"] arrow.r "Filter": [C_"out", K, K, C_"in"] arrow.r "Output": [H', W', C_"out"] $
]

== 畳み込みの実装 (Im2Col)

for文で画素を回すと遅い。*Im2Col (Image to Column)* で行列積に変換する。

1.  フィルタを適用する領域（パッチ）を切り出し、横1列に並べる。
2.  これを全領域分行い、巨大な行列を作る。
3.  フィルタも1列に並べる。
4.  巨大な行列 $times$ フィルタ行列 を一発で計算（GPUが得意）。
5.  計算結果を元の画像の形に戻す (Col2Im)。

// 画像: Im2Colの展開図。ブロックを行列に変換する様子。

== プーリング層 (Pooling Layer)

特徴マップを縮小（ダウンサンプリング）する。

- *Max Pooling*: 領域内の*最大値*を取る。
  - 特徴が「あるかどうか」を強調。位置ズレに強くなる（ロバスト性）。
- *Average Pooling*: 領域内の*平均値*を取る。
  - 全体的な特徴を滑らかにする。

*特徴*: 学習するパラメータが無い。
最近は「ストライド付き畳み込み」で代用し、プーリングを使わないモデルもある。

= CNNアーキテクチャと応用

== 代表的なCNNアーキテクチャ

歴史を知ることで、なぜ今の形になったかが分かる。

1.  *LeNet-5 (1998)*: 元祖CNN。MNISTで郵便番号認識。Conv-Pool-Conv-Pool-FC。
2.  *AlexNet (2012)*: Deep Learningブームの火付け役。ReLU、Dropoutを採用。層を深くした。
3.  *VGG (2014)*: $3 times 3$ の小さなフィルタを重ねるのが有効と示した。シンプルで美しい。
4.  *ResNet (2015)*: *スキップ結合 (Residual Connection)* を導入。100層以上の学習が可能に。現在のデファクトスタンダード。

// 画像: 各ネットワークの構造図（層の深さ比較など）

== フィルタの可視化

CNNは何を見ているのか？

- *浅い層*: エッジ（縦線、横線）、色、テクスチャなどの単純な特徴。
- *深い層*: 目、タイヤ、幾何学模様など、意味のある複雑なパーツ。

階層的に特徴を組み合わせて物体を認識している。

== CNNの応用

画像分類 (Classification) だけではない。

- *物体検出 (Object Detection)*: YOLO, SSD。画像の「どこに」「何が」あるか。
- *セグメンテーション (Segmentation)*: U-Net。画素単位でクラス分類（背景切り抜きなど）。
- *1D-CNN*: 時系列データ、音声波形、自然言語処理。
- *3D-CNN*: 動画処理（時間軸を3次元目とみなす）、CT/MRI画像（立体）。

= まとめ

== 今回のまとめ

1.  *Batch Norm*: 学習を爆速にする魔法。ただしTrain/Evalモードの切り替えは命取り。
2.  *Dropout*: 過学習を防ぐアンサンブル効果。推論時のスケーリングに注意。
3.  *CNN*:
    - *畳み込み*: フィルタによる局所特徴抽出 + パラメータ共有。
    - *チャネル*: 立体的な特徴マップの変化を意識する。
    - *プーリング*: 情報を圧縮し、位置ズレを許容する。

次回、いよいよCNNの実装に入ります。Pytorchなら `nn.Conv2d` 一行ですが、中身を知っていることがデバッグ力に繋がります。