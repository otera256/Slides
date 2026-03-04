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
    title: [機械学習勉強会 第6回],
    subtitle: [時系列データと言語モデル],
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

1.  *時系列データ*: 時間的な依存関係を持つデータをどのように扱うかを学ぶ。
2.  *RNN (回帰型ニューラルネットワーク)*: 過去の情報を保持する仕組みと、その学習上の課題（勾配消失・爆発）を理解する。
3.  *LSTM / GRU*: RNNの弱点を克服するゲート機構について数式ベースで理解する。
4.  *言語モデルとTransformer*: LLMの根幹をなす「次単語予測」タスクと、Attention機構の基礎を知る。

= 時系列データ

== 時系列データとは

*時間の経過に伴うデータの変化* を記録したデータ。
前回扱った画像（2次元の空間的構造）に対し、時系列データは「時間軸」に沿った構造を持つ。

- *例*: 株価の変動、気温の変化、音声波形、テキストデータ（単語の並び）など。

*特徴*:
- *時間的依存関係*: 過去の状態が未来の状態に影響を与える。
- *ノイズ・外れ値* が含まれることが多い。
- *季節性やトレンド*（周期的な変化や全体的な上昇・下降傾向）が存在する。
- *非定常性*: 時間とともに統計的な性質（平均や分散）が変化することがある。

== 時系列データを扱う戦略

*目的*: 過去の値に基づいて未来の値を予測する。

- *基本戦略*: 過去の値を「特徴量」として使用する。
  - 例: $t$ 日目の株価を予測するために、$t-1, t-2, t-3$ 日目の株価を入力とする。
- *問題点*: $i$ ステップ前ごとに別々の重み（モデル）を用意すると、パラメータ数が膨大になり、様々な長さの系列に対応できない。

$arrow.r$ *解決策*: 前回学んだCNNのように、*「時間方向」にパラメータを共有する* アーキテクチャを導入する。

= RNN (リカレントニューラルネットワーク)

== RNNの基本構造

時系列データの処理に特化したニューラルネットワーク。
*「過去の隠れ状態（記憶）」と「現在の入力」を組み合わせて、新しい状態を作る。*

#grid(
  columns: (1.5fr, 1fr),
  [
    $ h_t = sigma(W_h h_(t-1) + W_i x_t) $
    $ y_t = W_o h_t $
    - $sigma$: 活性化関数 (Tanhなど)
    - $h_t$: 時刻 $t$ の隠れ状態
    - $x_t$: 時刻 $t$ の入力
    - $W_h, W_i, W_o$: 各重み行列 (パラメータ共有)
  ],
  align(center + horizon)[
    #diagram(
      node-stroke: 1pt,
      spacing: (20pt, 30pt),
      node((0,2), $x_t$, name: <x>),
      node((0,1), $h_t$, name: <h>, shape: rect),
      node((0,0), $y_t$, name: <y>),
      // Recurrent edge
      edge(<h.east>, <h.south>, "->",bend: 150deg, label: $W_h$, label-side: left),
      edge(<x>, <h>, "->", label: $W_i$, label-side: left),
      edge(<h>, <y>, "->", label: $W_o$, label-side: left),
      
    )
  ]
)
ループ構造を持つことで、理論上は過去すべての情報を現在の状態 $h_t$ に圧縮して保持できる。

== 時間方向への展開 (Unrolling)

RNNのループ構造は、時間方向に展開することで、非常に深い多層のニューラルネットワークとみなすことができる。

#align(center)[
  #diagram(
    node-stroke: 1pt,
    spacing: (60pt, 40pt),
    for i in range(0, 8) {
      node((i,2), $x_#i$);
      node((i,1), $h_#i$);
      node((i,0), $y_#i$);
      edge((i,2), (i,1), "->", label: $W_i$, label-side: left);
      edge((i,1), (i,0), "->", label: $W_o$, label-side: left);
      if i > 0 {
        edge((i - 1, 1), (i, 1), "->", label: $W_h$, label-side: right, label-pos: 20%);
      }
    }
  )
]

通常はこれをRNNレイヤーとして定義し、その前後にMLP（全結合層）を繋げて使用する。

== RNNの学習: BPTT

展開した計算グラフに対して、通常の誤差逆伝播法を適用する。これを *BPTT (Backpropagation Through Time)* と呼ぶ。

- *問題点*: 
  - 系列長が長い（展開図が横に長い）と、非常に深いネットワークになる。
  - 時間方向に直列に計算する必要があるため、*GPUの並列計算の恩恵を受けにくい*。
  - 全時刻の中間状態をメモリに保持するため、*メモリ使用量が多くなる*。
  - 通常の層（MLPなど）は時間軸をバッチ次元のようにまとめて一気に計算し、RNNの箇所だけループを回すなどの実装上の工夫が行われる。

== 勾配消失と勾配爆発 (1/2)

第4回で「分散の維持」が重要だと学んだが、RNNは同じ重み $W_h$ を何回も掛け合わせるため、この問題が極めて深刻になる。

簡単のため、活性化関数を恒等写像 ($sigma(x) = x$) とし、入力を無視して隠れ状態の更新だけを考える：
$ h_t = W_h h_(t-1) arrow.r.double h_t = (W_h)^t h_0 $

- $W_h$ の固有値が *1より大きい* 場合：
  $h_t$ は指数関数的に増大し、勾配も爆発する $arrow.r$ *勾配爆発 (Exploding Gradient)*
- $W_h$ の固有値が *1より小さい* 場合：
  $h_t$ は指数関数的に減少し、勾配もゼロに近づく $arrow.r$ *勾配消失 (Vanishing Gradient)*

== 勾配消失と勾配爆発 (2/2): 対策

非線形関数 (Tanhなど) を使っても、微分値が掛け合わされるため同様の問題が起こる。これにより、RNNは*長期的な依存関係（ずっと昔の記憶）を学習するのが非常に困難*になる。

*解決策*:
1.  *アーキテクチャの工夫*: LSTM や GRU などの「ゲート機構」を導入する（後述）。
2.  *勾配クリッピング (Gradient Clipping)*:
    - 勾配のノルム（大きさ）が閾値を超えたら、ノルムが閾値に収まるように勾配ベクトルをスケーリングする。*勾配爆発*に対する強力な対抗策。
3.  *注意深い重みの初期化*:
    - $W_h$ の固有値が1に近くなるように（例：直交行列で）初期化する。

== [発展] 状態空間モデル (SSM) への布石

活性化関数が恒等写像の場合の線形代数的なアプローチ。
もし $W_h$ がエルミート行列（対称行列）なら、ユニタリ行列 $U$ と対角行列 $D$ を用いて $W_h = U^T D U$ と対角化できる。

$ h_t &= W_h h_(t-1) + W_i x_t \
      &= U^T D U h_(t-1) + W_i x_t $

これを展開すると、以下のような畳み込みの形になる。
$ h_t = h_0 + sum_(k=1)^t U^T D^(t-k) U W_i x_k $

対角行列の累乗 $D^(t-k)$ は計算が容易。さらに、この形式は *FFT (高速フーリエ変換) を用いて並列に高速計算* できる。
近年話題の *Mamba* や *S4 (Structured State Space sequence model)* は、この理論を発展させたものである。

= LSTM と GRU

== LSTM (Long Short-Term Memory)

勾配消失を防ぎ、長期記憶を実現するための特殊なRNNアーキテクチャ。
隠れ状態 $h_t$ とは別に、情報をバイパスさせる *セル状態 $c_t$* を持つ。

要素ごとの積（アダマール積）を $dot.o$ とすると、LSTMは以下の6つの式で定義される：
$ i_t &= sigma(W_i x_t + U_i h_(t-1) + b_i) quad &text("(入力ゲート)") \
f_t &= sigma(W_f x_t + U_f h_(t-1) + b_f) quad &text("(忘却ゲート)") \
o_t &= sigma(W_o x_t + U_o h_(t-1) + b_o) quad &text("(出力ゲート)") \
tilde(c)_t &= tanh(W_c x_t + U_c h_(t-1) + b_c) quad &text("(セル状態の候補)") \
c_t &= f_t dot.o c_(t-1) + i_t dot.o tilde(c)_t quad &text("(セル状態の更新)") \
h_t &= o_t dot.o tanh(c_t) quad &text("(隠れ状態の更新)") $

== LSTMにおける各ゲートの役割

各ゲートは $0 approx 1$ の値 ($sigma$) を出力し、情報の「通過量」をコントロールするバルブの役割を果たす。

- *忘却ゲート ($f_t$)*:
  前のセル状態 $c_(t-1)$ の情報を「どれくらい忘れる（捨てる）か」を決める。
- *入力ゲート ($i_t$)*:
  新しい候補 $tilde(c)_t$ を「どれくらいセル状態に追加するか」を決める。
- *出力ゲート ($o_t$)*:
  更新されたセル状態 $c_t$ のうち、「どれくらいを隠れ状態 $h_t$ として外に出すか」を決める。

*ポイント*: セル状態の更新 $c_t = f_t dot.o c_(t-1) + dots$ において、行列のかけ算ではなく「要素ごとの足し算と掛け算」が行われるため、条件が良ければ勾配が減衰せずに過去まで伝わる（エラーカルーセル）。

== GRU (Gated Recurrent Unit)

LSTMは強力だがパラメータが多いため、それを簡略化したアーキテクチャ。セル状態を廃止し、隠れ状態 $h_t$ のみで管理する。

$ z_t &= sigma(W_z x_t + U_z h_(t-1) + b_z) quad &text("(更新ゲート)") \
r_t &= sigma(W_r x_t + U_r h_(t-1) + b_r) quad &text("(リセットゲート)") \
tilde(h)_t &= tanh(W_h x_t + U_h (r_t dot.o h_(t-1)) + b_h) quad &text("(隠れ状態の候補)") \
h_t &= (1 - z_t) dot.o h_(t-1) + z_t dot.o tilde(h)_t quad &text("(隠れ状態の更新)") $

- *リセットゲート ($r_t$)*: 過去の記憶をどれくらい無視して新しい候補を作るかを決める。
- *更新ゲート ($z_t$)*: 過去の記憶と新しい候補の「ブレンド割合」を決める。LSTMの忘却・入力を1つのゲートにまとめたような役割。

= 言語モデル

== 言語モデルとは

自然言語のテキストデータを処理し、ある単語の列が与えられたときに *「次に来る単語」の確率分布を予測するタスク*（自己教師あり学習）。

- *例*: `["The", "apple", "is"]` $arrow.r$ 次は `"red"` が 60%、`"ripe"` が 20%...
- 昔は *n-gramモデル*（統計的な出現頻度）が主流だったが、今はニューラルネットワークが主流。
- 翻訳、要約、質問応答など、あらゆる自然言語処理の基盤技術であり、ChatGPTなどの対話型AIの根本でもある。

== 言語モデリングの典型的なフロー

1. *コーパス収集*: 大量のテキストデータを集める。
2. *Tokenization*: テキストを「トークン（文字、単語、サブワードなど）」に分割する。
3. *Embedding*: トークンを数値ベクトルに変換する。
4. *モデル学習*: 入力トークン列から、1つ未来のトークンを予測するように学習する。
   $ "Loss" = - sum t_i log y_i quad text("(交差エントロピー誤差)") $

#note(title: "Perplexity (パープレキシティ)")[
  言語モデルの性能評価指標。クロスエントロピー損失 $L$ の指数関数 $exp(L)$ で定義される。
  「次の単語の選択肢を、平均して何個まで絞り込めているか」を表す。*値が小さいほど高性能*。
]

== Tokenization と Embedding

- *Tokenization (トークン化)*:
  - 単位: 文字（細かすぎ）、単語（未知語に弱い）、*サブワード*（BPEなど。現在の主流。単語をよく出るパーツに分ける）。
  - 語彙サイズ (Vocabulary Size): モデルが直接扱えるトークンの種類数。計算量に直結する。

- *Embedding (埋め込み)*:
  - トークンのID（整数）を、高次元の連続空間上のベクトルに変換する（例: `ID:25` $arrow.r$ `[0.1, -0.4, 0.8, ...]`)。
  - 意味が似ているトークンは、ベクトル空間上でも近くに配置されるようになる。
  - 現在はTransformerの学習と同時にEnd-to-Endで学習される。

#text(size: 0.8em)[参考動画: 言語モデルの全体像 https://youtu.be/KlZ-QmPteqM?si=wRu8YPqBar4__bUp]

= Transformer

== Transformer の登場

2017年の論文 *"Attention is All You Need"* で提案されたアーキテクチャ。
RNNやCNNを使わず、*Self-Attention機構のみ*で構成されており、言語モデルの分野に革命をもたらした。

*RNNの弱点の克服*:
- RNN: 過去から順番に処理するため、長距離の依存関係が苦手で、並列計算ができない。
- *Transformer*: シーケンス内の *全トークンの関係性を一度に計算* するため、遠く離れた単語の依存関係も一瞬で捉えられ、*GPUでの超並列計算* が可能。

== Self-Attention (自己注意機構)

入力シーケンス内の各トークンが、他のどのトークンに「注意を払うべきか」を計算する。

$ "Attention"(Q, K, V) = "softmax"( (Q K^T) / sqrt(d_k) ) V $

- $Q$ (Query): 「自分は何を探しているか」
- $K$ (Key): 「自分は何者か」
- $V$ (Value): 「自分が持っている情報の中身」
- $Q K^T$: 検索クエリとキーの関連度（内積）を計算する。関連度が高いほど、そのトークンの $V$ を強く引っ張ってくる。
- $sqrt(d_k)$: スケーリングファクター。内積の値が大きくなりすぎてSoftmaxの勾配が消失するのを防ぐ。

== Transformer の特徴とまとめ

- *計算量*: 系列長 $N$ に対して $O(N^2)$ の計算量が必要。非常に長い文章は工夫なしでは扱えないが、GPUの行列計算との相性は抜群。
- *最新AIの標準*: LLM（GPT, Claude等）だけでなく、画像（ViT）や音声処理など、あらゆる分野のデフォルト・スタンダードとなっている。
- 内部にはMulti-Head Attentionや位置エンコーディング(Positional Encoding)など様々な工夫があるが、コアとなるアイデアは非常にシンプル。

#text(size: 0.8em)[参考動画: Transformerの全体像 https://youtu.be/j3_VgCt18fA?si=V50Pjw_xxfAkC-BR]

#note(title: "推薦文献")[
  機械学習の論文を読み始めるなら、まずは原著論文 *"Attention is All You Need"* (https://arxiv.org/abs/1706.03762) を読むことを強くお勧めします。
]

== 今回のまとめ

1.  *RNN*: 時系列データを扱う基本。ただし、直列計算による速度低下と、勾配消失・爆発という致命的な弱点があった。
2.  *LSTM / GRU*: ゲート機構によって勾配の消失を防ぎ、長期的な記憶を可能にした立役者。
3.  *言語モデル*: テキストをトークン化・埋め込みし、「次に来る単語」を予測し続けるタスク。
4.  *Transformer*: RNNを駆逐した現代の覇者。Attention機構により、長距離の依存関係と圧倒的な計算並列性を手に入れた。