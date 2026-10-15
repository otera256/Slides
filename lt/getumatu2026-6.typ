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
    title: [おすすめのSlackチャンネル],
    author: [49th furakuta],
    date: [2026-06-29],
    institution: [#image("../images/KMClogo_trans.png", height: 2em)],
  )
)

#set text(lang: "ja", font: "UD Digi Kyokasho NP")
#set heading(numbering: numbly("{1}.", default: "1.1"))

// 数式ブロックのスタイル
#show math.equation.where(block: true): it => block(inset: 10pt, radius: 5pt, fill: rgb("#f0f0f0"), width: 100%, it)

#show raw.where(block: false): it => box(inset: 7pt, radius: 5pt, baseline: 20%, fill: rgb("#f0f0f0"), it)

#show raw.where(block: true): it => block(inset: 10pt, radius: 5pt, fill: rgb("#f0f0f0"), width: 100%, it)

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

= 自己紹介
#slide(composer: (1fr, 2.5fr))[
  #align(center)[
    #image("../images/furakuta.png", height: 10em)
  ]
][
  - kmc_id: furakuta
  - 所属: 京都大学工学部情報学科数理工学コース2回生
  - KMC49代副会長
  #pause
  - KMCでの活動分野
    #pause
    - 機械学習: 機械学習勉強会サボっていてごめんなさい
    #pause
    - ゲーム開発: 最近やっていない。Bevy engineはおすすめ?
    - 競技プログラミング: ICPCが近づいてるけど特に何もしていない。ちなみに昨日のARCはC問題で5WA
  #pause
  - 最近VRChatのプレイ時間が増えてきた (だってHMD買ったから使わないともったいないもん)\
    休みが多いわけでもないのに過去2週間で37.5時間!?
]

= 本題
#focus-slide[
  皆さんSlackは使っていますか?
]

#slide[
  #align(center)[
    #text(1.3em)[
      もう何十個もチャンネルに入ってるしSlack完全理解?
      #pause

      そんなわけ無い
      #pause

      KMC Slackには1680個のチャンネルがある
    ]
  ]
]

#image-slide(
  align(center)[
    #image("images/channel_distribution.png", height: 100%)
  ]
)

== 過去30日で投稿したメンバー数が3人以上で投稿件数が多いtop20チャンネル
#align(center)[
  #image("images/top_posts20-11.png", height: 100%)
]
#pagebreak()
#align(center)[
  #image("images/top_post10-1.png", height: 100%)
]

== おすすめのチャンネルをPick UP!!
=== \#ai-api-sandbox
Slack上でつよつよAIに質問、画像生成、HTML&部内サーバーへのデプロイなどができるチャンネル。murataさんがAPI代金を負担してくれている。
#align(center)[
  #image("images/ai-api-sandbox.png", height: 72%)
]

=== \#zatsu
大喜利チャンネル。OBの方がよく投稿している。
#align(center)[
  #image("images/zatsu.png", height: 72%)
]

=== \#にゃーん
「にゃーん」と鳴くチャンネル。日々に疲れたあなたにおすすめ
#align(center)[
  #image("images/nya-n.png", height: 72%)
]

=== \#イベント案出し
イベントを沢山やっていきたいから案を出していこうというチャンネル。すでにここから生まれたイベントもいくつかある。
#align(center)[
  #image("images/event-idea.png", height: 72%)
]

=== \#daimonjiyama_rta
大文字山の入山、火床の時間を記録するチャンネル。sashiさんがほぼ毎日登っている事がわかる。
#align(center)[
  #image("images/daimonjiyama_rta.png", height: 72%)
]

=== \#sushi-pizza-provider-san
主に外部の無料で飲食できる技術系のイベント情報を提供してくれるチャンネル。寿司ピザprovider-sanに感謝。
#align(center)[
  #image("images/sushi-pizza-provider-san.png", height: 72%)
]

=== \#毎日〇〇するだけ
分野ごとにチャンネルが立ってる

継続は力なり

- \#毎日コーディングするだけ
- \#毎日モデリングするだけ
- *\#毎日絵を描いて上げるだけ*
- *\#毎日競プロやるだけ*
- \#毎日シェーダーを書いて上げるだけ
- \#毎日授業に行くだけ
- *\#毎週土曜にゲーム考えて上げるだけ*

=== \#furakuta-memo
私のメモチャンネル。入ってね。
#align(center)[#image("images/furakuta-memo.png", height: 72%)
]