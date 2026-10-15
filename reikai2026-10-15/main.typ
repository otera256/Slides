#import "@preview/touying:0.7.0": *
#import themes.metropolis: title-slide, slide
#import "../templates/kmc.typ": *
#import "diagrams/tracking.typ" as diagrams
#import "sources.typ": cite, source-line
#import "product-image-slides.typ" as product-images

#show: kmc-theme.with(
  title: [VR向けトラッキングシステムの原理],
  subtitle: [部屋にコイルを張って、IMUのドリフトを補正したい],
  date: [2026-10-15],
  institution: image("../images/KMClogo_trans.png", height: 1.4em),
  notes: sys.inputs.at("notes", default: "false") == "true",
)

#let notes = json("speaker-notes.json")
#let oral(n) = speaker-note(
  setting: body => {
    set text(size: 16pt)
    set par(leading: 0.35em, spacing: 0.6em)
    body
  },
  notes.at(str(n)).replace("**", ""),
)

#title-slide()

// 説明18枚はoutline.mdと対応。画像6枚は各節の持ち時間内で提示する。
// 予定28分＋質疑2分。
#slide(title: [自己紹介と、自作したい理由])[
  #columns(widths: (2fr, 1fr))[
    *furakuta* ／ KMC49代副会長

    京都大学工学部 情報学科 数理工学B2

    機械学習・ゲーム開発(今年してない)\
    競技プログラミング(やる気低迷中)\
    鯖缶(サーバー管理)勉強会をNF以降に主催の予定

    #pause

    #image("references/vrchat_playtime.png", width: 50%)
  ][
    #align(right)[#image("../images/furakuta.png", width: 100%)]
  ]
  #oral(1)
]


#slide(title: [VRが知りたいのは位置と姿勢])[
  *トラッキング*：位置と姿勢を時間とともに推定する。合わせて #emphasis[6DoF（6自由度）]。
  #v(0.45em)
  #align(center)[#diagrams.pose()]
  #caption[HMDは頭に着けるディスプレイ。「姿勢」は機器の向きを指す。]
  #source-line("R05")
  #oral(2)
]

#slide(title: [頭と手の計測から全身の動きへ])[
  #align(center)[#diagrams.skeleton()]
  #takeaway[頭・手が同じ位置でも、脚の姿勢は一つに決まらない。]
  #caption[IK（逆運動学）は、手足の到達位置などから関節の曲がり方を求める処理。\ 追加の観測と骨長などの拘束を使い、未観測の関節も推定する。]
  #source-line("R01", "R02")
  #oral(3)
]

#product-images.full-body-images()

#slide(title: [方式を「何を測るか」で整理する])[
  #plain-table(
    columns: (1.15fr, 1.55fr, 1.5fr, 1.45fr),
    table.header([*方式*], [*直接観測するもの*], [*位置・姿勢の基準*], [*主な難しさ*]),
    [Lighthouse], [赤外線掃引の受光時刻], [部屋に固定した局], [遮蔽・反射・設置],
    [カメラ], [特徴点・マーカーなど], [環境地図／外部カメラ], [視野・遮蔽・照明],
    [IMU], [角速度・比力], [初期状態・重力など], [バイアス・ドリフト],
    [磁場], [磁場ベクトルなど], [既知の磁場源・地図], [歪み・配置・較正],
  )
  #caption[誤差＝真値からのずれ　ジッタ＝細かな揺れ\ ドリフト＝蓄積するずれ　遅延＝反映までの時間]
  #source-line("R04", "R05", "R10", "R13")
  #oral(4)
]

#slide(title: [Lighthouse：受光時刻から方向を測る])[
  #align(center)[#diagrams.lighthouse()]
  #takeaway[受光の時刻から「方向」を得る。光の往復時間で距離を測る方式ではない。]
  #caption[図は1回の掃引の断面。実際は複数の掃引と受光点を使い、IMUも併用する。見通しや反射が観測に影響する。]
  #source-line("R04", "R09")
  #oral(5)
]

#product-images.lighthouse-products()
#product-images.lighthouse-principle()

#slide(title: [カメラ方式：画像の中の目印を追う])[
  #align(center)[#diagrams.cameras()]
  #v(0.4em)
  - 機器側から部屋を見るか、部屋側から対象を見るかの違い
  - SLAM：部屋の地図作成と、自己位置・姿勢の推定を同時に行う
  #caption[カメラとIMUの融合で画像の間の動きも推定できる。どの方式も、使う目印や対象がカメラに見える必要がある。]
  #source-line("R05", "R06", "R07", "R08")
  #oral(6)
]

#product-images.camera-product()

#slide(title: [IMU：速く動きを追えるが、基準が必要])[
  #columns(widths: (1.2fr, 1fr))[
    - ジャイロ：角速度を測り、姿勢を更新
    - 加速度計：比力を測る。静止時にも重力に対応する値が出る
    - 座標変換と重力補償の後、速度・位置を積分する

    #takeaway[小さな誤差が、積分で蓄積する。]
  ][
    一定の加速度誤差の例
    $ delta a = 0.01 thin "m/s²" $
    $ delta p = 1/2 delta a t^2 $
    #align(center)[#text(size: 25pt)[10秒で #emphasis[約0.5 m]]]
    #caption[特定製品の性能値ではなく、積分の影響を示す例。]
  ]
  #caption[*SlimeVR*：骨の向きと長さをつないで位置を再構成（順運動学）。HMDを位置の基準にする。]
  #source-line("R02", "R10")
  #oral(7)
]

#product-images.imu-products()

#slide(title: [磁場方式：見えなくても観測できる基準])[
  - 既知の磁場を作り、観測した磁場から位置・姿勢を推定
  - 衣服や身体による光学的な遮蔽を避けられる
  - 磁場の歪み、距離による信号の弱さが課題

  #v(0.6em)
  #plain-table(columns: (1fr, 2.5fr),
    [*製品例：FluxPose*], [腰のビーコンを基準に、磁場と慣性センサーを融合],
    [*今回の自作案*], [部屋に固定した大型コイルで、正負に切り替えた磁場を観測],
  )
  #source-line("R12", "R13")
  #oral(8)
]

#product-images.magnetic-products()

#slide(title: [自作案の全体像])[
  #align(center)[#diagrams.architecture()]
  #v(0.6em)
  #align(center)[
    #emphasis[目標：長時間使う際の、手動リセットの頻度を減らす]
  ]
  #caption[候補：IMUはLSM6DSV16X、磁気はBMM350、通信はXIAO ESP32S3。]
  #source-line("P01", "R16", "R17")
  #oral(9)
]

#slide(title: [大型コイルで位置ごとに違う磁場を作る])[
  #align(center)[#diagrams.field-gradient()]
  #takeaway[位置を区別するには、磁場の強さだけでなく勾配が必要。]
  #caption[図は姿勢を固定した1次元の局所近似で、磁場の1成分を示す。3次元では各方向の感度を調べる。]
  #caption[小型コイルの遠方では $B prop r^(-3)$。部屋規模のコイルはBiot–Savart則で実形状を計算し、導線量・電力と空間内の感度を比較する。]
  #source-line("R11", "P01")
  #oral(10)
]

#slide(title: [電流を反転してコイルの磁場を取り出す])[
  #align(center)[#diagrams.reversal()]
  $ m_+ = m_"bg" + B quad m_- = m_"bg" - B quad => quad B = (m_+ - m_-)/2 $
  #takeaway[同じ位置・姿勢で測ると、変化しない背景が差分で消える。]
  #caption[3軸それぞれで計算。正負の電流は実測で確認する。0状態は背景と整定の監視に使う。]
  #oral(11)
]

#slide(title: [複数コイルを順番に測る])[
  #align(center)[#diagrams.tdma()]
  #v(0.4em)
  - TDMA（時分割）：どのコイルの磁場かを区別する
  - 測定時刻を保持し、切替後の安定した区間だけ使う
  #takeaway[動作中は、各時刻の位置・姿勢を使って観測を扱う。]
  #caption[600 msは説明用の設計例。待ち時間と平均時間は実機で決める。]
  #oral(12)
]

#slide(title: [磁場から位置と姿勢を推定する])[
  #align(center)[#diagrams.field-match()]
  $ bold(d)_k approx R^top bold(B)_k (bold(p)) $
  #caption[$R$：センサー座標から部屋座標への回転。\ $R^top$：部屋の磁場をセンサー座標へ変換。$k$：コイル番号。]
  #takeaway[複数の独立した磁場パターンとIMUの事前推定を使う。]
  #caption[同じコイルの正負は一つのパターンの符号反転。静止・同期済みの観測モデルを示している。]
  #oral(13)
]

#slide(title: [速いIMUと遅い磁場観測を融合する])[
  #align(center)[#diagrams.fusion()]
  #v(0.4em)
  #takeaway[磁場の更新を待つ間も、IMUで現在の推定を出し続ける。]
  #caption[過去状態を保持する方法の一つがFixed-lag smoothing。時計の同期、運動中の観測モデル、補正時の出力の跳びを検証する。]
  #caption[VR出力前にHMDと座標系を合わせる。仮想トラッカーやOSC Trackersを検討。]
  #source-line("R14", "R03")
  #oral(14)
]

#slide(title: [部屋の金属と磁場マップ])[
  $ bold(B)_(k, "real") (bold(p)) = bold(B)_(k, "physics") (bold(p)) + Delta bold(B)_(k, "room") (bold(p)) $
  #plain-table(columns: (1fr, 2fr),
    [*固定した環境*], [再現性のある歪みを実測マップへ取り込む候補],
    [*動く金属・家具*], [地図自体が変わり、以前の地図との照合が崩れる],
    [*コイルに同期する歪み*], [正負の差分にも残るため、モデル・較正が必要],
  )
  #takeaway[まず物理モデルと実測を比較し、残差の再現性を確かめる。]
  #caption[鉄・鋼などの強磁性体、切替時の導体の渦電流に注意。既知地図との照合はlocalization。地図も同時推定する段階がSLAM。]
  #source-line("R13")
  #oral(15)
]

#slide(title: [シミュレーションの表示と未検証の点])[
  #columns(widths: (1.2fr, 1fr))[
    #image("references/simulation-coverage.png", width: 100%)
    #caption[参照会話の評価画面。白色・独立ノイズを仮定したモデル上の表示。]
  ][
    #text(fill: colors.primary)[*79.3%*]

    最悪方向の $1 sigma$ が10 mm以下\
    となる空間の割合

    #v(0.3em)
    #text(size: 20pt)[
      実効ノイズ：19 / 19 / 45 nT\
      1秒平均・有効100サンプル
    ]
    #caption[実測の達成率ではない。姿勢・運動・環境歪みを含めた評価が次の課題。]
  ]
  #caption[CRLBは仮定したモデルの下での分散の理論的下限。画像だけでは姿勢の扱い・独立モード等の実装は未検証。]
  #source-line("P01", "R15", "R16")
  #oral(16)
]

#slide(title: [最初の試作と検証計画])[
  #columns(widths: (1.3fr, 1fr))[
    1. 1センサー・1コイルで正負反転を測る
    2. 既知の位置で理論場と実測場を比較
    3. 複数コイルで測位し、IMUを融合
    4. 30分間のずれとリセット回数を測る
  ][
    *最初のコイルの概算*

    #text(size: 20pt)[
      2.8 m角 × 4巻 ＝ 44.8 m\
      線径0.65 mm、5 V\
      約2.1 A ／ 約8.6 At
    ]
    #caption[配線・駆動回路の損失などを無視した概算。シミュレーションの30 / 50 Atとは異なる条件。]
  ]
  #takeaway[IMUのみの場合と比較し、独立した位置基準で評価する。]
  #source-line("P01")
  #oral(17)
]

#slide(title: [まとめ])[
  #text(size: 23pt)[
    - トラッキングは、観測できる物理量から\ 位置・姿勢を求める推定問題
    - 自作案では、正負に切り替えた部屋の磁場を\ IMUの長期的な基準にする
    - 次に確かめたいのは、実際の部屋で、\ 動きながらドリフトを抑えられるか
  ]
  #v(0.7em)
  #align(right)[#emphasis[ご清聴ありがとうございました。]]
  #oral(18)
]

// 参考資料は本編の後。資料名はクリック可能。
#include "reference-slides.typ"

// 発表後の閲覧用。technical-notes.md の説明と数式を収録。
#include "supplement-slides.typ"
