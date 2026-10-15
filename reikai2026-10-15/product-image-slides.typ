// 既存の説明スライドとは独立させた、製品画像と公式解説図の引用。
#import "@preview/touying:0.7.0": *
#import "../templates/kmc.typ": colors, image-slide

#let image-sources = json("image-sources.json")
#let asset(id) = image-sources.find(it => it.id == id)
#let photo(id, width: 100%, height: 10.5em) = image(
  asset(id).path, width: width, height: height, fit: "contain",
)
#let image-credit(..ids) = {
  set text(size: 11pt, fill: colors.muted)
  for id in ids.pos() {
    let item = asset(id)
    block(above: 0.1em, below: 0pt)[
      画像引用 #id：#item.publisher「#link(item.page_url)[#item.title]」
      （#item.accessed 閲覧）
    ]
  }
}
#let photo-label(body) = align(center, text(size: 20pt, body))

#let full-body-images() = image-slide(
  title: [フルトラッキングの装着例と利用イメージ],
  description: [腰や足に付けた機器の動きを、仮想の身体へ反映する。],
  credit: image-credit("I02", "I01"),
)[
  #grid(columns: (1fr, 1.5fr), gutter: 1em, align: horizon,
    [#photo("I02") #photo-label[身体に装着するトラッカー]],
    [#photo("I01") #photo-label[身体の動きをアバターへ]],
  )
]

#let lighthouse-products() = image-slide(
  title: [Lighthouse方式の実機],
  description: [トラッカーが赤外線を受光し、部屋に固定したベースステーションを基準にする。],
  credit: image-credit("I03", "I04"),
)[
  #grid(columns: (1fr, 1fr), gutter: 1.5em, align: horizon,
    [#photo("I03") #photo-label[VIVE Tracker (3.0)]],
    [#photo("I04") #photo-label[SteamVR Base Station 2.0]],
  )
]

#let lighthouse-principle() = image-slide(
  title: [受光時刻と掃引角の関係：Valveの解説図],
  description: [Lighthouse 1.0の原理図。2.0では同期・信号方式が異なる。],
  credit: [#image-credit("I05") #text(size: 11pt, fill: colors.muted)[原資料PDF p.8の図部分を抜粋。]],
)[
  #grid(columns: (1fr, 1.15fr), gutter: 1.1em, align: horizon,
    photo("I05", height: 12em),
    [
      1. 同期フラッシュで基準時刻を知る
      2. 回転レーザーが受光素子を横切る
      3. 時間差から掃引角を求める
    ],
  )
]

#let camera-product() = image-slide(
  title: [カメラを搭載した身体用トラッカー],
  description: [VIVE Ultimate Tracker。2つのカメラで周囲を観測するインサイドアウト方式。],
  credit: image-credit("I06"),
)[
  #align(center)[#photo("I06", height: 12em)]
]

#let imu-products() = image-slide(
  title: [IMU方式の製品例：SlimeVR],
  description: [部位の姿勢を測るセンサーと通信回路を、小型の装着デバイスにまとめる。],
  credit: image-credit("I07", "I08"),
)[
  #grid(columns: (1fr, 1fr), gutter: 1em, align: horizon,
    [#photo("I07") #photo-label[本体とストラップ]],
    [#photo("I08") #photo-label[筐体内部の基板と電池]],
  )
]

#let magnetic-products() = image-slide(
  title: [磁場方式の製品例：FluxPose],
  description: [ビーコンが作る磁場を観測し、ビーコンに対する位置・姿勢を推定する。],
  credit: image-credit("I09", "I10"),
)[
  #grid(columns: (1fr, 1fr), gutter: 1.5em, align: horizon,
    [#photo("I09") #photo-label[Beacon：磁場の信号源]],
    [#photo("I10") #photo-label[Tracker：身体に装着]],
  )
]
