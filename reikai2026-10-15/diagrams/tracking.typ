// 編集可能な原理図。長さ・軌跡・波形は説明用の模式図であり、実測値ではない。
// 本編では原寸で全幅表示。色に加え、ラベル・線種・形でも意味を区別する。
#import "@preview/cetz:0.4.2": canvas, draw
#import "../../templates/kmc.typ": colors
#let orange = colors.primary
#let navy = colors.secondary
#let gray = colors.muted
#let blue = colors.blue
#let label(body, size: 17pt, fill: navy) = text(size: size, fill: fill, body)
#let dashed(paint) = (paint: paint, thickness: 1pt, dash: "dashed")
#let faint = rgb("#eaf0f6")
#let bounds(w, h) = draw.rect((0, 0), (w, h), stroke: none)
#let arrow(a, b, paint: orange, width: 1.7pt) = draw.line(a, b, stroke: paint + width, mark: (end: ">"))
#let dot(p, paint: orange, radius: 0.13) = draw.circle(p, radius: radius, fill: paint, stroke: white + 0.6pt)

#let headset(x, y, paint: navy) = {
  draw.rect((x - 0.8, y - 0.37), (x + 0.8, y + 0.37), radius: 0.14,
    stroke: paint + 1.4pt, fill: colors.paper)
  draw.rect((x - 0.6, y - 0.22), (x + 0.6, y + 0.22), radius: 0.06, stroke: none, fill: paint)
  draw.line((x - 0.8, y + 0.2), (x - 1, y + 0.08), (x - 1, y - 0.15),
    (x - 0.8, y - 0.23), stroke: paint + 1.2pt)
}

#let pose() = canvas(length: 1cm, {
  import draw: *
  bounds(24, 7)
  content((4.7, 6.6), label([位置：3方向への移動], size: 20pt))
  content((17, 6.6), label([姿勢：3軸まわりの回転], size: 20pt))
  line((10, 0.4), (10, 6.8), stroke: colors.rule + 0.8pt)
  line((2.1, 3), (7.8, 3), mark: (start: ">", end: ">"), stroke: orange + 1.8pt)
  line((4.6, 1.1), (4.6, 5.3), mark: (start: ">", end: ">"), stroke: blue + 1.8pt)
  line((2.5, 1.5), (6.5, 4.5), mark: (start: ">", end: ">"), stroke: gray + 1.8pt)
  headset(4.6, 3)
  content((8.2, 3), label[左右])
  content((4.6, 5.75), label[上下])
  content((1.9, 1.1), label[前後])
  content((4.7, 0.25), label([$bold(p) = (x, y, z)$], size: 20pt))
  // yawは上面、pitchは側面、rollは正面から見た回転。
  for (x, name, view, paint) in (
    (12.8, [ヨー], [左右を向く\ （上面）], orange),
    (17.2, [ピッチ], [うなずく\ （側面）], blue),
    (21.6, [ロール], [首をかしげる\ （正面）], navy),
  ) {
    content((x, 5.35), label(name, size: 20pt, fill: paint))
    circle((x, 3.35), radius: 0.68, fill: white, stroke: paint + 1.3pt)
    if x == 12.8 {
      rect((x - 0.67, 3.65), (x + 0.67, 4.02), radius: 0.08, fill: paint, stroke: none)
      arrow((x, 4.05), (x, 4.45), paint: paint)
    } else if x == 17.2 {
      rect((x + 0.48, 3.18), (x + 0.88, 3.62), radius: 0.05, fill: paint, stroke: none)
      line((x + 0.68, 3.12), (x + 0.83, 2.96), (x + 0.59, 2.96), stroke: paint + 1pt)
    } else { headset(x, 3.35, paint: paint) }
    arc((x, 3.35), anchor: "origin", start: -45deg, stop: 210deg, radius: 1.18,
      stroke: paint + 1.7pt, mark: (end: ">"))
    content((x, 1.1), label(view, size: 17pt))
  }
})

// 一定骨長の2リンク脚。両端の違いを膝の位置で吸収する模式図。
#let leg(hip, foot, side, paint: navy, uncertain: false) = {
  let dx = foot.at(0) - hip.at(0)
  let dy = foot.at(1) - hip.at(1)
  let d = calc.sqrt(dx * dx + dy * dy)
  let h = calc.sqrt(1.45 * 1.45 - d * d / 4)
  let knee = ((hip.at(0) + foot.at(0)) / 2 + side * (-dy / d) * h,
              (hip.at(1) + foot.at(1)) / 2 + side * (dx / d) * h)
  draw.line(hip, knee, foot, stroke: if uncertain { dashed(paint) } else { paint + 2pt })
  draw.circle(knee, radius: 0.10, fill: white, stroke: paint + 1pt)
}

#let skeleton() = canvas(length: 1cm, {
  import draw: *
  bounds(24, 7.2)
  for (x, extra) in ((5.2, false), (18.5, true)) {
    content((x, 6.8), label(if extra { [腰・足にも観測を追加] } else { [頭と手だけを観測] }, size: 20pt))
    circle((x, 5.6), radius: 0.38, stroke: navy + 1.5pt)
    line((x, 5.2), (x, 3.05), stroke: navy + 2pt)
    line((x - 0.65, 4.65), (x + 0.65, 4.65), stroke: navy + 2pt)
    line((x - 0.65, 4.65), (x - 1.25, 4.05), (x - 1.9, 4.2), stroke: navy + 2pt)
    line((x + 0.65, 4.65), (x + 1.25, 4.05), (x + 1.9, 4.2), stroke: navy + 2pt)
    line((x - 0.35, 3.05), (x + 0.35, 3.05), stroke: navy + 2pt)
    for side in (-1, 1) {
      let hip = (x + side * 0.35, 3.05)
      let foot = (x + side * 1.65, 0.95)
      if not extra { leg(hip, (x + side * 0.65, 0.35), side, paint: gray, uncertain: true) }
      leg(hip, foot, side, paint: if extra { navy } else { gray }, uncertain: not extra)
      if extra { dot(foot, paint: blue) }
    }
    for p in ((x, 5.6), (x - 1.9, 4.2), (x + 1.9, 4.2)) { dot(p) }
    if extra { dot((x, 3.05), paint: blue) }
    content((x, -0.05), label(if extra { [観測を満たす身体モデルを求める] } else { [脚の候補が複数残る（破線）] }, size: 17pt))
  }
  arrow((9, 3.6), (14.4, 3.6), paint: blue)
  content((11.7, 4.25), label[追加の手がかり])
  content((11.7, 2.6), label([#text(fill: orange)[● 頭・手]\ #text(fill: blue)[● 腰・足]], size: 17pt))
})

#let lighthouse() = canvas(length: 1cm, {
  import draw: *
  bounds(24, 7.2)
  content((5.8, 6.85), label([① 光が横切った時刻を測る], size: 20pt))
  content((18.8, 6.85), label([② 既知の素子配置に当てはめる], size: 20pt))
  rect((0.5, 2.5), (1.6, 4.2), radius: 0.13, fill: navy, stroke: none)
  circle((1.05, 3.75), radius: 0.16, stroke: gray + 1pt)
  content((1.2, 1.8), label([固定局], size: 18pt))
  // 一つの掃引を2D断面で描く。受光時刻を飛行時間による距離と取り違えない。
  line((1.65, 3.4), (9.5, 1.15), (9.5, 5.65), close: true, fill: colors.pale, stroke: none)
  rect((8, 1.35), (10.3, 5.5), radius: 0.2, stroke: navy + 1.7pt, fill: white)
  for (p, t) in (((8.45, 1.95), [$t_1$]), ((9.7, 3.35), [$t_2$]), ((8.55, 4.85), [$t_3$])) {
    line((1.65, 3.4), p, stroke: dashed(orange))
    dot(p)
    content((11, p.at(1)), label(t, size: 19pt))
  }
  arc((1.65, 3.4), anchor: "origin", start: -16deg, stop: 15deg, radius: 4,
    stroke: orange + 1.7pt, mark: (end: ">"))
  content((4.6, 5.55), label([レーザーが掃引], size: 17pt, fill: orange))
  content((6.4, 0.35), label([橙の点：トラッカー上の受光素子], size: 17pt))
  arrow((12.2, 3.4), (13.8, 3.4))
  content((18.9, 5.2), label([$t_i$ → 掃引角 $theta_i$], size: 22pt))
  content((18.9, 4), label([局から各素子への方向が分かる], size: 18pt))
  line((15, 3.3), (22.8, 3.3), stroke: colors.rule + 1pt)
  content((18.9, 2.3), label([素子間の位置関係は既知], size: 19pt))
  content((18.9, 0.9), label([方向と整合する位置・姿勢を推定], size: 18pt, fill: orange))
})

#let cameras() = canvas(length: 1cm, {
  import draw: *
  bounds(24, 6.6)
  for (x, inside) in ((0, true), (12.7, false)) {
    content((x + 5.5, 6.2), label(if inside { [インサイドアウト] } else { [アウトサイドイン] }, size: 20pt))
    rect((x + 0.8, 1.35), (x + 10.2, 5.35), stroke: colors.rule + 1pt)
    if inside {
      line((x + 3.5, 3.2), (x + 9.6, 1.7), (x + 9.6, 5), close: true, stroke: none, fill: colors.pale)
      headset(x + 2.7, 3.2)
      rect((x + 3.35, 2.98), (x + 3.7, 3.42), fill: orange, stroke: none)
      for p in ((x + 8.8, 2), (x + 9.6, 4.65), (x + 7.8, 3.9)) {
        line((x + 3.7, 3.2), p, stroke: dashed(orange))
        rect((p.at(0) - 0.1, p.at(1) - 0.1), (p.at(0) + 0.1, p.at(1) + 0.1), fill: orange, stroke: none)
      }
      content((x + 2.7, 2.1), label[機器のカメラ])
      content((x + 7.9, 0.65), label[部屋の特徴点])
    } else {
      line((x + 1.7, 4.55), (x + 7.7, 4.6), (x + 7.7, 1.7), close: true, stroke: none, fill: faint)
      rect((x + 0.95, 4.3), (x + 1.8, 4.8), fill: blue, stroke: none)
      line((x + 1.4, 4.3), (x + 1.4, 3.5), stroke: blue + 1pt)
      circle((x + 6.8, 4.1), radius: 0.35, stroke: navy + 1.6pt)
      line((x + 6.8, 3.75), (x + 6.8, 2.65), (x + 6.1, 1.7), stroke: navy + 1.7pt)
      line((x + 6.8, 2.65), (x + 7.5, 1.7), stroke: navy + 1.7pt)
      line((x + 6.1, 3.3), (x + 6.8, 3.6), (x + 7.6, 3.3), stroke: navy + 1.7pt)
      for p in ((x + 6.8, 4.1), (x + 6.1, 3.3), (x + 7.5, 1.7)) {
        line((x + 1.8, 4.55), p, stroke: dashed(blue))
        dot(p, paint: blue, radius: 0.11)
      }
      content((x + 2.8, 2.35), label[部屋のカメラ])
      content((x + 6.8, 0.65), label[身体・マーカー])
    }
  }
})

#let architecture() = canvas(length: 1cm, {
  import draw: *
  bounds(24, 7.8)
  for (x, title) in ((3.4, [部屋：磁場を作る]), (11.7, [身体：時刻付きで測る]), (20.3, [PC：位置・姿勢を求める])) {
    content((x, 7.4), label(title, size: 19pt))
  }
  // A/B/Cは独立に駆動する例。実際の部屋形状やコイル寸法を表さない。
  for (x, y, letter, paint) in ((0.8, 4.6, [A], orange), (2.5, 4.95, [B], blue), (4.2, 4.6, [C], navy)) {
    rect((x, y), (x + 1.5, y + 1.55), radius: 0.06, stroke: paint + 2pt)
    content((x + 0.75, y + 0.78), label(letter, size: 18pt, fill: paint))
    line((x + 0.75, y), (x + 0.75, 3.65), stroke: paint + 1pt)
  }
  rect((0.8, 2.1), (5.7, 3.65), stroke: navy + 1pt)
  content((3.25, 2.87), label([電流を反転\ 実電流を測定], size: 17pt))
  arrow((6.3, 5.1), (8.6, 5.1))
  content((7.45, 5.8), label([磁場], size: 18pt, fill: orange))
  rect((8.9, 4.3), (14.5, 6.4), stroke: navy + 1pt)
  content((11.7, 5.35), label([磁気：3軸の磁場\ IMU：比力・角速度], size: 17pt))
  arrow((11.7, 4.2), (11.7, 3.7), paint: blue)
  rect((8.9, 2.1), (14.5, 3.65), stroke: navy + 1pt)
  content((11.7, 2.87), label([無線マイコン\ 値と時刻を送信], size: 17pt))
  arrow((14.7, 2.9), (16.8, 2.9), paint: blue)
  content((15.75, 3.6), label([無線], size: 17pt, fill: blue))
  rect((17.1, 2.1), (23.5, 6.4), stroke: navy + 1pt)
  content((20.3, 5.6), label([物理モデルと実測地図], size: 17pt))
  content((20.3, 4.65), label([磁場とIMUを融合], size: 18pt, fill: orange))
  arrow((20.3, 4.1), (20.3, 3.6), paint: blue)
  content((20.3, 2.9), label([HMDと座標合わせ], size: 17pt))
  arrow((20.3, 2), (20.3, 1.2), paint: blue)
  content((20.3, 0.65), label([VRへ位置・姿勢を出力], size: 18pt))
  line((3.25, 2), (3.25, 0.8), (11.7, 0.8), (11.7, 2), stroke: dashed(gray))
  content((7.5, 0.2), label([制御時刻と測定時刻を同期], size: 16pt, fill: gray))
})

#let field-gradient() = canvas(length: 1cm, {
  import draw: *
  bounds(24, 7)
  for (x, slope, paint, title) in (
    (0.6, 0.12, blue, [変化が小さい場所]), (13, 0.65, orange, [変化が大きい場所]),
  ) {
    content((x + 5, 6.6), label(title, size: 20pt, fill: paint))
    let cx = x + 5
    let half = 0.34 / slope
    // 同じ磁場の不確かさ±σ_Bを、位置軸へ写した局所近似。
    rect((x + 1.1, 3.16), (x + 9, 3.84), fill: faint, stroke: none)
    rect((cx - half, 1.1), (cx + half, 1.48), fill: paint.lighten(65%), stroke: none)
    line((x + 1.1, 1.1), (x + 9.3, 1.1), stroke: gray + 1pt, mark: (end: ">"))
    line((x + 1.1, 1.1), (x + 1.1, 5.9), stroke: gray + 1pt, mark: (end: ">"))
    line((x + 1.5, 3.5 - 3.5 * slope), (x + 8.5, 3.5 + 3.5 * slope), stroke: paint + 2pt)
    for (px, py) in ((cx - half, 3.16), (cx + half, 3.84)) {
      line((px, 1.5), (px, py), stroke: dashed(paint))
    }
    dot((cx, 3.5), paint: paint)
    content((x + 0.2, 5.5), label([$B$], size: 20pt))
    content((x + 9.4, 0.6), label([$p$], size: 20pt))
    content((cx, 0.45), label(if slope < 0.2 { [位置の候補が広がる] } else { [位置の候補を絞れる] }, size: 17pt))
    content((x + 5, 5.8), label([同じ磁場ノイズ幅 $2 sigma_B$], size: 16pt, fill: gray))
  }
})

#let reversal() = canvas(length: 1cm, {
  import draw: *
  bounds(24, 7.1)
  content((8.3, 6.75), label([同じ位置・向きで測る理想的な1軸の例], size: 18pt))
  let x0 = 2.8
  let step = 2.8
  for j in range(4) {
    let x = x0 + j * step
    line((x, 0.9), (x, 6.1), stroke: (paint: colors.rule, thickness: 0.7pt, dash: "dotted"))
    content((x + step / 2, 6), label(([$+I$], [$0$], [$-I$], [$0$]).at(j), size: 20pt))
    content((x + step / 2, 0.55), label(([32 µT], [30 µT], [28 µT], [30 µT]).at(j), size: 17pt))
  }
  content((1, 4.8), label[電流])
  content((1, 2.3), label[観測値])
  line((x0, 4.6), (14.7, 4.6), mark: (end: ">"), stroke: gray + 0.8pt)
  line((x0, 5.25), (5.6, 5.25), (5.6, 4.6), (8.4, 4.6), (8.4, 3.95),
    (11.2, 3.95), (11.2, 4.6), (14, 4.6), stroke: orange + 2pt)
  line((x0, 2.25), (14.7, 2.25), stroke: dashed(gray))
  line((x0, 3.1), (5.6, 3.1), (5.6, 2.25), (8.4, 2.25), (8.4, 1.4),
    (11.2, 1.4), (11.2, 2.25), (14, 2.25), stroke: blue + 2pt)
  for (x, y) in ((4.2, 3.1), (7, 2.25), (9.8, 1.4), (12.6, 2.25)) { dot((x, y), paint: blue) }
  content((15, 4.5), label([$t$], size: 19pt))
  line((16, 0.8), (16, 6.1), stroke: colors.rule + 1pt)
  content((20, 5.8), label([差分でコイル成分を抽出], size: 18pt))
  content((20, 4.25), label([$(32 - 28) / 2 = 2$ µT], size: 21pt, fill: orange))
  content((20, 2.9), label([背景30 µTは共通して消える], size: 16pt))
  content((20, 1.65), label([0状態：背景・整定の監視], size: 17pt))
})

#let tdma() = canvas(length: 1cm, {
  import draw: *
  bounds(24, 7.1)
  let x0 = 3
  let step = 1.65
  content((12.5, 6.75), label([3コイル × 4区間 × 50 ms ＝ 1巡600 ms（設計例）], size: 19pt))
  for k in range(3) {
    let y = 5.2 - k * 1.5
    let paint = (orange, blue, navy).at(k)
    content((1.4, y + 0.12), label([コイル #("ABC".at(k))], size: 18pt, fill: paint))
    line((x0, y - 0.38), (22.8, y - 0.38), stroke: colors.rule + 1pt)
    for j in range(4) {
      let x = x0 + (k * 4 + j) * step
      rect((x, y - 0.38), (x + step, y + 0.55), fill: paint.lighten(88%), stroke: paint + 0.8pt)
      rect((x, y - 0.38), (x + 0.27, y + 0.55), fill: rgb("#b8bcc3"), stroke: none)
      content((x + 0.93, y + 0.1), label(([$+I$], [$0$], [$-I$], [$0$]).at(j), size: 17pt))
    }
  }
  for j in range(4) {
    let x = x0 + 4 * j * step
    line((x, 1.35), (x, 5.9), stroke: (paint: colors.rule, thickness: 0.7pt, dash: "dashed"))
    content((x, 0.85), label([#(j * 200) ms], size: 16pt))
  }
  arrow((x0, 1.3), (23.3, 1.3), paint: gray, width: 0.9pt)
  content((12, -0.05), label([灰色：切替後に捨てる区間　／　残り：安定した測定窓], size: 17pt))
})

// 磁場マップとの照合には、観測座標系の向きも含む。
#let field-match() = canvas(length: 1cm, {
  import draw: *
  bounds(24, 5.8)
  for (x, title, body) in (
    (2.6, [候補の状態], [位置 $bold(p)$\ 姿勢 $R$]),
    (9, [磁場マップ], [$bold(B)_k (bold(p))$\ 部屋座標]),
    (15.7, [向きを反映], [$R^top bold(B)_k (bold(p))$\ センサー座標]),
    (22, [実測と比較], [観測 $bold(d)_k$\ との差を計算]),
  ) {
    content((x, 5.25), label(title, size: 18pt))
    content((x, 3.75), label(body, size: 18pt))
  }
  for (a, b) in ((4.4, 6.25), (11.65, 12.95), (18.4, 19.4)) { arrow((a, 3.7), (b, 3.7), paint: blue) }
  line((22, 2.6), (22, 1.6), (2.6, 1.6), (2.6, 2.6), stroke: orange + 1.5pt, mark: (end: ">"))
  content((12, 0.65), label([複数コイルの残差が小さくなるように、位置・姿勢を更新], size: 19pt, fill: orange))
})

#let fusion() = canvas(length: 1cm, {
  import draw: *
  bounds(24, 7.5)
  let past = 6.3
  let now = 21.4
  content((12, 7.1), label([① 到着した磁場観測を、その測定時刻へ結びつける], size: 18pt))
  line((now, 5.25), (now, 6.3), (past, 6.3), (past, 5.25), stroke: orange + 1.5pt, mark: (end: ">"))
  arrow((1.1, 4.9), (23.5, 4.9), paint: gray, width: 1pt)
  for i in range(43) { line((1.5 + i * 0.5, 4.76), (1.5 + i * 0.5, 5.04), stroke: blue + 1pt) }
  dot((past, 4.9), radius: 0.17)
  dot((now, 4.9), radius: 0.17)
  content((past, 4.1), label([測定時刻 $t_m$], size: 18pt))
  content((now, 4.1), label([到着時刻 $t_r$：現在], size: 18pt))
  content((2.4, 5.65), label([IMUで逐次予測], size: 16pt, fill: blue))
  arrow((past, 3.65), (past, 2.5))
  content((2.75, 2.95), label([② 保存した\ 過去状態を補正], size: 17pt))
  arrow((past, 2.2), (now, 2.2), paint: blue, width: 2pt)
  for i in range(9) { circle((past + i * (now - past) / 8, 2.2), radius: 0.11, fill: white, stroke: blue + 1pt) }
  dot((past, 2.2), radius: 0.17)
  dot((now, 2.2), paint: blue, radius: 0.17)
  content((13.6, 3), label([③ 保存済みのIMUで、現在まで再計算], size: 18pt, fill: blue))
  content((21.4, 1.35), label([補正後の現在状態], size: 17pt, fill: blue))
  content((12, 0.25), label([保持する履歴は、観測の古さを覆う長さが必要], size: 18pt))
})
