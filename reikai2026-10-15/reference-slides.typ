#import "@preview/touying:0.7.0": *
#import themes.metropolis: slide
#import "../templates/kmc.typ": *
#import "sources.typ": ref-entry

#slide(title: [参考資料：まず読む5件])[
  #ref-entry("R02", [姿勢と順運動学による全身推定。IMU式フルトラッキングの入口。])
  #ref-entry("R05", [カメラによる地図・目印の観測と、IMUによる高速な更新。])
  #ref-entry("R10", [慣性航法と、センサー誤差が積分によって蓄積する仕組み。])
  #ref-entry("R16", [磁気センサーのノイズ、出力周期、内部平均の条件。])
  #ref-entry("R13", [既知の磁場による測位と、金属が生む磁場の歪み。])
  #reference-caption[資料名をクリックすると原典へ移動。外部資料の閲覧日：2026-10-15。]
]

#slide(title: [参考資料：VRとLighthouse])[
  #ref-entry("R01", [身体の追加観測点、IKとアバターのキャリブレーション。])
  #ref-entry("R02", [IMUの姿勢から骨格を構成し、HMDの位置を基準にする仕組み。])
  #ref-entry("R03", [自作プログラムからトラッカーの位置・回転を送る公式の入力仕様。])
  #ref-entry("R04", [Lighthouseの受光素子・固定局・ホストの役割。2016年の一次資料。])
  #ref-entry("R09", [受光回路・マイコン・IMUを組み合わせた作者自身の試作報告。])
  #reference-caption[世代や方式を区別して参照する。試作報告を市販機と同等の性能の根拠にはしない。]
]

#slide(title: [参考資料：カメラと磁場によるトラッキング])[
  #ref-entry("R05", [初代Quest / Rift Sのvisual-inertial SLAMとコントローラー観測。])
  #ref-entry("R06", [画像から素手の関節を推定。コントローラー追跡とは別の処理。])
  #ref-entry("R07", [外部カメラと反射マーカーから3次元を再構成する方式の例。])
  #ref-entry("R08", [本体のカメラで自己位置を追う身体用トラッカーの製品例。])
  #ref-entry("R12", [腰のビーコンを基準とする磁場方式と慣性センサーの融合。])
  #reference-caption[メーカーの公称値と独立した比較実験を区別する。価格・出荷状況は本編の比較に用いていない。]
]

#slide(title: [参考資料：電磁気と推定])[
  #ref-entry("R10", [Cambridge技術報告 UCAM-CL-TR-696。慣性航法と誤差伝播。])
  #ref-entry("R11", [導線の形と電流から磁場を計算するBiot–Savart則。])
  #ref-entry("R13", [Appendix B：透磁率と渦電流に起因する歪み。AC方式の資料。])
  #ref-entry("R14", [一定時間分の時刻付き状態を保持し、観測で更新する実装の参考。])
  #ref-entry("R15", [Fisher情報と推定分散の限界。CRLBの意味を理解する資料。])
  #reference-caption[自作案への適用は設計モデル。各資料が本トラッカーの性能を検証したという意味ではない。]
]

#slide(title: [参考資料：部品と設計計画])[
  #ref-entry("R16", [3軸磁気センサー。ODRと平均設定の組合せ、ノイズ条件を確認。])
  #ref-entry("R17", [3軸加速度計＋3軸ジャイロ。センサー出力と内蔵姿勢推定を区別。])
  #ref-entry("P01", [大型空芯コイルの正負切替とIMUドリフト補正を検討した会話。])
  #v(0.5em)
  #text(size: 17pt)[
    設計モデルの導出・評価計画：#link("technical-notes.md")[technical-notes.md]\
    各資料の読みどころと条件：#link("references.md")[references.md]
  ]
  #reference-caption[会話は計画資料として扱う。文献の性能値を、自作案の実績へ転用しない。]
]

#slide(title: [参考画像：シミュレーションのコイル配置])[
  #grid(columns: (0.8fr, 0.8fr, 1.4fr), gutter: 1em, align: horizon,
    align(center, image("references/simulation-coil-upper.png", height: 10.3em)),
    align(center, image("references/simulation-coil-floor.png", height: 10.3em)),
    [*画像に表示された条件*

    #text(size: 17pt)[
      上部：2.8 m角、$z = 0.6$ m、+30 At\
      床側：1.8 m角、$z = -1.1$ m、−50 At
    ]

    #caption[原点は評価空間の中央。正負は巻線・電流方向の符号。画像をそのまま保存したもので、今回の再計算結果ではない。]],
  )
]

#slide(title: [参考画像：部屋スケールの磁場分布])[
  #columns(widths: (1.6fr, 1fr))[
    #image("references/simulation-field.png", width: 100%)
  ][
    評価空間\
    *2.8 × 2.8 × 2.2 m*

    #caption[参照会話の表示画面。\ 合成場の可視化か、独立モードを束ねた観測かは画像だけでは未検証。]

    #caption[79.3%はモデル上の体積率。実機の成功率や、動作中の6DoF精度を示す値ではない。]
  ]
]
