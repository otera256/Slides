# VR向けトラッキングシステムの原理

表紙1枚、本編の説明18枚・画像6枚、参考資料7枚、技術補足27枚の計59ページ。説明18枚は outline.md の番号と対応し、画像は各節の持ち時間内で提示します。28分＋質疑2分を想定しています。末尾の技術補足は発表後に読むための資料です。

## ファイル構成

- `main.typ`：表紙と本編。スライド単位で編集します。
- `../templates/kmc.typ`：共通テーマ、配色、表・キャプション等の部品。
- `diagrams/tracking.typ`：CeTZによる原理図10種。位置と姿勢、骨格推定、Lighthouse、カメラ、自作構成、磁場勾配、反転差分、時分割、磁場照合、遅延補正。座標やラベルはここで編集します。
- `diagrams/README.md`：原理図の構成、意味、編集時の注意点。
- `review-notes.md`：全体レビューで見直した説明とレイアウトの記録。
- `product-image-slides.typ`：装着例、Lighthouseの実機とValve公式原理図、VIVE Ultimate Tracker、SlimeVR、FluxPoseの独立した画像スライド6枚。
- `image-sources.json`：画像I01〜I10の掲載元・URL・閲覧日・加工内容・ファイルのハッシュ。
- `image-credits.md`：画像の引用元一覧と説明上の位置付け。
- `reference-slides.typ`：本編の後に続く文献・参考画像のスライド。
- `supplement-slides.typ`：technical-notes.md の説明・数式・評価計画・用語集を収録した補足。
- `sources.json`：文献ID、資料名、URL。R01〜R17は references.md と対応します。
- `sources.typ`：本文中の出典リンクと参考資料の表示関数。
- `speaker-notes.json`：outline.md の口頭補足を転記した発表者ノート。
- `references/`：参照会話のシミュレーション画像4点。`products/` 以下は製品画像と公式解説図の引用素材10点。
- `main.pdf`：投影用PDF。
- `main-notes.pdf`：ノートを主表示にした発表者用PDF。

アウトラインや参考資料の変更は自動同期しません。スライドとノートも必要に応じて更新してください。

## コンパイル

プロジェクトのルート `C:\Users\otera\Slides` で実行します。親フォルダの画像とテーマを参照するため `--root .` を指定します。

```powershell
typst compile --root . reikai2026-10-15/main.typ reikai2026-10-15/main.pdf
typst compile --root . --input notes=true reikai2026-10-15/main.typ reikai2026-10-15/main-notes.pdf
```

編集しながら更新する場合：

```powershell
typst watch --root . reikai2026-10-15/main.typ reikai2026-10-15/main.pdf
```

使用環境：Typst 0.15.0、Touying 0.7.0、CeTZ 0.4.2。フォントは過去のスライドと同じ `UD Digi Kyokasho NP`。テーマの `font` 引数で変更できます。

発表者ノートには口頭補足を入れています。長い説明も収まるよう、ノート本文は16ptにしています（投影用の本文・キャプションは20pt）。参考資料7枚・技術補足27枚は本編28分の配分に含めていません。文献タイトルと本文中のR番号はリンクです。技術補足はノート版でも通常のスライドとして表示します。

画像スライドは共通テーマの `image-slide` を使い、説明スライドから分離しています。説明文・ラベルは本文と同じ20pt、引用元だけ11ptです。画像スライドはノート版にも表示します。画像内のメーカーによる視覚効果は計測結果ではありません。Valveの掃引角の図はLighthouse 1.0の資料であり、2.0の同期・信号方式とは区別しています。
