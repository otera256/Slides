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
    title: [Gitの知っておくと便利な機能],
    author: [49th furakuta],
    date: [2026-06-08],
    institution: [#image("./images/KMClogo_trans.png", height: 2em)],
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

= はじめに
== 自己紹介
#slide(composer: (2fr, 1fr))[
  - kmc_id: furakuta
  - 所属: 京都大学工学部情報学科数理工学コース2回生
  - KMC49代副会長
  - KMCでの活動分野
    - 機械学習: 応用よりも基礎理論中心
    - ゲーム開発: みんゲーや個人でちょっと作ってる
    - 競技プログラミング: 週1でABCに出ているくらいだけど最近青パフォが取れるようになってきた
][
  #align(center)[
    #image("./images/furakuta.png", height: 10em)
  ]
]

== Gitとは
#slide(composer: (2fr, 1fr))[
  #text(1.3em)[
    *分散型バージョン管理システム*
  ]

  / 分散型:\
    各ユーザーが完全なリポジトリのコピーを持ち、中央サーバーに依存しない構造

  / バージョン管理:\
    ファイルの変更履歴を記録・管理するシステム
][
  #image("images/Git-Logo-2Color.svg")
]



まあKMCの人なら大体知っているはずだし、新入生の方も#link("https://docs.google.com/presentation/d/1Edcwnx9eiZT0ZwfQ8_AUQDM6OGJsTAVUMiikdqv0lmE/edit?slide=id.g3d1fc4899e7_0_398#slide=id.g3d1fc4899e7_0_398")[#text(fill: blue)[taitaiくんの講座]]で基本的な概念や使い方は学んでるはず。

#pause
#align(center)[
  #text(1.2em)[
    でも、
    #pause

    `git pull` `git add` `git commit` `git push`

    みたいな基本的なコマンドを打っているだけで
    
    こんなことにはなっていませんか?
  ]
]

#image-slide(
  align(center)[
    #text(1.2em)[
      「軽微」とかいう謎のコミットが大量にあって何があったのか全くわからない
    ]
    #image("images/many_micro_commits.png", height: 80%)
  ]
)

#image-slide(
  align(center)[
    #text(1.2em)[
      `dev` ブランチへの直接コミットが多すぎて、\
      どのコミットが何の機能追加やバグ修正に対応しているのか分かりづらい
    ]
    #image("images/single_branch.png", height: 75%)
  ]
)

#image-slide(
  align(center)[
    #text(1.5em)[
      コミット数が馬鹿みたいに多くなる
    ]
    #image("images/num_commit.png", height: 50%)
  ]
)

私もGitにそこまで詳しくないですが、流石にこれよりはうまくやる方法がわかってきたので紹介してゆきます。

= Gitの仕組み
== よくある勘違い
*Gitはファイルの"変更履歴"を管理している*

```
$ git diff
diff --git a/reikai2026-6-8.typ b/reikai2026-6-8.typ
index adbff08..8299fb0 100644
--- a/reikai2026-6-8.typ
+++ b/reikai2026-6-8.typ
@@ -17,8 +17,8 @@
   config-info(
     title: [Gitの知っておくと便利な機能],
     author: [49th furakuta],
-    date: [2026-01-12],
-    institution: [KMC],
+    date: [2026-06-08],^M
+    institution: [#image("./images/KMClogo_trans.png", height: 2em)],^M
   )
 )
```

こんな感じで `git diff` を見るとファイルの変更が見れるし、GitHubでもcommitをクリックすると差分が見れるからGitは差分を管理している??

#pause
実はGitはファイルの変更履歴を管理しているわけではなく、ファイルのスナップショットを管理している。

/ スナップショット:\
  ある時点でのファイルの状態を丸ごと保存したもの。変更点だけでなく、ファイル全体の内容を保存する

#pagebreak()
実際には
#align(center)[
  ファイルに変更が加わりStageされる\
  ↓\
  差分ではなく新しいスナップショットが作成される\
  ↓\
  新しいコミットではこのスナップショットへの参照が保存される
]
#pagebreak()

もっというとそれぞれのコミットが参照の木という形式でコードベース全体を保存しているが、それぞれのファイルに対して変更が加えられない限り同じスナップショットを参照する。

なので愚直にプロジェクト全体をバックアップするようなバージョン管理システムが少し賢くなって変更のないファイルのスナップショットを使い回すようになったと思ったほうが良いかもしれない。

#pause

`git diff` とかもこの保存されたスナップショット同士の差分を取っているだけで、Gitが変更点を管理しているわけではない。

逆にだからこそ大昔のコミットとの比較も高速にできるし、変更点だけでなくファイル全体の状態をいつでも復元できる。

#pagebreak()
*branch はプロジェクトをコピーかなにかしている？*\
ブランチはただのコミットへの参照
#align(center)[
  #image("images/git_branch.png", height: 72%)
]
#align(right)[
  #text(size: 0.6em, fill: luma(70))[
    [Qiita] たぶんもう怖くないGit ~Git内部の仕組み~ https://qiita.com/marchin_1989/items/2ec01553e907f3a9e6bb より引用
  ]
]

`git checkout` をすると一気にファイルが切り替わるけど、コミットへの参照を切り替えているだけ。

さっき言ったように一つのコミットはプロジェクト全体を保存（参照）しているので、ブランチを切り替えるとプロジェクト全体の状態が切り替わる。

どんどんブランチを切っていこう！！

== 実はコミットは後からも変えられる
コミットは一度作ったら変更できないと思っていませんか？
#pause
実はコミットは後からも変更できるし、ブランチも後から変更できる。

`git commit --amend` を使うと最後のコミットを変更できる

#align(center)[
  #text(1.2em)[
    *Unityのシーンファイル保存忘れで新しいコミットをするな*
  ]
]

`git rebase -i` を使うと過去のコミットも変更できる

詳細は #link("https://git-scm.com/book/ja/v2/Git-%e3%81%ae%e3%83%96%e3%83%a9%e3%83%b3%e3%83%81%e6%a9%9f%e8%83%bd-%e3%83%aa%e3%83%99%e3%83%bc%e3%82%b9")[#text(fill: blue)[Gitのブランチ機能 - リベース]] を参照

== コミットしたくないけど変更を保存したいときは
`git stash` を使うと変更を一時的に保存して、作業ツリーをクリーンな状態に戻せる

#align(center)[
  コミットした後に変更を作成\
  ↓\
  `git stash` で変更を一時保存\
  ↓\
  ブランチを切り替えて作業\
  ↓\
  `git stash pop` で変更を復元
]

#align(center)[
  #text(1.2em)[
    *コミットしたくないけど変更を保存したいときは `git stash` を使おう！*
  ]
]