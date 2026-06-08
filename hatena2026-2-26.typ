#import "@preview/touying:0.6.1": *
#import themes.metropolis: *
#import "@preview/numbly:0.1.0": numbly
#import "@preview/algorithmic:1.0.7"
#import algorithmic: algorithm-figure, style-algorithm
#show: style-algorithm
#import "@preview/theorion:0.4.1": *
#import "@preview/codelst:2.0.2": sourcecode
#import "@preview/showybox:2.0.4": showybox

// --- 図解用ライブラリのインポート ---
#import "@preview/fletcher:0.5.8": diagram, node, edge, shapes
#import "@preview/lilaq:0.4.0" as lq

// --- 設定 ---
#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  footer: self => self.info.institution,
  config-info(
    title: [マークダウンでTypstの数式を使いたい],
    author: [49th furakuta],
    date: datetime.today(),
    institution: [KMC],
  )
)

#set text(lang: "ja", font: "UD Digi Kyokasho NP")
#set heading(numbering: numbly("{1}.", default: "1.1"))

// 数式ブロックのスタイル
#show math.equation.where(block: true): it => block(inset: 10pt, radius: 5pt, fill: rgb("#f0f0f0"), width: 100%, it)

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

#show raw.where(block: true, lang: "md"): it => block(inset: 10pt, radius: 5pt, fill: rgb("#f0f0f0"), width: 100%, height: 100%, it) 
#show raw.where(block: true, lang: "html"): it => block(inset: 10pt, radius: 5pt, fill: rgb("#f0f0f0"), width: 100%, height: 100%, it) 


#show raw: set text(size: 14pt)

#title-slide()

= はじめに
== 自己紹介

#columns(2)[
  furakuta
  / 所属: 
    - KMC 49th 副会長
    - 京都大学工学部情報学科1回生
  / 活動分野:
    - 機械学習勉強会
    - 競プロ
  / 趣味・好きなこと:
    - マイクラ
    - フラクタル図形
  / 好きな言語:
    - Rust
  #colbreak()
  #image("images/furakuta_icon.png")
]

== Markdownとは?
// - 2004年3月にJohn Gruberによって作られた軽量マークアップ言語
//   - HTMLよりも簡単に構造化された文章を書くことができる
//   - プレーンテキストで書かれることが多く、様々なツールやプラットフォームでサポートされている
//   - HTMLに変換して表示されることが多いが、プレーンテキストとしても読みやすい
// #pause
// - 元の仕様定義が曖昧であったため多くの方言(flavor)が存在する
//   - 2014年に*CommonMark*という標準仕様が策定された
//   - GitHubもこのCommonMarkをベースにした*GitHub Flavored Markdown (GFM)*という仕様を策定している
//   - サービスの合わせてこのGFMをさらに拡張したMarkdownがサポートされることが多い

// #pagebreak()
#grid(rows: (1em, 17.5em), columns: (1fr, 1fr, 1fr), gutter: 10pt,
  [#align(center)[*Markdown*]],
  [#align(center)[*HTML*]],
  [#align(center)[*出力例*]],
  [#align(top)[```md
# 見出し1
## 見出し2
- 箇条書き
- 箇条書き

**太字**や*斜体*も簡単に書ける
[リンク](https://example.com)も簡単に書ける
![画像](https://example.com/image.png)
  ```]],
  [#align(top)[```html
<h1 id="見出し1">見出し1 </h1>
<h2 id="見出し2">見出し2 </h2>
<ul>
<li>箇条書き</li>
<li>箇条書き</li>
</ul>
<p><strong>太字</strong>や<em>斜体</em>も簡単に書ける<br>
<a href="https://example.com">リンク</a>も簡単に書ける<br>
<img src="https://example.com/image.png" alt="画像"></p>
  ```]],
  [#block(inset: 10pt, radius: 5pt, fill: rgb("#f0f0f0"), width: 100%, height: 100%)[#image("images/example_markdown.png")]]
)

== Markdownの利用例
#grid(rows: (1fr, 1fr), columns: (1fr, 1fr), gutter: 25pt,
  showybox(title-style: (boxed-style: (x: left, y: center)), frame: (title-color: purple.darken(50%), body-color: purple.lighten(90%), border-color: purple.darken(70%)), title: "ノートを取る")[
    #image("images/obsidian.png", width: 100%, height: auto)
  ],
  showybox(title-style: (boxed-style: (x: left, y: center)), frame: (title-color: aqua.darken(50%), body-color: aqua.lighten(90%), border-color: aqua.darken(70%)), title: "ブログを書く")[
    #image("images/hatena_blog.png", width: 100%, height: auto)
  ],
  showybox(title-style: (boxed-style: (x: left, y: center)), frame: (title-color: black.darken(50%), body-color: black.lighten(90%), border-color: black.darken(70%)), title: "AIの出力")[
    #image("images/gemini.png", width: 100%, height: auto)
  ],
  showybox(title-style: (boxed-style: (x: left, y: center)), frame: (title-color: orange.darken(50%), body-color: orange.lighten(90%), border-color: orange.darken(70%)), title: "README")[
    #image("images/github.png", width: 100%, height: auto)
  ]
)

== Typstとは?
LaTeXの複雑さを解消するために開発された、高速でモダンな組版システム

/ 高速: 
  - Rustで実装されており、LaTeXよりも高速にコンパイルできる
  - プレビューも高速で、編集とプレビューのサイクルが短い
#pause
/ モダンな機能:
  - Markdownに近い直感的な文法で、複雑なレイアウトも簡単に作成できる
  - 数式や図の描画も強力で、科学技術文書の作成に適している
#pause
/ 柔軟なカスタマイズ:
  - テーマやスタイルの定義が簡単で、独自のデザインを作成できる
  - ちょっとしたスクリプト言語機能があるので、条件分岐やループも可能
  - プラグインやライブラリも充実しており、様々な用途に対応可能
#pause
*ちなみにこのスライドもTypstで作っています！*

== Markdownの数式サポート
- *MathJax*や*KaTeX*などのJavaScriptライブラリを使うことで、Markdown内で*LaTeX形式*の数式をサポートすることができる
  - 数式に対応する部分をこれらのライブラリで構文解析して`<span>`や`<div>`タグに変換しCSSでスタイリングして表示するというかなり強引な方法で実装されている
#pause
- 例えば、GitHubのREADME.mdでは、数式を `$...$` で囲むことでインライン数式を、```` ```math ... ``` ````で囲むことでブロック数式を記述できる
  - 公式仕様ではないので仕様が環境により異なることに注意
#pause
*Markdownで数式を書くときにLaTeXの文法を使うことを強制されるのが嫌だなというのが今回の動機*

== そんなにLaTeXの数式が嫌いなの?

#table(
  columns: (1fr, 1fr, 1fr),
  align: center,
  [*LaTeX*], [数式の例], [*Typst*],
  [```tex 
  \frac{1}{\sqrt{2\pi\sigma^2}} 
  \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
  ```], [$ 1 / sqrt(2 pi sigma^2) exp(- (x - mu)^2 / (2 sigma^2)) $], [```typst 1 / sqrt(2 pi sigma^2) exp(- (x - mu)^2 / (2 sigma^2)) ```],
  [```tex 
\begin{pmatrix}
  a & b \\
  c & d
\end{pmatrix} 
  ```], [$ mat(a, b; c, d) $], [```typst mat(a, b; c, d) ```],
  [```tex 
f(x) = \begin{cases}
  x & \text{if } x \ge 0 \\
  0 & \text{otherwise}
\end{cases}
  ```], [$ f(x) = cases(
  x "if" x >= 0,
  0 "otherwise"
) $], [```typst
f(x) = cases(
  x "if" x >= 0,
  0 "otherwise"
) 
```],
)

#pause
*特に私はTypstの数式に慣れてしまったので、LaTeXの数式を書くのが億劫になってしまった*

== 今回したいこと
#align(center)[
  #text(size: 28pt)[*MarkdownでTypstの数式を使いたい*]
]
#v(20pt)
#pause
/ 前提として:
  - 実はTypstそのものをHTMLにもコンパイルできるようにしようという動きは前からあり、まだ数式には対応していないが活発に開発が進んでいる
  #pause
  - Typst全体ではなく数式だけを変換できる、Typst版MathJaxのような*Wypst*(https://github.com/andredalbosco/wypst)というプロジェクトも存在する
  #pause
  - ただ今回は既存のMarkdown環境でTypstの数式を使えるようにすることが目的なので、JavaScriptを使う方法は避けたい

= 本題

== 何をしたか

#align(center)[
  多くのプラットフォームではJaveScriptが使えない

  （セキュリティ上などの理由）

  #pause
  $arrow.b$

  *でも画像をURLとして組み込むことはできる*
]
#pause
つまりこういうふうにして

#showybox()[
  #text(size: 20pt)[
  ![typst math](https://typst-math-api.com/render?expr=\<ここに数式\>)
  ]
]
APIサーバーから数式のSVG画像を生成してもらい、それをMarkdownの画像として組み込むことで、Markdown内でTypstの数式を使えるようにすることができるのではないかと考えた

== 使用した技術

/ typst(-as-lib):
  - Typstのコンパイラ自体がRustのOSSであり、typst-as-libというTypstをライブラリとして使うためのラッパーも存在する
  - コードをRustで完結させることで実装をシンプルに保てる

/ axum:
  - RustのWebフレームワークの一つで、シンプルで高速なAPIサーバーを構築するのに適している

== 実装の流れ
1. `template.typ`というTypstのテンプレートファイルを用意する
  - 数式を埋め込むためのプレースホルダを用意しておく
#sourcecode(```typ
#import sys: inputs
#set page(width: auto, height: auto, margin: 5pt)
#set text(size: 14pt)
#eval(inputs.expr, mode: "math")// 数式を描画（外部から注入できるようにしておく）
```)
#pause
2. クエリのパラメータやTypstのテンプレートに読み込むための型を定義する
#sourcecode(```rust
#[derive(Deserialize, IntoValue, IntoDict)] // 色々な便利トレイトを自動で実装
struct MathQuery { expr: String }
impl From<MathQuery> for Dict {
    fn from(value: MathQuery) -> Self {
        value.into_dict()
    }
}
```)
#pagebreak()
3. API用のハンドラー関数を定義する
  - Typstのテンプレートを読み込む（本当はキャッシュしておくべき）
  - クエリの数式をテンプレートに注入しつつTypstのコンパイルをする
  - SVG画像として出力する
  - HTTPレスポンスとして返す
#sourcecode(```rust
static FONT_DATA: &[u8] = include_bytes!("./fonts/NewCMMath-Regular.otf");
static TEMPLATE: &str = include_str!("./template.typ");
async fn render_math_svg(Query(query): Query<MathQuery>) -> impl IntoResponse {
    let template = TypstEngine::builder()
        .main_file(TEMPLATE)
        .fonts([FONT_DATA])
        .build();
    let doc: PagedDocument = template
        .compile_with_input(query)
        .output
        .expect("typst::compile() returned error!");
    let svg = svg(&doc.pages[0]);
    ( [(header::CONTENT_TYPE, "image/svg+xml")], svg )
}
```)
#pause
4. APIサーバーを起動する
#sourcecode(```rust
#[tokio::main]
async fn main() {
    let app = Router::new().route("/render", get(render_math_svg));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```)

== 動作検証1
ローカルでAPIサーバーを起動して、VSCodeのMarkDown Preview Enhanced拡張によるMarkdownプレビューで動作を確認してみる
#align(center)[
  #image("images/test_local.png", width: 100%)
  #pause
  #text(size: 30pt)[
    *ちゃんと表示された！*
  ]
]

== 動作検証2
GitHubのREADME.mdで動作を確認してみる
#pause
#caution-box[
  GitHubの画像読み込みではCamoというプロキシサーバーが介在するため、ローカルで起動したAPIサーバーのURLを直接指定しても画像が表示されない
  
  今回は動作検証時だけngrokを使ってローカルサーバーをインターネットに公開し、そのURLを画像のURLとして指定することで回避した
]
#pagebreak()
#align(center)[
  #image("images/test_github.png", height: 70%)
  #text(size: 30pt)[
    *GitHubのREADME.mdでもちゃんと表示された！*
  ]
]

== オチ
この数式に対応するMarkdownは
#sourcecode(```md
![typst math](https://squishier-julieta-uncalamitously.ngrok-free.dev/render?expr=(x,y)in{RR^2%20%7C%20x%5E2%20%2B%20y%5E2%20%3C%201})
```)
とかなり冗長

#pause
当たり前だがURLエンコードが必要なため人間には読めない上、１行に収めなければならないという制約もあるため、数式が複雑になるとさらに読みにくくなる

== 結論
#align(center)[
  #text(size: 28pt)[*素直にLaTeXの数式に変換して貼り付けよう*]
]