// KMC共通テーマ。配色は shinkan/Apr24.typ を継承。
#import "@preview/touying:0.7.0": *
#import themes.metropolis: metropolis-theme, slide

#let colors = (
  primary: rgb("#ce6636"), primary-light: rgb("#dc997a"),
  secondary: rgb("#0e0b31"), secondary-lighter: rgb("#2c2a6c"),
  secondary-light: rgb("#4a4780"), secondary-lightest: rgb("#6a66a0"),
  ink: rgb("#26383e"), muted: rgb("#66717a"), paper: rgb("#fafafa"),
  rule: rgb("#dedee6"), pale: rgb("#f7eee8"), blue: rgb("#406c9d"),
)

#let kmc-theme(title: none, subtitle: none, author: [49th furakuta], date: none,
  institution: [KMC], font: "UD Digi Kyokasho NP", notes: false, body) = {
  show: metropolis-theme.with(
    aspect-ratio: "16-9", align: horizon,
    footer: self => self.info.institution,
    config-colors(
      primary: colors.primary, primary-light: colors.primary-light,
      secondary: colors.secondary, secondary-lighter: colors.secondary-lighter,
      secondary-light: colors.secondary-light, secondary-lightest: colors.secondary-lightest,
      neutral-lightest: colors.paper, neutral-dark: colors.ink, neutral-darkest: colors.ink,
    ),
    config-info(title: title, subtitle: subtitle, author: author, date: date, institution: institution),
    config-common(show-only-notes: notes),
  )
  set text(lang: "ja", font: font)
  set par(leading: 0.55em)
  set list(spacing: 0.65em, indent: 1em, body-indent: 0.35em)
  set enum(spacing: 0.6em, indent: 1em)
  set heading(numbering: none)
  show math.equation.where(block: true): it => block(
    width: 100%, fill: colors.pale, inset: 9pt, radius: 2pt,
    align(center, it),
  )
  body
}

#let emphasis(body) = text(fill: colors.primary, weight: "bold", body)
#let caption(body) = block(above: 0.55em, body)
#let reference-caption(body) = block(above: 0.55em, text(size: 11pt, fill: colors.muted, body))
#let takeaway(body) = block(above: 0.7em, text(weight: "bold", body))
#let columns(left, right, widths: (1fr, 1fr), gutter: 1.2em) = grid(
  columns: widths, gutter: gutter, align: horizon, left, right,
)
#let plain-table(..args) = {
  set text(size: 18pt)
  table(stroke: (x: none, y: 0.5pt + colors.rule), inset: (x: 8pt, y: 10pt), ..args)
}

// 画像は独立ページに配置し、既存の説明スライドの組版へ影響させない。
// descriptionは本文サイズ、creditのみ出典用のサイズを指定する。
#let image-slide(title: none, description: none, credit: none, body) = slide(
  title: title,
  align: top,
  config: utils.merge-dicts(
    config-page(margin: (top: 3em, bottom: 1.5em, x: 1em)),
    config-common(show-only-notes: false),
  ),
)[
  #block(width: 100%, height: 12.5em, breakable: false)[
    #align(center + horizon)[#body]
  ]
  #if description != none { block(above: 0.35em, description) }
  #if credit != none { block(above: 0.35em, credit) }
]
