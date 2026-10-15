#import "../templates/kmc.typ": colors, reference-caption
#let sources = json("sources.json")
#let cite(id) = link(sources.at(id).url)[#id]
#let source-line(..ids) = reference-caption[
  参考：#for (i, id) in ids.pos().enumerate() {
    if i > 0 { [ ／ ] }
    cite(id)
  }
]
#let ref-entry(id, description) = block(above: 0.65em, below: 0.65em)[
  #grid(columns: (2.6em, 1fr), gutter: 0.4em,
    text(fill: colors.primary, weight: "bold", size: 15pt, id),
    [#text(size: 16pt, weight: "bold")[#link(sources.at(id).url)[#sources.at(id).title]]
    #v(0.2em)
    #text(size: 13pt, fill: colors.muted, description)],
  )
]
