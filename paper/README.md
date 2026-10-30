# Paper (IEEE conference format)

`main.pdf` is the compiled paper. To edit or rebuild it on Overleaf:

1. New Project → Upload Project → upload `overleaf.zip` (or this folder).
2. Menu → Compiler: **XeLaTeX** (needed for the Tamil script; pdfLaTeX will fail).
3. Recompile. Fonts for Tamil are bundled in `fonts/` (Noto Sans Tamil, SIL OFL).

Locally: `latexmk -xelatex main.tex`.

Numbers come from the committed results: `python paper/collect.py` rebuilds `data.json`,
`python paper/make_figures.py` rebuilds `figures/*.pdf`.
