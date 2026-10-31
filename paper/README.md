# Preprint draft

`main.tex` is the preprint. Tables in `tables/` are generated from `../results/` by
`make_tables.py`, so every number in the paper comes from the saved results.

```bash
python3 make_tables.py
latexmk -pdf main.tex
```

To edit on Overleaf: upload this folder (main.tex, refs.bib, tables/, figures/).
