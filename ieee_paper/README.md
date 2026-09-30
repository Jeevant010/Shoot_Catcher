# IEEE paper folder

This folder is a new write-up. The rest of the project is unchanged.

| File | What it is |
|---|---|
| `PAPER.md` | Full paper in Markdown. Read this first. |
| `main.tex` | Same paper in IEEE conference format. |
| `RESULTS_TABLES.md` | Every number used in the paper, with the report it came from. |
| `figures/` | Plots drawn from those numbers. |
| `generate_figures.py` | Regenerates the plots. |

From this folder:

```bash
python3 generate_figures.py
pdflatex main.tex
```

The introduction is only the problem, the site, and what this paper does. Cleaning, models, metrics, and numbers are in the later sections.

The one-bit alert is in the conclusion. The notes called that alert a binary neural network. The paper says what was actually built: a normal network that outputs 1 or stays silent. Weight-binarized networks were not trained. Voronoi clustering is not in this draft.
