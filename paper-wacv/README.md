# WACV 2027 submission

Supersedes `../paper/` (LNCS format, classification only). Built on the
official WACV 2027 author kit.

```
pdflatex main; bibtex main; pdflatex main; pdflatex main
pdflatex supplementary; pdflatex supplementary
```

Main paper: 9 pages, of which the body ends on page 8 — within the 8-page
limit that excludes references. Supplementary: 2 pages. Both compile with
0 errors, 0 overfull boxes, no undefined references or citations.

## Before submitting

1. **Set the paper ID.** `\def\wacvPaperID{*****}` in `main.tex` and
   `supplementary.tex`, once CMT assigns it.
2. **Pick the track.** Currently `\usepackage[review,algorithms]{wacv}`.
   Switch to `applications` if that fits better — the method is
   pipeline-level and either is arguable.
3. **Fill `\TODO` markers.** Two, both in `sec/4_setup.tex`: the per-model
   configuration table and the shared-settings paragraph. Search for `TODO`.
   The red `\TODO{}` rendering must be gone before submission.
4. **Camera-ready.** Swap to `\usepackage{wacv}` and restore the author block
   in `main.tex`.

## Layout

| File | Contents |
|---|---|
| `main.tex` | document, title, track/ID switches |
| `preamble.tex` | packages plus `\ms`/`\best` table cell macros |
| `sec/0_abstract.tex` … `sec/7_conclusion.tex` | body |
| `sec/tab_acc.tex` | classification accuracy, 30 configs (main paper) |
| `sec/tab_f1.tex` | classification macro-F1, 30 configs (supplementary) |
| `sec/tab_det.tex` | detection + segmentation, 4 configs |
| `refs.bib` | 39 entries, 36 cited |
| `supplementary.tex` | pulls in `sec/tab_f1.tex` |

The three result tables were generated from `WACVResults.xlsx` rather than
typed. All 420 cells were verified against the spreadsheet by re-extracting
the text layer of the compiled PDF and matching every `mean ± std` string.

## Space budget

The 8-page limit is the binding constraint. Two things were done to meet it:

- The per-configuration macro-F1 table moved to the supplementary. Macro-F1 is
  still in the main paper in aggregate (Tab. 4) and the headline numbers are
  quoted in the text. If you would rather keep it in the body, the config
  table is the next-largest block that could move.
- The configuration table is laid out as two side-by-side halves rather than
  34 stacked rows.

Room to reclaim if you need it: drop `mAP@0.75` from the detection table, or
cut the "When it should help" paragraph in `sec/6_discussion.tex`.
