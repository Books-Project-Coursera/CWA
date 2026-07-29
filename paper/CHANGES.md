# Results section — what was filled in and why

Source of every number: `WACVResults.xlsx`, sheets `Final results` (aggregated
mean ± std per dataset), `Seed_1…Seed_500` (per-seed CIFAR-100 records) and
`Config_cifar`.

## Verification of what was already written

All existing claims were re-derived from the spreadsheet and match, including
the 30-configuration aggregates, the arg-max-$k$ counts (14/11/4/1 for accuracy,
16/10/3/1 for macro-F1), the "25 of 30" and "23 of 30" standard-deviation
counts, and every per-configuration figure quoted in the prose.

One correction: Table 8, macro-F1 mean at $k=3$ was 83.54; the spreadsheet gives
83.5349, and the table's own $\Delta$ of $+1.88$ against a baseline of 81.65
implies 83.53. Changed to 83.53.

## What was added

The spreadsheet holds macro-precision, macro-recall, AUC, per-seed checkpoint
indices, epochs trained and wall-clock time. Adding all of it to the five
per-dataset tables would have tripled their size to restate a trend the
accuracy and macro-F1 rows already show, so it went in where it carries new
information instead:

1. **Table 8** now reports macro-precision, macro-recall and AUC alongside
   accuracy and macro-F1, with $\Delta$ inline. Recall gains more than
   precision (+2.22 vs +1.85 pp), and AUC rises in 23 of the 24 configurations
   where it was recorded — evidence that averaging improves score ranking, not
   only the arg-max decision. AUC is missing for Tiny ImageNet in the
   spreadsheet, hence the 24-configuration footnote.

2. **New Section 5.4 + Table 9 — "Which Checkpoints Does the Criterion
   Select?"** Sections 2 and 3 claim the selected set need not be contiguous or
   confined to late epochs, but nothing in the paper demonstrated it. From the
   per-seed checkpoint indices (CIFAR-100, 6 models × 5 seeds = 150 selected
   checkpoints at $k=5$): mean span 12.0 epochs against 5 for a contiguous
   block; only 5 of 30 runs contiguous, all of them ViT-Base, which early-stops
   at ~14 epochs; only 11.3% of selected checkpoints inside the final five
   epochs, with $S_5$ wholly disjoint from that window in 20 of 30 runs; and
   $S_{k-1}\subset S_k$ in all 30 runs. This makes the contrast with last-$k$
   averaging empirical rather than rhetorical. The Discussion notes that it
   establishes structural difference, not superiority — the two were not
   compared under a matched protocol.

3. **Quantified variance reduction.** Previously "frequently reduces"
   variability with three examples. Now also: mean across-seed std falls 22.3%
   (accuracy) and 23.5% (macro-F1), ranging from −56% on Tiny ImageNet to −8%
   on Tomato Leaf.

4. **Storage cost** in Section 5.4: runs retain 41.3 checkpoints on average,
   reducible to $k+1$ with a running top-$k$ set.

## Training configuration

Not every field in `Config_cifar` was added as a column. Weight decay, dropout,
classifier head, optimizer, warmup fraction, $\eta_{\min}$, label smoothing and
input resolution are constant across configurations (weight decay being the one
exception), so they are stated once in the Section 5.1 prose. Table 2 keeps only
the four quantities that genuinely vary per row: batch, epochs, LR, patience.
Adding four near-constant columns would have overflowed the longtable.

**Open question:** `Config_cifar` is the only config sheet in the workbook, so
the shared settings above are documented for CIFAR-100 only. Confirm they hold
for Tomato, Burmese Grape, Potato and Tiny ImageNet, or the sentence in
Section 5.1 needs qualifying.

## Tiny ImageNet reported as validation metrics

The benchmark releases no labelled test split, so the official validation split
is the held-out evaluation set. Made consistent throughout: Table 1 caption and
a `†` footnote on the row; Section 4.3 rewritten to state the protocol and that
the selection split (10,000 images held out of train) is disjoint from it;
Table 4 caption and both row labels ("Validation Accuracy", "Validation
Macro-F1"); Section 5.1 defining "final evaluation" per dataset; Section 5.2
prose; and the Tiny ImageNet panels in `sensitivity_figures.tex`.

## LaTeX fixes

- `\ref{sec:Discussion}` and `\ref{sec:Conclusion}` did not match the
  lowercase `\label`s, so both rendered as "Section ??" in the PDF.
- A stray `\` on its own line before `\section{Conclusion}`.
- Burmese Grape used `[H]` while the other four results tables used `[htbp]`,
  so Table 5 was typeset before Table 4. Now `[htbp]`; tables 1–9 appear in
  order.
- `sensitivity_figures.tex` used `\addplot ... coords{...}`, which is not
  pgfplots syntax (`coordinates` is) — 60 occurrences, fatal on compile. Also
  removed a partial std-band overlay that existed for VGG16 alone and, via
  `\closedcycle`, filled to the axis rather than between curves; and wrapped
  each `tikzpicture` in `adjustbox` since `clip=false` annotations overhung the
  margin.
- Preamble: added `subcaption` and the six `colorX` definitions the figures
  file needs.

## Sensitivity figures

`\input{sensitivity_figures}` sits commented out in Section 5.3. Enabled it
compiles cleanly and adds 2 pages (13 → 15), but the five figures restate
Tables 3–7 graphically. Uncomment for a version with room, or lift individual
figure blocks out of the file.

## Build

`pdflatex` × 3 + `bibtex`, TeX Live 2023: 0 errors, 0 overfull boxes, no
undefined references or citations, in both the default (13 pp.) and
figures-enabled (15 pp.) configurations.
