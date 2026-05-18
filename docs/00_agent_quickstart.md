# 00 · Agent Quickstart

**LLM agents and new collaborators: read this first, then `01_project_overview.md`.**

---

## TL;DR

We're building a regime-switching asset allocator framed as a POMDP and solved with QMDP. Code is in Python. Pipeline is data-fetch → HMM-fit → MDP-solve → QMDP-policy → walk-forward backtest. There's a synthetic demo in `experiments/00_synthetic_demo.py` that runs end-to-end with no network needed and produces `figures/synthetic_equity_curve.pdf`.

## What you should NOT do

1. **Do not** rewrite or replace anyone else's homework PDFs in `homework/`. Those are immutable references.
2. **Do not** push to `main` without running the synthetic demo first. It is the smoke test.
3. **Do not** commit the contents of `data/raw/` or `data/processed/*.pkl` — they're in `.gitignore`.
4. **Do not** assume the partner team has done their work yet. The `homework/dario/` and `homework/even/` folders are placeholders; if you need to reference their material, ask them in Slack first.
5. **Do not** modify the proposal docx (`proposal/ENGS177_Term_Project_Proposal.docx`). It is the version that was submitted to Canvas.

## What you SHOULD do

1. Read `docs/01_project_overview.md` for the canonical problem statement.
2. Read `docs/02_class_concepts.md` to see exactly which ENGS 177 lecture formulas live in which file.
3. If you're adding a new experiment, put it in `experiments/NN_*.py` with a numeric prefix in pipeline order, and add a row in `docs/05_experimental_design.md`.
4. If you're adding new code, put reusable modules in `src/<area>/` and import them from experiments. Tests live next to the module as `test_*.py`.
5. Use `numpy`, `pandas`, `matplotlib` only — no deep-learning frameworks. Add deps to `requirements.txt`.

## Common tasks

### "Add a new policy"
Add it to the `run_policy` / `backtest_policy` dispatch in `experiments/00_synthetic_demo.py` and `experiments/05_backtest_compare.py`, plus a one-line entry in `docs/05_experimental_design.md` under "comparator policies."

### "Add a new sensitivity dimension"
Extend the grid in `experiments/06_sensitivity.py` (not yet written; create it). Make sure to update the deliverables table in `docs/06_deliverables.md` so the report mentions the new dimension.

### "Make a figure for the report"
Use `matplotlib`, export to `figures/*.pdf`. Use sane defaults (`figsize=(11, 4)` for time series, `figsize=(6, 6)` for square plots). Caption the figure in the report and reference it in prose to avoid the −0.25 rubric deduction.

### "Add a new dataset"
Update `src/data/fetch_data.py`. Verify the new column lands in `data/processed/monthly.csv` with no NaNs. Document the source in `docs/03_external_research.md` under "Tier 5 — Data references."

## Slack / Canvas signals to watch

- Marrero posts weekly: check `engs177@thayer.dartmouth.edu` and Canvas announcements.
- Project check-in dates: **weeks 3, 5, 7, 9** (oral, 2–3 minutes each).
- Final deadline: **week 10** for report + presentation.

## Where the team docs live

- **Code lives here** — this repo.
- **Slides** — we'll add a `presentation/` directory once we start drafting.
- **Report draft** — overleaf project link in the team Slack.
- **Meeting notes** — team Slack pinned messages.

## Synthetic demo cheat sheet

```bash
cd ENGS177_Final_Project
python experiments/00_synthetic_demo.py
# outputs:
#   figures/synthetic_equity_curve.pdf
#   figures/synthetic_equity_curve.png
#   results/synthetic_metrics.csv
```

The current synthetic-demo headline (commit at time of writing): QMDP **CAGR 12.4% / Sharpe 0.99** vs Static 60/40 **CAGR 8.4% / Sharpe 0.80** over 25 simulated years. This proves the machinery; the live backtest (`experiments/05_backtest_compare.py`) will be the real number when `experiments/01_fetch_data.py` succeeds.
