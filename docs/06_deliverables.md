# 06 · Deliverables & Grading Map

Everything we owe Marrero. Mapped to the grading rubric so we don't lose points on form.

---

## Grading scheme reminder (from syllabus, p.6)

| Component | Weight |
|---|---|
| Term Project Proposal | 5% |
| Term Project Check-ins | 5% |
| Term Project Final Presentation | 15% |
| **Term Project Report** | **20%** |
| Term Project Peer Review | 5% |

Term project total: **50% of course grade.**

## Submission timeline

| Week | Deliverable | Format | Status |
|---|---|---|---|
| Wk 2 | Team formed on Canvas | Canvas action | ✓ Done |
| Wk 4 | Proposal (1-page, Canvas) | PDF/DOCX | ✓ Submitted |
| Wks 3, 5, 7, 9 | Oral check-in (2–3 min) | Live | Ongoing |
| Wk 10 | Final report (Canvas) | PDF | **In progress** |
| Wk 10 | Final presentation (last class) | Slides PDF | Drafting |
| Wk 10 | Peer review (after submission) | PDF | TBD |

---

## Deliverable 1, Final Report (20%)

**Length:** 8–10 pages excluding references. 11pt, letter, single column, single space, 1-inch margins.

**Required sections** (from `Report and Peer Review Guidelines.pdf`):

1. **Introduction** (incl. goal + deliverables)
2. **Methods** (incl. description + reasoning)
3. **Results** (incl. description + potential reasons)
4. **Conclusions** (incl. implications, significance, extensions)

**Title** must include type: "**A New Application of POMDP to Regime-Switching Asset Allocation**" (or similar with "New Application").

**Contributions table** required at end (not counted toward page limit). See template at end of this file.

### Rubric (from `Report Grading Rubric.pdf`)

Deductions to avoid:

| Rubric item | Deduction | How we avoid |
|---|---|---|
| Unmet length requirements | −0.25 | Aim for 9 pages of body + references |
| Unclear goals | −0.5 | Explicit "Goal" subsection in §1 |
| Unclear justification for assumptions | −1 | §3 lists every assumption + why |
| Unclear justification for modeling approach | −1 | "Why POMDP" subsection in §3 |
| Unclear motivation for parameters | −1 | Every parameter ($\lambda$, $\gamma$, cost) has a one-line justification |
| Incorrect technical claim | −0.5 | Cross-check by running the code; cite class results |
| Figures/tables not referenced in text | −0.25 | Every figure has a numbered ref in prose |
| Illegible text in plots | −0.25 | Use minimum 10pt font in matplotlib |
| Figures with display errors | −0.25 | PDF export, no rasterization at low DPI |
| Typos in proofs | −0.25 | Proofread by all 4 team members |
| Incorrect results | −1 | Sanity checks, multiple solvers cross-checked |
| No results | −2 | Cannot happen, we have backtest output |
| Missing implications discussion | −0.5 | Explicit "Implications" paragraph in §4 |
| Missing extensions discussion | −0.5 | Explicit "Future work" paragraph in §4 |

**Max recoverable deductions if we are sloppy: −8.25 of 20 points (~41% of report grade).** Don't.

### Report outline (proposed)

```
1. Introduction (1 page)
   1.1 Motivation
   1.2 Goal
   1.3 Deliverables

2. Background (1 page)
   2.1 Regime-switching in finance (Hamilton, Ang-Bekaert, etc.)
   2.2 POMDP framework and QMDP approximation

3. Methods (3-4 pages)
   3.1 Why POMDP (not DT, ID, or MDP)
   3.2 POMDP tuple
   3.3 HMM calibration via Baum-Welch
   3.4 Underlying MDP and value iteration
   3.5 QMDP and belief filter
   3.6 Backtest protocol

4. Results (2-3 pages)
   4.1 HMM model selection
   4.2 Regime interpretation
   4.3 Backtest (headline figure: equity curves)
   4.4 Sensitivity analysis

5. Conclusions (1 page)
   5.1 What we learned
   5.2 Implications
   5.3 Extensions

References
Appendix: Contributions table
```

## Deliverable 2, Presentation (15%)

8–10 minutes + 1–2 minutes Q&A. Each team member required to present.

### Slide outline (10 slides target)

1. Title / team / project type
2. The problem: 60/40 lost 17% in 2022 because of regime ignorance
3. Why POMDP (one-line comparison table)
4. POMDP tuple (1 slide, compact)
5. HMM regime calibration (regime timeline figure)
6. QMDP solver (algorithm box + policy map figure)
7. Backtest setup (walk-forward diagram)
8. **Headline figure: equity curves and metrics table**
9. Sensitivity / robustness
10. Conclusions + extensions + thanks

### Per-member speaking allocation

| Slides | Speaker | Time |
|---|---|---|
| 1–2 | Dario (intro) | 1.5 min |
| 3–4 | Taka (POMDP framework) | 2 min |
| 5–6 | Even (HMM + QMDP) | 2.5 min |
| 7–8 | Kyle / KDLL (backtest + results) | 2.5 min |
| 9–10 | All (sensitivity, conclude) | 1.5 min |

## Deliverable 3, Peer Review (5%)

One page, sections:
- Summary and contributions
- Project strengths
- Project weaknesses + suggested point deductions

We get assigned another team after submission. Read their report carefully; do not lowball but be honest.

---

## Contributions Table (template, fill in at submission time)

| Task | Dario | Even | Kyle (KDLL) | Taka |
|---|---|---|---|---|
| Defined the problem | 25% | 25% | 25% | 25% |
| Collected data | 0% | 0% | 50% | 50% |
| Data analysis | 30% | 20% | 30% | 20% |
| Built uncertainty model (HMM) | **70%** | 0% | 20% | 10% |
| Built decision model (POMDP) | 10% | 30% | 10% | **50%** |
| Developed solution technique (QMDP / VI) | 10% | **60%** | 20% | 10% |
| Drafted initial report | 20% | 20% | 20% | **40%** |
| Revised report | 25% | 25% | 25% | 25% |
| Drafted initial slides | 25% | 25% | 25% | 25% |
| Revised slides | 25% | 25% | 25% | 25% |

(Numbers are first-draft estimates. Adjust based on actual work.)

---

## Canvas submission checklist

- [ ] Report (PDF) uploaded to Canvas before deadline.
- [ ] Slides uploaded ahead of presentation class time (Wk 10, last class).
- [ ] Peer review submitted within window after release of other teams' reports.
- [ ] Contribution table at end of report (not counted toward page limit).
- [ ] Title explicitly says "New Application".
