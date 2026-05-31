# Homework Directory

Per-team-member homework artifacts. Everyone drops their own HW1/HW2/HW3 here in their named subfolder. **These files are immutable**: do not edit anyone else's HW.

## Layout

```
homework/
├── README.md                 ← you are here
├── taka/                     Taka Khoo
│   ├── Taka_HW1.tex/pdf
│   ├── Taka_HW2.tex/pdf
│   ├── Taka_HW3.tex/pdf
│   └── solve_hw3.py          Computational support for HW3 inventory + MDP problems
├── kdll/                     Kyle David Ledda-Lewaren
│   ├── KDLL_HW2.pdf
│   └── KDLL_HW3.pdf
├── dario/                    Dario Blanco Morales  (placeholder, drop in)
└── even/                     Even Hogberget        (placeholder, drop in)
```

## How to add yours (dario, even, or updates)

1. `cd` into your named folder.
2. Drop in `YourName_HWn.pdf` (and `.tex` source if you wrote it in LaTeX).
3. Commit with `git add homework/<your_name>/ && git commit -m "homework: add <name> HWn"`.
4. Push.

## How the project draws on each member's HW

| Member | HW2 contribution | HW3 contribution |
|---|---|---|
| Taka | POMDP formulation, why-not-MDP argument | Calibration update, model-based vs model-free defense |
| Kyle (KDLL) | (see PDF) | (see PDF) |
| Dario | TBD | TBD |
| Even | TBD | TBD |

(Update this table once everyone's drops are in.)

## Pointers to the project-relevant HW sections

- **Taka HW2 §5**, full POMDP tuple, QMDP formulation, comparator policies.
- **Taka HW3 P4**, progress update: HMM calibration done, QMDP implemented, preliminary backtest results.
- **Taka HW3 P1**, inventory DP (not project, but reusable VI code).
- **Taka HW3 P2**, 4-state stay/leave MDP (modified policy iteration + policy iteration, directly relevant to our MDP solver).
