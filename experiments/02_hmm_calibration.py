"""Step 2 of pipeline: fit Gaussian HMM with 2/3/4 states, model select via BIC, pickle the winner."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.models.hmm import select_n_states, standardize_with_train_stats  # noqa: E402

DATA_PROCESSED = REPO / "data" / "processed"
RESULTS = REPO / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

OBS_COLS = ["vix", "term_spread", "hy_oas"]
TRAIN_END = "2014-12-31"   # everything up to 2014 inclusive for training


def main() -> None:
    df = pd.read_csv(DATA_PROCESSED / "monthly.csv", parse_dates=["DATE"]).set_index("DATE")
    train = df.loc[:TRAIN_END]
    test = df.loc[TRAIN_END:]
    print(f"Train: {train.index.min().date()} → {train.index.max().date()}  ({len(train)} obs)")
    print(f"Test:  {test.index.min().date()} → {test.index.max().date()}  ({len(test)} obs)")

    train_obs, full_obs = standardize_with_train_stats(train, df, OBS_COLS)
    test_obs = full_obs[len(train):]

    fits = select_n_states(train_obs, test_obs, candidates=(2, 3, 4))
    rows = [{
        "n_states": f.n_states,
        "train_ll": f.train_ll,
        "test_ll":  f.test_ll,
        "bic":      f.bic,
    } for f in fits]
    pd.DataFrame(rows).to_csv(RESULTS / "hmm_selection.csv", index=False)

    # Persist all models so downstream scripts can read whichever they want
    for f in fits:
        with open(DATA_PROCESSED / f"hmm_{f.n_states}state.pkl", "wb") as fh:
            pickle.dump(f.model, fh)
    print(f"\nWrote {RESULTS / 'hmm_selection.csv'} and {len(fits)} pickled models to {DATA_PROCESSED.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
