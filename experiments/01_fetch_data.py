"""Step 1 of pipeline: fetch FRED + Yahoo, write data/raw/* and data/processed/monthly.csv."""
import sys
from pathlib import Path

# Make `src` importable when running from repo root.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.fetch_data import main  # noqa: E402

if __name__ == "__main__":
    main()
