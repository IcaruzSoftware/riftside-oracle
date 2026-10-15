"""
Download every Oracle's Elixir year file (2014 onward) into src/data/raw/oe/.

About 860 MB in total, a few minutes. Files younger than oe.MAX_AGE_HOURS (20 h)
are skipped, so a re-run only fetches what is stale; --force re-downloads them
anyway (it never deletes anything).

CLI
---
  python src/scripts/bootstrap_data.py
  python src/scripts/bootstrap_data.py --force
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import oe


def main(force: bool = False) -> None:
    files = oe.list_files()
    for year in sorted(files):
        t0 = time.monotonic()
        path = oe.download_year(year, file_id=files[year], force=force)
        print(f"{year}  {path.stat().st_size / 1e6:7.1f} MB  {time.monotonic() - t0:6.1f}s  {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Download all Oracle's Elixir year files")
    parser.add_argument("--force", action="store_true",
                        help=f"Re-download files younger than {oe.MAX_AGE_HOURS} h too")
    main(force=parser.parse_args().force)
