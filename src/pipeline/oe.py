"""
Oracle's Elixir client: the yearly LoL esports match-data CSVs (2014 onward),
one file per year in a public Google Drive folder, refreshed once a day.

Endpoints
---------
  folder view  https://drive.google.com/embeddedfolderview?id={FOLDER_ID}
  download     https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t

File IDs are resolved from the folder view on every run (a new year means a new
ID). The download endpoint needs no auth and supports Range requests.
"""

import re
import time

import requests

FOLDER_ID = "1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"
FOLDER_URL = f"https://drive.google.com/embeddedfolderview?id={FOLDER_ID}"
DOWNLOAD_URL = "https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"

_TIMEOUT = 30

# One folder entry: the file id, then (without crossing into the next entry) its title.
_ENTRY_RE = re.compile(
    r'/file/d/([\w-]+)/(?:(?!/file/d/).)*?'
    r'flip-entry-title">(\d{4})_LoL_esports_match_data_from_OraclesElixir\.csv<',
    re.DOTALL,
)


def parse_folder(html: str) -> dict[int, str]:
    """Year -> Drive file id from the embedded folder view HTML."""
    return {int(year): file_id for file_id, year in _ENTRY_RE.findall(html)}


def list_files() -> dict[int, str]:
    """Fetch the folder view and return year -> file id. Raises if nothing matched."""
    resp = requests.get(FOLDER_URL, timeout=_TIMEOUT)
    resp.raise_for_status()
    files = parse_folder(resp.text)
    if not files:
        raise RuntimeError(f"OE folder {FOLDER_ID}: no year files found (page shape changed?)")
    return files


# ---------------------------------------------------------------------------
# Connectivity check (CLI)
# ---------------------------------------------------------------------------

def check() -> int:
    """
    List the folder, then fetch the first 20 KB of the latest year's CSV and print
    status / size / latency. Returns 0 if both work and the body is the OE CSV, else 1.
    """
    t0 = time.monotonic()
    try:
        files = list_files()
    except (requests.exceptions.RequestException, RuntimeError) as exc:
        print(f"folder    {type(exc).__name__}: {exc}  {time.monotonic() - t0:5.2f}s")
        print("FAILED - could not list the OE folder.")
        return 1
    print(f"folder    HTTP 200  {len(files)} files ({min(files)}-{max(files)})  "
          f"{time.monotonic() - t0:5.2f}s")

    year = max(files)
    t0 = time.monotonic()
    try:
        resp = requests.get(DOWNLOAD_URL.format(file_id=files[year]),
                            headers={"Range": "bytes=0-20479"}, timeout=_TIMEOUT)
    except requests.exceptions.RequestException as exc:
        print(f"download  {type(exc).__name__}: {exc}  {time.monotonic() - t0:5.2f}s")
        print(f"FAILED - could not download the {year} file.")
        return 1
    total = resp.headers.get("Content-Range", "?").rpartition("/")[2]
    print(f"download  {year} HTTP {resp.status_code}  {len(resp.content):,} B  "
          f"{time.monotonic() - t0:5.2f}s  total {total} B  "
          f"last-modified {resp.headers.get('Last-Modified', '?')}")

    ok = resp.status_code in (200, 206) and resp.content.startswith(b"gameid,")
    print(f"OK - OE folder and {year} download work." if ok
          else f"FAILED - {year} download did not return the OE CSV.")
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Oracle's Elixir client")
    parser.add_argument("--check", action="store_true",
                        help="List the OE folder and range-fetch the latest year's CSV")
    args = parser.parse_args()
    if args.check:
        sys.exit(check())
    parser.print_help()
