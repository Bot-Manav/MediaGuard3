"""
Helper: download_datasets

Downloads required CSV datasets into a top-level `datasets/` folder if missing.
Uses only `requests` and standard libraries so it works on Azure App Service.
"""
import os
import sys
import requests
from typing import Dict, Tuple


FILES = {
    "politifact_fake.csv": "https://github.com/Bot-Manav/MediaGuard3/releases/download/csv/politifact_fake.csv",
    "politifact_real.csv": "https://github.com/Bot-Manav/MediaGuard3/releases/download/csv/politifact_real.csv",
    "gossipcop_fake.csv": "https://github.com/Bot-Manav/MediaGuard3/releases/download/csv/gossipcop_fake.csv",
    "gossipcop_real.csv": "https://github.com/Bot-Manav/MediaGuard3/releases/download/csv/gossipcop_real.csv",
}


def _get_project_root() -> str:
    # modules/ is one level below project root
    this_dir = os.path.dirname(__file__)
    root = os.path.abspath(os.path.join(this_dir, os.pardir))
    return root


def ensure_datasets(datasets_dir: str = None) -> Dict[str, str]:
    """Ensure all required CSVs exist locally; download any that are missing.

    Returns a dict mapping filename -> status ("downloaded" | "skipped" | "failed: <err>").
    """
    root = _get_project_root()
    if datasets_dir is None:
        datasets_dir = os.path.join(root, "datasets")

    os.makedirs(datasets_dir, exist_ok=True)

    results = {}
    for fname, url in FILES.items():
        dest_path = os.path.join(datasets_dir, fname)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            msg = f"exists - skipping: {fname}"
            print(msg)
            results[fname] = "skipped"
            continue

        print(f"Downloading {fname} from {url} ...")
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = r.headers.get('content-length')
                if total is not None:
                    total = int(total)

                # write to temp file then rename to avoid partial files
                tmp_path = dest_path + ".part"
                downloaded = 0
                with open(tmp_path, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=8192):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded * 100 // total
                            sys.stdout.write(f"\r{fname}: {pct}% ({downloaded}/{total} bytes)")
                            sys.stdout.flush()

                if total:
                    sys.stdout.write("\n")

                # final sanity check
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                    os.replace(tmp_path, dest_path)
                    print(f"Saved: {dest_path}")
                    results[fname] = "downloaded"
                else:
                    # remove empty tmp
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                    results[fname] = "failed: empty file"

        except Exception as e:
            print(f"Failed to download {fname}: {e}")
            results[fname] = f"failed: {e}"

    return results


if __name__ == "__main__":
    print("Ensuring datasets are present...")
    res = ensure_datasets()
    print("Dataset check complete:")
    for k, v in res.items():
        print(f" - {k}: {v}")
