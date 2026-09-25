"""Check every downloaded shard against the Hugging Face listing `fetch_data.sh` saved next to it: size and, for
LFS files, sha256. Writes `$DATA/verify.csv` (path, expected size, local size, sha256 ok, status) and exits 1 if
anything is missing or differs. Hashing 214 GiB takes tens of minutes; run it detached or in a CPU job:

    source .envrc && setsid nohup python scripts/reproduce/verify_data.py > "$DATA/verify.log" 2>&1 &
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

DATA = Path(os.path.expandvars(os.path.expanduser(os.environ["DATA"])))
LISTINGS = {  # listing file -> local directory its paths land in (basename only)
    "hf_tree_splits_mcfm.json": DATA / "splits" / "mcfm",
    "hf_tree_splits_lcfm.json": DATA / "splits" / "lcfm",
    "hf_tree_manifests.json": DATA / "manifests",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    rows, bad = [], 0
    for listing, folder in LISTINGS.items():
        entries = json.load((DATA / listing).open())
        for e in entries:
            local = folder / Path(e["path"]).name
            expected = e.get("size") or (e.get("lfs") or {}).get("size")
            oid = (e.get("lfs") or {}).get("oid")
            if not local.exists():
                status = "missing" if folder.name != "lcfm" or "train" not in local.name else "not downloaded (train)"
                rows.append((e["path"], expected, None, "", status))
                bad += status == "missing"
                continue
            size = local.stat().st_size
            ok_size = expected is None or size == expected
            ok_hash = (sha256(local) == oid) if (oid and ok_size) else None
            status = "ok" if ok_size and ok_hash is not False else ("size differs" if not ok_size else "sha256 differs")
            bad += status != "ok"
            rows.append((e["path"], expected, size, "" if ok_hash is None else ok_hash, status))
            print(f"{status:14s} {e['path']}", flush=True)
    with (DATA / "verify.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "expected_size", "local_size", "sha256_ok", "status"])
        w.writerows(rows)
    print(f"{len(rows)} entries, {bad} not ok -> {DATA / 'verify.csv'}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
