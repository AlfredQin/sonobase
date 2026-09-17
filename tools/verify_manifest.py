#!/usr/bin/env python3
"""Verify a local SonoCorpus copy against the Zenodo manifest (sonocorpus_manifest.csv).

Usage: python tools/verify_manifest.py sonocorpus_manifest.csv --dataset-dir $DATASET_DIR [--datasets BUSI HC18]
Checks that every converted image and annotation file listed for the chosen datasets exists under
<dataset-dir>/<converted_relpath> and that its MD5 matches converted_md5 / converted_mask_md5.
Reports missing and mismatched files per dataset; exit code 1 if any."""
import argparse, csv, hashlib, os, sys
from collections import Counter

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""): h.update(c)
    return h.hexdigest()

ap = argparse.ArgumentParser(); ap.add_argument("manifest"); ap.add_argument("--dataset-dir", required=True); ap.add_argument("--datasets", nargs="*")
a = ap.parse_args(); stats = {}
with open(a.manifest, newline="") as fh:
    for r in csv.DictReader(fh):
        ds = r["dataset"]
        if a.datasets and ds not in a.datasets: continue
        st = stats.setdefault(ds, Counter())
        for rel, want in ((r["converted_relpath"], r["converted_md5"]), (r["converted_mask_relpath"], r["converted_mask_md5"])):
            if not rel: continue
            p = os.path.join(a.dataset_dir, rel); st["checked"] += 1
            if not os.path.exists(p): st["missing"] += 1; print("MISSING", rel)
            elif md5(p) != want: st["mismatch"] += 1; print("MISMATCH", rel)
bad = 0
for ds, st in sorted(stats.items()):
    print(f"{ds:24s} checked {st['checked']:7d}  missing {st['missing']:6d}  mismatch {st['mismatch']:6d}"); bad += st["missing"] + st["mismatch"]
sys.exit(1 if bad else 0)
