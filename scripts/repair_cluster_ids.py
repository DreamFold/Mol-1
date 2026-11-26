#!/usr/bin/env python3
"""Fill in cluster ids that a design variant's id prevented from resolving.

LOCAL ADDITION -- a repair for indices built before the parser learned to
strip the variant suffix. New builds do not need it.

The problem -- two independent causes
-------------------------------------
1. ``prepare_training_data.py --cluster_file`` **defaults to None**, and if it
   is not passed the parser builds an EMPTY cluster dict
   (``mol/data/core/parser.py``) and records *every* protein chain as
   ``NotInClusterTxt``. This is the dominant cause and it hits plain and
   variant ids alike.
2. Even with the file passed, a design variant's id (``1xct_v1``) is not a
   real PDB entry, so its key misses. The parser now strips the suffix;
   indices built before that still carry the misses.

Why it matters: ``calc_weights_for_df`` computes ``cluster_size`` by grouping
on ``cluster_id`` and weights each row by ``1 / cluster_size``. Every miss
therefore lands in ONE enormous cluster. Measured on ``design_data_v2``:
**28.2% of rows received 0.26% of the sampling probability -- a 150x
suppression.** Those examples are not merely under-sampled, they are
effectively absent from training.

Why a repair rather than a rebuild: ``cluster_id`` is baked into the
``*.idx.pkl.gz`` sidecars at parse time, so fixing the parser alone means
re-parsing every structure. The index CSV is what training reads, and the
cluster id is recoverable from the base PDB id plus the entity id that the
CSV already carries -- so the repair is exact, not an approximation.

    python scripts/repair_cluster_ids.py --indices .../indices_subset.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mol.data.utils import parse_pdb_cluster_file_to_dict  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("repair_cluster_ids")

MISS = "NotInClusterTxt"
VARIANT = re.compile(r"_v\d+$")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indices", required=True, type=Path)
    ap.add_argument("--cluster-file", type=Path,
                    default=Path("/mnt/shared_storage/.cache/mol1/common/"
                                 "clusters-by-entity-40.txt"))
    ap.add_argument("--output", type=Path,
                    help="Defaults to overwriting --indices in place.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    with args.indices.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        rows = list(reader)
    if not rows:
        log.error("%s is empty", args.indices)
        return 1

    clusters = parse_pdb_cluster_file_to_dict(str(args.cluster_file))
    log.info("%d cluster entries", len(clusters))

    fixed = collisions = still_missing = 0
    for row in rows:
        # Repair plain and variant ids alike: cause 1 above means a plain
        # id can be a miss too, so restricting this to variants would leave
        # most of the damage in place.
        base = VARIANT.sub("", row["pdb_id"])

        # Repair each side independently, then recompute the combined key the
        # same way the parser does: "a:b" for an interface, bare for a chain.
        new = {}
        for side in ("1", "2"):
            col = f"cluster_{side}_id"
            ent = (row.get(f"entity_{side}_id") or "").strip()
            if row.get(col) != MISS or not ent:
                continue
            hit = clusters.get(f"{base}_{ent}")
            if hit:
                new[col] = hit[0]
            else:
                still_missing += 1
        if not new:
            continue
        row.update(new)

        c1, c2 = row.get("cluster_1_id", ""), row.get("cluster_2_id", "")
        combined = f"{c1}:{c2}" if c2 else c1
        if row.get("cluster_id") != combined:
            row["cluster_id"] = combined
            collisions += 1
        fixed += 1

    n_miss_after = sum(1 for r in rows if r.get("cluster_id") == MISS)
    log.info("repaired %d row(s); %d combined ids rewritten; "
             "%d side(s) still unresolved", fixed, collisions, still_missing)
    log.info("rows still NotInClusterTxt: %d / %d", n_miss_after, len(rows))

    if args.dry_run:
        log.info("--dry-run: nothing written")
        return 0
    out = args.output or args.indices
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
