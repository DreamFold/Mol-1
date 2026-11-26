#!/usr/bin/env python3
"""Rebalance the design/folding mix of a prepared ``indices.csv``.

LOCAL ADDITION -- not part of upstream Mol-1.

Why this exists
---------------
``prepare_design_data.py`` writes, for each input structure, a plain folding
copy alongside its design variants. That is deliberate -- co-training on
folding is what keeps a design fine-tune from destroying the structural prior
the model was chosen for -- but the resulting mix is far heavier on folding
than the recipe being copied.

Measured on ``design_data_mix``: of the 847 indexed structures, **315 (37.2%)
contain no designed residue at all**. BoltzGen's antibody task table
(``boltzgen/data/const.py``, ``training_task_probabilities_with_antibodies``)
gives ``select_none`` -- its pure-folding task -- a probability of **0.025**,
and puts 0.5 on ``select_antibody_loops``. So the prepared set spends roughly
fifteen times as many steps on folding as the recipe calls for, on top of the
diffusion loss already being spread across every atom.

Rebalancing is a filter on ``indices.csv``, not a rebuild: the bioassemblies
for the design structures already exist and are untouched. That matters --
the bioassembly build is the expensive step and is only partially complete.

What it does NOT do
-------------------
It does not change *which* residues are designed in a structure, or add
variants. Raising the design rate past what the prepared CIFs contain is not
possible here; rerun ``prepare_design_data.py`` for that.

Usage
-----
    python scripts/filter_design_indices.py \
        --indices  $MOL_ROOT_DIR/design_data_mix/indices.csv \
        --cif-dir  $MOL_ROOT_DIR/design_data_mix/cifs \
        --output   $MOL_ROOT_DIR/design_data_mix/indices_design.csv \
        --folding-rate 0.025

Then point training at the new file:
    --data.<set>.base_info.indices_fpath .../indices_design.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("filter_design_indices")

#: The residue code a designed residue takes. A structure with none of these
#: is a folding example.
DESIGN_RESIDUE = "UNK"


def find_cif(cif_dir: Path, pdb_id: str) -> Path | None:
    """Case-tolerant lookup; the prepared CIFs are upper-case, the index lower."""
    for candidate in (pdb_id.upper(), pdb_id, pdb_id.lower()):
        path = cif_dir / f"{candidate}.cif"
        if path.is_file():
            return path
    return None


def has_designed_residues(path: Path) -> bool:
    """Does this CIF contain at least one designed residue?

    A substring test over the whole file rather than a parse: this runs over
    thousands of files and only needs a yes/no. It can only err towards
    *keeping* a structure (``UNK`` appearing in a comment or an unrelated
    field), never towards dropping a real design example, which is the safe
    direction for a filter whose job is to preserve design data.
    """
    with path.open("r", errors="replace") as handle:
        return any(DESIGN_RESIDUE in line for line in handle)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indices", required=True, type=Path)
    ap.add_argument("--cif-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument(
        "--folding-rate", type=float, default=0.025,
        help="Target fraction of rows that are pure-folding examples. The "
             "default matches BoltzGen's select_none probability (0.025). "
             "Pass 0 to drop folding entirely -- not recommended, it removes "
             "the co-training that protects the folding prior.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report the composition and write nothing.")
    args = ap.parse_args(argv)

    if not 0.0 <= args.folding_rate < 1.0:
        ap.error("--folding-rate must be in [0, 1)")

    with args.indices.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not rows:
        log.error("%s is empty", args.indices)
        return 1

    # One filesystem read per structure, not per row: indices.csv holds a row
    # per chain/interface and there are ~18 of them per structure.
    verdict: dict[str, bool] = {}
    missing = set()
    for pdb_id in {r["pdb_id"] for r in rows}:
        path = find_cif(args.cif_dir, pdb_id)
        if path is None:
            missing.add(pdb_id)
            verdict[pdb_id] = True  # keep: absence of a CIF is not evidence
            continue
        verdict[pdb_id] = has_designed_residues(path)
    if missing:
        log.warning("%d structure(s) had no CIF and were kept: %s",
                    len(missing), ", ".join(sorted(missing)[:5]))

    design_rows = [r for r in rows if verdict[r["pdb_id"]]]
    folding_rows = [r for r in rows if not verdict[r["pdb_id"]]]

    # Keep f such that f / (len(design) + f) == rate.
    rate = args.folding_rate
    keep_n = 0 if rate == 0 else min(
        len(folding_rows), round(rate * len(design_rows) / (1.0 - rate))
    )
    rng = random.Random(args.seed)
    # Sample whole *structures*, not rows: keeping 3 of a structure's 18 rows
    # would silently bias which chains and interfaces that structure is ever
    # seen through.
    folding_ids = sorted({r["pdb_id"] for r in folding_rows})
    rng.shuffle(folding_ids)
    kept_ids: set[str] = set()
    kept = 0
    for pdb_id in folding_ids:
        n = sum(1 for r in folding_rows if r["pdb_id"] == pdb_id)
        if kept + n > keep_n:
            continue
        kept_ids.add(pdb_id)
        kept += n
    out_rows = design_rows + [r for r in folding_rows if r["pdb_id"] in kept_ids]

    def pct(part, whole):
        return f"{part / whole:.1%}" if whole else "n/a"

    n_struct = len({r["pdb_id"] for r in rows})
    log.info("input   : %d rows over %d structures", len(rows), n_struct)
    log.info("  design: %d rows (%s)", len(design_rows), pct(len(design_rows), len(rows)))
    log.info("  folding: %d rows (%s) over %d structures",
             len(folding_rows), pct(len(folding_rows), len(rows)), len(folding_ids))
    log.info("target folding rate: %.1f%%", rate * 100)
    log.info("output  : %d rows -- %d design + %d folding (%s) over %d structures",
             len(out_rows), len(design_rows), kept, pct(kept, len(out_rows)),
             len({r["pdb_id"] for r in out_rows}))

    if args.dry_run:
        log.info("--dry-run: nothing written")
        return 0

    # Preserve the input row order so the file stays diffable against the
    # original, and so any order-dependent downstream step behaves the same.
    keep = {id(r) for r in out_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(r for r in rows if id(r) in keep)
    log.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
