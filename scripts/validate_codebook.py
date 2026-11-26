#!/usr/bin/env python3
"""How often can the codebook reproduce a sequence from 3D coordinates?

LOCAL ADDITION. Measures the *representation's* fidelity ceiling -- the model
is not involved. Every residue is encoded into the 14-slot layout from its
deposited coordinates and decoded straight back; a mismatch is a bug in the
codebook, not a modelling result.

Three numbers, and reporting only the first would flatter the result:

  recovery   of the residues that CAN be encoded, how many decode correctly.
             Anything below 100% is a defect.
  coverage   what fraction of standard residues can be encoded at all. A
             residue whose codebook atoms are not all resolved in the
             structure has no label and is skipped at data-prep time too, so
             this is the real ceiling on how much of a chain is designable.
  margin     the closest a GENUINE sidechain atom comes to a backbone atom.
             Decode treats anything within 0.5 A of N/O as a parked slot, so
             this is the headroom before a real atom is miscounted.

    python scripts/validate_codebook.py --cif-dir <dir> --limit 300
"""

from __future__ import annotations

import argparse
import collections
import logging
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mol.data.core.design_residue import (  # noqa: E402
    BACKBONE_ATOMS_4,
    N_DESIGN_SLOTS,
    build_codebook,
    decode_residue,
    encode_residue,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("validate_codebook")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cif-dir", type=Path,
                    default=Path("/mnt/shared_storage/datasets/representative_ab/cifs"))
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Decode threshold, to match decode_residue.")
    args = ap.parse_args(argv)

    import gemmi

    codebook = build_codebook()
    files = sorted(args.cif_dir.glob("*.cif"))
    random.Random(args.seed).shuffle(files)
    files = files[: args.limit]
    log.info("%d CIF(s) from %s", len(files), args.cif_dir)

    n_res = n_nonstd = n_skipped = n_enc = n_ok = 0
    wrong = collections.Counter()
    skipped_by_type = collections.Counter()
    seen_by_type = collections.Counter()
    margins: list[float] = []
    bad_files = 0

    for path in files:
        try:
            st = gemmi.read_structure(str(path))
            st.setup_entities()
        except Exception as exc:
            bad_files += 1
            log.debug("%s unreadable: %s", path.name, exc)
            continue

        for chain in st[0]:
            for res in chain:
                n_res += 1
                if res.name not in codebook:
                    n_nonstd += 1
                    continue
                seen_by_type[res.name] += 1
                coords = {a.name: (a.pos.x, a.pos.y, a.pos.z)
                          for a in res if a.element != gemmi.Element("H")}
                try:
                    slots = encode_residue(res.name, coords, codebook)
                except KeyError:
                    # an atom the codebook needs is unresolved -> no label
                    n_skipped += 1
                    skipped_by_type[res.name] += 1
                    continue
                n_enc += 1
                got = decode_residue(slots, codebook, threshold=args.threshold)
                if got == res.name:
                    n_ok += 1
                else:
                    wrong[(res.name, got)] += 1

                # closest genuine sidechain atom to a backbone atom
                real = codebook[res.name][0]
                if len(real) > 4:
                    bb = np.asarray(slots[:4], dtype=float)
                    side = np.asarray(slots[4:len(real)], dtype=float)
                    d = np.linalg.norm(side[:, None, :] - bb[None, :, :], axis=-1)
                    margins.append(float(d.min()))

    print()
    print("=" * 62)
    print(f"files read            : {len(files) - bad_files} "
          f"({bad_files} unreadable)")
    print(f"residues seen         : {n_res:,}")
    print(f"  non-standard        : {n_nonstd:,}  ({n_nonstd/max(n_res,1):.1%})")
    std = n_res - n_nonstd
    print(f"  standard            : {std:,}")
    print(f"    encodable         : {n_enc:,}  ({n_enc/max(std,1):.2%})  <- coverage")
    print(f"    skipped, atoms    : {n_skipped:,}  ({n_skipped/max(std,1):.2%})")
    print()
    print(f"RECOVERY (of encodable) : {n_ok:,}/{n_enc:,} = "
          f"{n_ok/max(n_enc,1):.4%}")
    print(f"end-to-end (of standard): {n_ok:,}/{std:,} = {n_ok/max(std,1):.2%}")
    if margins:
        m = np.asarray(margins)
        print()
        print(f"sidechain-to-backbone margin over {len(m):,} residues:")
        print(f"   min {m.min():.3f} A   p0.1 {np.percentile(m,0.1):.3f}   "
              f"p1 {np.percentile(m,1):.3f}   median {np.median(m):.3f}")
        print(f"   decode threshold {args.threshold} A  -> "
              f"margin {m.min()/args.threshold:.2f}x at the worst residue")
    if wrong:
        print()
        print("MIS-DECODED (should be empty):")
        for (truth, got), n in wrong.most_common(10):
            print(f"   {truth} -> {got}: {n}")
    if skipped_by_type:
        print()
        print("most-skipped residue types (unresolved atoms):")
        for res, n in skipped_by_type.most_common(6):
            print(f"   {res}: {n:,} of {seen_by_type[res]:,} "
                  f"({n/seen_by_type[res]:.1%})")
    print("=" * 62)
    return 0 if not wrong else 1


if __name__ == "__main__":
    sys.exit(main())
