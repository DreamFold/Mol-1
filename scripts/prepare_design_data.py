# Copyright 2026 Molence Bio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LOCAL ADDITION: turn annotated antibody structures into CDR *design* tasks.

Writes a copy of each input mmCIF in which the CDR residues have been rewritten
into the fixed 14-slot designed-residue layout. The native coordinates are kept
-- parked into the slots by the geometric codebook -- so they remain the
training labels, while the inputs become the uniform ``UNK`` component that
carries no residue identity.

The point of doing this at *data preparation* time is that nothing downstream
needs to change: the output is an ordinary mmCIF, so Mol-1's stock
``prepare_training_data.py``, dataset, cropper, featurizer and losses all work
on it unmodified. The only runtime addition is the input masker, which blanks
the MSA over designed columns.

    python scripts/prepare_design_data.py \\
        --cif-dir  /mnt/shared_storage/datasets/representative_ab/cifs \\
        --annotations /mnt/shared_storage/datasets/representative_ab/all.csv \\
        --output-dir /mnt/shared_storage/.cache/mol1/design_cifs \\
        --num-workers 16

Then feed the result to the stock preparation step:

    python scripts/prepare_training_data.py \\
        --input_path <output-dir> --output_indices_csv <data>/indices.csv \\
        --bioassembly_output_dir <data>/bioassembly

Indexing convention, established empirically against the ``*_cdr_sequences``
columns over 150 chains: the ``*_cdr_ranges`` are **1-based inclusive over
``label_seq_id``**, and chains are named by **``label_asym_id``**. Author
numbering disagrees with both on real entries, so neither can be assumed.

A residue is skipped (left native, not designed) when any atom its codebook
entry needs is unresolved: such a residue cannot be supervised, because the
label coordinates for its slots do not exist.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

import gemmi
import numpy as np

import sys
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Task selection is shared by every generative backend, so it lives in
# df-boltzgen's backend-agnostic package rather than here -- one implementation,
# pinned to BoltzGen's ABSelector by tests/test_design_cdr_tasks.py.
_DFNV = Path("/mnt/user_storage/Projects/df-boltzgen")
if _DFNV.is_dir() and str(_DFNV) not in sys.path:
    sys.path.insert(0, str(_DFNV))

from dfnv.design.cdr_tasks import (  # noqa: E402
    LOOP_DESIGN,
    TASKS,
    build_cdr_groups,
    select_task,
)
from mol.data.core import design_residue  # noqa: E402
from mol.data.core.design_residue import (  # noqa: E402
    DESIGN_RESIDUE,
    N_DESIGN_SLOTS,
    build_codebook,
    design_slot_names,
    encode_residue,
)

logger = logging.getLogger("prepare_design_data")


def suffix_for(variant: int) -> str:
    """Variants share a pdb_id, so later ones get a suffix."""
    return "" if variant == 0 else f"_v{variant}"

#: Element per slot, mirroring the padded CCD component. A CIF's ``type_symbol``
#: is two characters, so the virtual slots cannot carry the spare element here;
#: ``Featurizer`` normalises them instead (see normalize_design_atom_elements).
SLOT_ELEMENTS = ("N", "C", "C", "O", "C", "C") + ("C",) * (N_DESIGN_SLOTS - 6)


class DesignStats:
    """Counters, so a silent no-op run is impossible to mistake for success."""

    def __init__(self) -> None:
        self.structures = 0
        self.written = 0
        self.residues_designed = 0
        self.residues_skipped_unresolved = 0
        self.residues_skipped_nonstandard = 0
        # Positions dropped because an entity's chain copies disagreed about
        # whether they could be coded. See `convertible_per_entity`.
        self.residues_dropped_copy_mismatch = 0
        self.chains_missing = 0
        self.failures = 0
        self.fold_examples = 0
        self.fold_duplicates = 0
        self.cif_missing = 0
        self.tasks = {}

    def merge(self, other: "DesignStats") -> None:
        for key, value in vars(other).items():
            if isinstance(value, dict):
                merged = dict(getattr(self, key))
                for k, v in value.items():
                    merged[k] = merged.get(k, 0) + v
                setattr(self, key, merged)
            else:
                setattr(self, key, getattr(self, key) + value)

    def __str__(self) -> str:
        return (
            f"structures={self.structures} written={self.written} "
            f"designed_residues={self.residues_designed} "
            f"skipped_unresolved={self.residues_skipped_unresolved} "
            f"skipped_nonstandard={self.residues_skipped_nonstandard} "
            f"dropped_copy_mismatch={self.residues_dropped_copy_mismatch} "
            f"chains_missing={self.chains_missing} failures={self.failures} "
            f"fold_examples={self.fold_examples} fold_dupes_skipped={self.fold_duplicates} tasks={self.tasks}"
        )


def parse_ranges(raw: str) -> list[tuple[int, int]]:
    """``"[(26, 32), ...]"`` -> inclusive 1-based label_seq ranges."""
    if not raw or not raw.strip():
        return []
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    out = []
    for item in value:
        start, end = int(item[0]), int(item[1])
        if end >= start:  # a chain with no CDR-L2 carries a degenerate range
            out.append((start, end))
    return out


def designed_label_seqs(ranges: Iterable[tuple[int, int]]) -> set[int]:
    return {i for start, end in ranges for i in range(start, end + 1)}


def encoded_slots(residue: gemmi.Residue, codebook, stats: DesignStats):
    """The 14 slot positions for ``residue``, or ``None`` if it cannot be coded.

    A residue is uncodeable when it is not a standard amino acid, or when any
    atom its codebook entry needs is unresolved -- in either case there is no
    label for the designed slots, so converting it would produce a residue the
    model is asked about and cannot be scored on.

    Split out of :func:`rewrite_residue` so convertibility can be decided for
    a whole entity *before* anything is rewritten; see
    :func:`convertible_per_entity`.
    """
    if residue.name not in codebook:
        stats.residues_skipped_nonstandard += 1
        return None

    coords = {}
    for atom in residue:
        if atom.element == gemmi.Element("H"):
            continue
        coords.setdefault(atom.name, (atom.pos.x, atom.pos.y, atom.pos.z))

    try:
        return encode_residue(residue.name, coords, codebook)
    except KeyError:
        stats.residues_skipped_unresolved += 1
        return None


def rewrite_residue(residue: gemmi.Residue, codebook, stats: DesignStats) -> bool:
    """Rewrite one residue in place into the designed 14-slot layout.

    Returns True when the residue was converted. A residue whose codebook atoms
    are not all resolved is left untouched -- it has no usable label.
    """
    slots = encoded_slots(residue, codebook, stats)
    if slots is None:
        return False

    template = residue[0]  # inherit b-factor / occupancy / altloc conventions
    new_atoms = []
    for index, (name, element) in enumerate(zip(design_slot_names(), SLOT_ELEMENTS)):
        atom = gemmi.Atom()
        atom.name = name
        atom.element = gemmi.Element(element)
        atom.pos = gemmi.Position(*(float(v) for v in slots[index]))
        atom.occ = template.occ
        atom.b_iso = template.b_iso
        new_atoms.append(atom)

    residue.name = DESIGN_RESIDUE
    del residue[:]
    for atom in new_atoms:
        residue.add_atom(atom)
    return True


def expand_targets_to_entity_copies(
    structure: gemmi.Structure, targets: dict[str, set[int]]
) -> dict[str, set[int]]:
    """Extend ``targets`` to every subchain sharing an entity with a target.

    Without this the prepared file contradicts itself, and the contradiction
    is invisible until training.

    The CDR annotation names one chain -- say entity 2's subchain ``E``. But
    ``patch_entity_sequences`` has to rewrite the *entity's* canonical
    sequence, because Mol-1's parser trusts ``entity_poly_seq`` over
    ``atom_site``, and an entity has no per-chain sequence to patch. So when
    entity 2 also owns subchain ``B`` -- a second copy of the same light chain,
    which is the norm in a crystal form with several Fabs in the asymmetric
    unit -- ``B`` inherits a sequence saying those positions are ``UNK`` while
    its ``atom_site`` records still hold the native residue.

    Mol-1 then builds ``B``'s residues from the 14-slot ``UNK`` component and
    keeps only the atoms whose names it has slots for: ``N``, ``CA``, ``C``,
    ``O`` and sometimes ``CB``/``CG``. Every virtual slot is unresolved. The
    result is a residue that *looks* designed to the model -- its inputs are
    masked, it is counted in the design loss and in ``design_seq_recovery`` --
    but whose label carries no codeword at all, so there is nothing for the
    model to learn from it.

    Measured on the prepared ``design_data_mix``: 8EB2's designed entities each
    own three subchains, so **two thirds** of its "designed" residues were
    phantoms of this kind; across a 20-batch sample, 5 of 14 design structures
    were affected and their designed-atom supervision rate was 0.38 instead of
    1.00.

    Converting every copy is the fix that keeps the file self-consistent, and
    it is also the right task definition: the copies are the same antibody
    chain, so the same CDRs are being designed in each.
    """
    by_entity: dict[str, list[str]] = {}
    for entity in structure.entities:
        for subchain_id in entity.subchains:
            by_entity.setdefault(entity.name, []).append(subchain_id)

    expanded = {k: set(v) for k, v in targets.items()}
    for subchains in by_entity.values():
        shared: set[int] = set()
        for subchain_id in subchains:
            shared |= targets.get(subchain_id, set())
        if not shared:
            continue
        for subchain_id in subchains:
            expanded.setdefault(subchain_id, set()).update(shared)
    return expanded


def convertible_per_entity(
    structure: gemmi.Structure,
    targets: dict[str, set[int]],
    codebook,
    stats: DesignStats,
) -> dict[str, set[int]]:
    """Narrow ``targets`` to positions convertible in **every** copy.

    A residue is skipped when its codebook atoms are unresolved, and two
    copies of the same chain need not be unresolved in the same places -- a
    sidechain can be ordered in one Fab of the asymmetric unit and disordered
    in the next. Since the entity carries one sequence for all its copies,
    a position converted in only some of them reproduces, in miniature,
    exactly the inconsistency :func:`expand_targets_to_entity_copies` exists
    to remove: the sequence says ``UNK``, one copy's atoms agree and another's
    do not.

    Taking the intersection costs a few residues per structure and makes the
    guarantee absolute: every position the entity sequence calls designed has
    14 coded atoms in every chain. The alternative -- the union -- cannot be
    made consistent at all, because the missing atoms do not exist.

    This pass, not the rewriting pass, is where uncodeable residues are seen
    and therefore where they are counted: the rewriting pass is only ever
    handed positions that already passed here, so leaving the counters to it
    would report zero skips however many there were.
    """
    per_subchain: dict[str, set[int]] = {}
    for chain in structure[0]:
        for subchain in chain.subchains():
            label_id = subchain.subchain_id()
            wanted = targets.get(label_id)
            if not wanted:
                continue
            ok = per_subchain.setdefault(label_id, set())
            for residue in subchain:
                if residue.label_seq in wanted and (
                    encoded_slots(residue, codebook, stats) is not None
                ):
                    ok.add(residue.label_seq)

    narrowed: dict[str, set[int]] = {}
    for entity in structure.entities:
        copies = [s for s in entity.subchains if s in per_subchain]
        if not copies:
            continue
        shared = set.intersection(*(per_subchain[s] for s in copies))
        for subchain_id in copies:
            stats.residues_dropped_copy_mismatch += len(
                per_subchain[subchain_id] - shared
            )
            narrowed[subchain_id] = set(shared)
    # Subchains no entity claims keep their own convertible set.
    for label_id, seqs in per_subchain.items():
        narrowed.setdefault(label_id, seqs)
    return narrowed


def design_structure(
    structure: gemmi.Structure,
    targets: dict[str, set[int]],
    codebook,
    stats: DesignStats,
) -> int:
    """Rewrite every targeted residue. ``targets`` maps label_asym_id -> seqs."""
    # Keep the original for the chains_missing count: that statistic means
    # "an annotated chain was not found in the file", and an entity copy that
    # the annotation never named should not be able to trip it.
    annotated = set(targets)
    expanded = expand_targets_to_entity_copies(structure, targets)
    targets = convertible_per_entity(structure, expanded, codebook, stats)
    converted: dict[str, set[int]] = {}
    seen: set[str] = set()
    for chain in structure[0]:
        for subchain in chain.subchains():
            label_id = subchain.subchain_id()
            if label_id in expanded:
                seen.add(label_id)
            wanted = targets.get(label_id)
            if not wanted:
                continue
            for residue in subchain:
                if residue.label_seq in wanted and rewrite_residue(
                    residue, codebook, stats
                ):
                    converted.setdefault(label_id, set()).add(residue.label_seq)
    stats.chains_missing += len(annotated - seen)
    return converted


def patch_entity_sequences(
    structure: gemmi.Structure, converted: dict[str, set[int]], stats: DesignStats
) -> None:
    """Mark the designed positions in each entity's canonical sequence.

    ``entity_poly_seq`` is the canonical sequence and Mol-1's parser trusts it
    over ``atom_site``. Rewriting only the coordinates leaves the entity still
    claiming the native residue, and the parser then rebuilds that residue and
    discards every designed atom -- which turns the whole example back into an
    ordinary folding sample, silently. ``setup_entities()`` does not recompute
    this, so it has to be patched directly.

    ``label_seq`` is 1-based over ``full_sequence``.
    """
    for entity in structure.entities:
        if entity.polymer_type == gemmi.PolymerType.Unknown:
            continue
        designed = set()
        for subchain_id in entity.subchains:
            designed |= converted.get(subchain_id, set())
        if not designed:
            continue
        sequence = list(entity.full_sequence)
        for label_seq in designed:
            if 1 <= label_seq <= len(sequence):
                sequence[label_seq - 1] = DESIGN_RESIDUE
            else:
                stats.residues_skipped_nonstandard += 1
        entity.full_sequence = sequence


def find_cif(cif_dir: Path, pdb_id: str) -> Optional[Path]:
    """Locate a structure, tolerating PDB-id case.

    A BoltzGen manifest keys records lowercase ("12e8") while a CIF directory
    may be uppercase ("9BT5.cif"), so an exact match silently finds nothing and
    the run reports zero structures.
    """
    for candidate in (pdb_id, pdb_id.upper(), pdb_id.lower()):
        path = cif_dir / f"{candidate}.cif"
        if path.is_file():
            return path
    return None


def entries_from_manifest(path: Path, instance_id: Optional[int]):
    """(pdb_id, CdrGroups) from a BoltzGen manifest.

    The manifest carries strictly more than the CSV: scFv (six loops on one
    chain), explicit antibody/antigen pairing for entries holding several
    antibodies, and the framework extent. Its CDR indices are 0-based
    label_seq, so the conversion is arithmetic -- see dfnv/design/manifest_cdrs.
    """
    import json

    from dfnv.design.manifest_cdrs import antibody_records, cdr_groups_from_record

    with open(path) as handle:
        manifest = json.load(handle)
    out = []
    for record in antibody_records(manifest):
        try:
            groups = cdr_groups_from_record(record, instance_id)
        except IndexError as exc:
            logger.warning("%s", exc)
            continue
        if groups:
            out.append((str(record["id"]), groups))
    return out


def entries_from_csv(rows, chains: str):
    """(pdb_id, CdrGroups) from a representative-antibody CSV."""
    out = []
    for row in rows:
        if row.get("error"):
            continue
        groups = build_groups(row, chains)
        if groups:
            out.append((row["pdb_id"], groups))
    return out


def build_groups(row: dict, chains: str):
    """CDR groups for this antibody, honouring --chains."""
    heavy_id = (row.get("heavy_label_id") or "").strip() or None
    light_id = (row.get("light_label_id") or "").strip() or None
    heavy = parse_ranges(row.get("heavy_cdr_ranges", ""))
    light = parse_ranges(row.get("light_cdr_ranges", ""))
    if chains == "heavy":
        light_id, light = None, []
    elif chains == "light":
        heavy_id, heavy = None, []
    return build_cdr_groups(heavy_id, heavy, light_id, light)


def process_row(
    entry: tuple,
    cif_dir: Path,
    output_dir: Path,
    codebook,
    task_options: Optional[dict] = None,
    variant: int = 0,
    seed: int = 0,
) -> DesignStats:
    stats = DesignStats()
    pdb_id, groups = entry
    source = find_cif(cif_dir, pdb_id)
    if source is None:
        stats.cif_missing = 1
        return stats

    stats.structures = 1

    options = dict(task_options or {})
    # Deterministic per (structure, variant): re-running reproduces the same
    # task set, and two variants of one structure never draw the same window.
    # zlib.crc32, NOT the builtin hash(): hash() of a str is salted per
    # process (PYTHONHASHSEED), so seeding with it made two identical runs
    # produce different task mixes.
    rng = np.random.default_rng(
        zlib.crc32(f"{pdb_id}|{variant}|{seed}".encode())
    )
    if not groups:
        return stats
    task = select_task(groups, rng, **options)
    stats.tasks[task.mode] = stats.tasks.get(task.mode, 0) + 1

    targets = {chain: set(seqs) for chain, seqs in task.designed.items()}
    if not targets:
        # A fold task: the unmodified structure IS the training example, so
        # copy it through rather than rewriting anything.
        #
        # Written WITHOUT a variant suffix on purpose. Mol-1 cannot express
        # the difference between ABSelector's full_fold and loop_fold (that
        # distinction is which residues are *conditioned*, and there is no
        # structure_group channel), so every fold variant of one structure
        # collapses to the same bytes. Sharing one filename makes them one
        # training example instead of N identical ones.
        out = output_dir / f"{pdb_id}.cif"
        if out.exists():
            stats.fold_duplicates = 1
            return stats
        out.write_bytes(source.read_bytes())
        stats.written = 1
        stats.fold_examples = 1
        return stats

    try:
        structure = gemmi.read_structure(str(source))
        # First pass assigns subchains, which is how CDR ranges are addressed
        # (label_asym_id + label_seq).
        structure.setup_entities()
        converted = design_structure(structure, targets, codebook, stats)
        if not converted:
            return stats
        patch_entity_sequences(structure, converted, stats)
        # Second pass is NOT redundant. `entity_poly_seq` is the canonical
        # sequence, and Mol-1's parser trusts it over `atom_site`: leaving it
        # listing the native residues makes the parser rebuild them and discard
        # every designed atom, silently producing an ordinary folding example.
        structure.setup_entities()
        # The filename is NOT what identifies a structure downstream.
        # `prepare_training_data.py` keys its output on the parsed
        # `_entry.id` (``mol/data/core/parser.py:165``), not on the file
        # stem, so two variants that both declare ``1A4J`` write to the same
        # ``1a4j.pkl.gz`` and the last one to finish wins. The effects are
        # worse than lost data: a fold variant can overwrite a design variant,
        # turning a design example into a folding one, and which one survives
        # depends on worker scheduling. Measured on the first build: every
        # ``_v1`` file was silently discarded, capping 5270 prepared CIFs at
        # 847 bioassemblies, and 37% of what remained had no designed residue.
        #
        # `_entry.id` comes from `Structure.info`, NOT from `Structure.name`
        # (gemmi 0.7: setting only `.name` renames the data block and leaves
        # `_entry.id` at its original value, which is exactly the field the
        # parser reads). Both are set so the file is self-consistent.
        variant_id = f"{pdb_id}{suffix_for(variant)}"
        structure.name = variant_id.upper()
        structure.info["_entry.id"] = variant_id.upper()
        structure.make_mmcif_document().write_file(
            str(output_dir / f"{variant_id}.cif")
        )
    except Exception as exc:  # a malformed entry must not kill the run
        logger.warning("%s: %s: %s", pdb_id, type(exc).__name__, exc)
        stats.failures = 1
        return stats

    stats.residues_designed = sum(len(v) for v in converted.values())
    stats.written = 1
    return stats


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cif-dir", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--annotations", type=Path,
                        help="CSV with *_label_id and *_cdr_ranges columns.")
    source.add_argument("--manifest", type=Path,
                        help="BoltzGen manifest.json. Carries more than the "
                             "CSV -- scFv, multi-antibody entries, framework "
                             "extent -- and needs no separate curation.")
    parser.add_argument("--antibody-instance", type=int, default=0,
                        help="With --manifest: which antibody_instances entry "
                             "to design. Pass -1 for every annotated chain.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chains", default="both",
                        choices=("heavy", "light", "both"))
    parser.add_argument("--task", default=LOOP_DESIGN,
                        choices=(*TASKS, "sample"),
                        help="Antibody task. 'sample' draws per variant with "
                             "ABSelector's probabilities; the others force one.")
    parser.add_argument("--cdrs", default=None,
                        help="Comma-separated CDR names to design, e.g. 'H3' "
                             "or 'H1,H2,H3'. Overrides window sampling, so a "
                             "targeted experiment is reproducible.")
    parser.add_argument("--loop-design-rate", type=float, default=0.5,
                        help="Window size is 1 + Binomial(N-1, rate): 0 is "
                             "always one loop, 1 is always all of them.")
    parser.add_argument("--loop-fold-prob", type=float, default=0.2,
                        help="With --task sample: probability of loop_fold.")
    parser.add_argument("--antibody-fold-prob", type=float, default=0.2,
                        help="With --task sample: probability of full_fold.")
    parser.add_argument("--variants-per-structure", type=int, default=1,
                        help="Emit N task variants per structure. Recovers "
                             "much of the per-epoch resampling that BoltzGen "
                             "gets from selecting at training time.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        instance = None if args.antibody_instance < 0 else args.antibody_instance
        entries = entries_from_manifest(args.manifest, instance)
        logger.info("%d annotated entries from manifest", len(entries))
    else:
        with open(args.annotations) as handle:
            entries = entries_from_csv(csv.DictReader(handle), args.chains)
        logger.info("%d annotated entries from CSV", len(entries))
    if args.limit:
        entries = entries[: args.limit]

    codebook = build_codebook()

    # Stamp which codeword table built this data. The codeword IS the label
    # geometry: a set prepared under one strategy and trained under another
    # is mislabelled in every designed residue, with matching shapes, finite
    # losses and no warning anywhere -- the model just learns the wrong code.
    # `load_codebook_stamp` turns that into a startup error.
    stamp = args.output_dir / "codebook.json"
    payload = {
        "strategy": design_residue.CODEBOOK_STRATEGY,
        "fingerprint": design_residue.codebook_fingerprint(codebook),
        "n_slots": design_residue.N_DESIGN_SLOTS,
        "codewords": {r: list(c) for r, (_, c) in sorted(codebook.items())},
    }
    if stamp.exists():
        previous = json.loads(stamp.read_text())
        if previous.get("fingerprint") != payload["fingerprint"]:
            logger.error(
                "%s was built with codebook %s (%s) but this run uses %s (%s). "
                "Mixing them produces a silently mislabelled dataset. Write to "
                "a new --output-dir, or rebuild the whole set.",
                args.output_dir, previous.get("fingerprint"),
                previous.get("strategy"), payload["fingerprint"],
                payload["strategy"])
            return 1
    else:
        stamp.write_text(json.dumps(payload, indent=1))
    logger.info("codebook %s (%s)", payload["fingerprint"], payload["strategy"])

    task_options = {
        "mode": None if args.task == "sample" else args.task,
        "loop_design_rate": args.loop_design_rate,
        "loop_fold_prob": args.loop_fold_prob,
        "antibody_fold_prob": args.antibody_fold_prob,
        "only_cdrs": (
            [c.strip() for c in args.cdrs.split(",") if c.strip()]
            if args.cdrs else None
        ),
    }
    logger.info("task options: %s", task_options)
    jobs = [
        (entry, variant)
        for entry in entries
        for variant in range(max(1, args.variants_per_structure))
    ]
    logger.info("%d structure-variant job(s)", len(jobs))

    if args.num_workers > 1:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=args.num_workers)(
            delayed(process_row)(
                entry, args.cif_dir, args.output_dir, codebook,
                task_options, variant, args.seed
            )
            for entry, variant in jobs
        )
    else:
        results = [
            process_row(entry, args.cif_dir, args.output_dir,
                        codebook, task_options, variant, args.seed)
            for entry, variant in jobs
        ]

    total = DesignStats()
    for item in results:
        total.merge(item)
    logger.info("%s", total)

    if total.written == 0:
        logger.error("no structures written -- check --cif-dir and the annotation source (%d entries had no CIF)", total.cif_missing)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
