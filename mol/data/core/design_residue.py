"""Fixed-slot atom layout for *designed* residues.

LOCAL ADDITION -- not part of upstream molence/Mol-1.

A generative design model needs every designed residue to occupy the *same*
number of atoms, because the number of atoms in a CCD component is derived
from the residue name and therefore leaks the residue identity the model is
supposed to be inventing. The fix, following BoltzGen, is a fixed layout of
:data:`N_DESIGN_SLOTS` atoms per designed residue, where the amino-acid
identity is carried geometrically -- by which backbone atom the surplus slots
are parked on -- rather than by any discrete sequence output.

Why ``UNK`` and not a new three-letter code
-------------------------------------------
Mol-1's residue vocabulary is sealed three ways, so a new code cannot be added
without breaking the released checkpoint:

1. ``Featurizer.encoder`` asserts ``len(dict) == max(values) + 1``
   (``mol/data/core/featurizer.py``), so an alias sharing ``UNK``'s index 20
   fails immediately.
2. A genuinely new index widens ``restype`` from 32 to 33, which
   ``load_strict: True`` rejects.
3. ``ELEMS`` is built as ``len(STD_RESIDUES) + idx``
   (``mol/data/constants.py``), so *any* new residue silently shifts every
   ligand element token id away from what the checkpoint was trained on.

``UNK`` already satisfies every name-keyed gate in the pipeline -- it is three
characters, sits at ``restype`` 20, and carries ``N/CA/C/O/CB``, so the
tokenizer emits one token per residue rather than atomizing it. Widening its
atom layout therefore needs **zero** changes to those gates.

The cost, accepted deliberately: this changes ``UNK`` globally, so a genuinely
unknown *polymer* residue in a deposited structure is now allocated
:data:`N_DESIGN_SLOTS` atoms too. Ligands are unaffected -- they take the
``mol_type == "ligand"`` branch in the tokenizer and get ``tokatom_idx = 0``.

Slot layout
-----------
Slots 0-5 are ``UNK``'s real CCD atoms (``N, CA, C, O, CB, CG``; ``OXT`` is
flagged as a leaving atom upstream and is already filtered out). Slots 6-13 are
virtual. Slots 0-3 being the backbone in that order is load-bearing: the
geometric decoder counts how many surplus slots sit within a threshold of
``N`` and of ``O``.
"""

from __future__ import annotations

import itertools
import logging
import os
from typing import Any, Mapping, Optional, Sequence

from functools import lru_cache
from types import MappingProxyType

import numpy as np

logger = logging.getLogger(__name__)

#: The CCD code reused as the designed-residue component.
DESIGN_RESIDUE = "UNK"

#: Atoms per designed residue. 14 is forced by the atom inventory: the largest
#: standard residue (TRP) has exactly 14 heavy atoms once ``OXT`` is excluded,
#: and 14 stays under ``max_atoms_per_token = 24``, so the released
#: ``ConfidenceHead`` weight tables are unaffected.
N_DESIGN_SLOTS = 14

#: ``UNK``'s real atoms, in CCD order. ``OXT`` is excluded: upstream flags it
#: ``pdbx_leaving_atom_flag = Y`` and ``get_component_atom_array`` drops it.
BASE_ATOMS: tuple[str, ...] = ("N", "CA", "C", "O", "CB", "CG")

#: The four backbone slots, always first and in this order. The geometric
#: decoder slices ``slots[:4]`` as the backbone, so the order is load-bearing.
BACKBONE_ATOMS_4: tuple[str, ...] = ("N", "CA", "C", "O")

#: Atoms excluded from the layout. ``OXT`` would push TRP to 15 atoms and make
#: a C-terminal designed residue a different size from an internal one.
EXCLUDED_ATOMS: tuple[str, ...] = ("OXT",)

#: The 20 standard amino acids, as three-letter codes.
STANDARD_AA: tuple[str, ...] = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)

#: Names for the surplus slots. Distinct from any real PDB atom name so they
#: can never collide with a deposited atom.
VIRTUAL_ATOM_NAMES: tuple[str, ...] = tuple(
    f"DV{i:02d}" for i in range(len(BASE_ATOMS), N_DESIGN_SLOTS)
)

#: Element for virtual atoms. ``get_all_elems()`` reserves ``UNK_ELEM_119``
#: through ``UNK_ELEM_128`` as spare one-hot slots, so this costs no
#: vocabulary change and keeps virtual atoms distinguishable from real carbon.
VIRTUAL_ELEMENT = "UNK_ELEM_119"

#: Atomic number for virtual atoms inside the RDKit mol. 0 is RDKit's dummy
#: atom ("*"); it keeps the mol sanitizable without claiming a real element.
VIRTUAL_ATOMIC_NUMBER = 0


def design_slot_names() -> tuple[str, ...]:
    """The ``N_DESIGN_SLOTS`` atom names, in slot order."""
    return BASE_ATOMS + VIRTUAL_ATOM_NAMES


def is_extended(atom_names: Sequence[str]) -> bool:
    """Whether this atom-name list has already been given virtual slots."""
    return bool(set(VIRTUAL_ATOM_NAMES) & set(map(str, atom_names)))


def extend_component_atom_array(comp: Any) -> Any:
    """Pad a biotite ``AtomArray`` out to :data:`N_DESIGN_SLOTS` atoms.

    Virtual atoms are appended as copies of ``CA`` -- duplicating an existing
    row means every annotation, including ones added by callers, is carried
    over rather than guessed -- and are then overridden in place.

    Reference coordinates are deliberately left *on top of* ``CA`` rather than
    at the origin: a slot at ``(0, 0, 0)`` would sit a long way from the rest
    of the residue and inject spurious geometry into any distance feature
    computed before masking.

    Returns ``comp`` unchanged if it is ``None``, already extended, or already
    at least :data:`N_DESIGN_SLOTS` long.
    """
    if comp is None:
        return None

    names = [str(n) for n in comp.atom_name]
    if is_extended(names):
        return comp

    # Drop anything outside the fixed layout -- in practice OXT, which the
    # polymer builder keeps on a C-TERMINAL residue. Without this a designed
    # residue would be 14 atoms mid-chain but 15 at the C-terminus, breaking
    # the fixed-size invariant the geometric decoder depends on. OXT is also
    # excluded from the codebook for the same reason: with it, TRP needs 15.
    layout = set(design_slot_names())
    keep = np.array([name in layout for name in names], dtype=bool)
    if not keep.all():
        comp = comp[keep]
        names = [str(n) for n in comp.atom_name]

    n_missing = N_DESIGN_SLOTS - len(comp)
    if n_missing <= 0:
        logger.warning(
            "design residue component already has %d atoms (>= %d slots); "
            "leaving it alone",
            len(comp),
            N_DESIGN_SLOTS,
        )
        return comp

    # Duplicate CA for the surplus rows so inherited annotations stay valid.
    # Bonds have to come off first: biotite's BondList refuses duplicate
    # indices ("Duplicate indices are not supported"). Real atoms keep indices
    # 0..n-1 because we only ever append, so the original bond array stays
    # valid and is simply reattached against the larger atom count, leaving the
    # virtual slots unbonded -- which is what they are.
    import biotite.structure as struc

    bonds = getattr(comp, "bonds", None)
    unbonded = comp.copy()
    unbonded.bonds = None

    ca_index = names.index("CA") if "CA" in names else 0
    extended = unbonded[list(range(len(comp))) + [ca_index] * n_missing]
    if bonds is not None:
        extended.bonds = struc.BondList(N_DESIGN_SLOTS, bonds.as_array())

    virtual = slice(len(comp), N_DESIGN_SLOTS)
    virtual_names = VIRTUAL_ATOM_NAMES[:n_missing]

    # Rebuild the string annotations with a wide dtype. Biotite sizes these
    # narrowly (element is typically "U2"), which would silently TRUNCATE
    # "UNK_ELEM_119" to "UN" and blow up one-hot encoding downstream.
    for annotation, fill in (
        ("atom_name", virtual_names),
        ("element", [VIRTUAL_ELEMENT] * n_missing),
    ):
        values = np.asarray(getattr(extended, annotation), dtype="U16")
        values[virtual] = fill
        extended.set_annotation(annotation, values)

    # Mirror the atom names into the CCD bookkeeping annotations when present,
    # so anything keying off them sees the virtual names too.
    for annotation in ("alt_atom_id", "pdbx_component_atom_id"):
        if annotation in extended.get_annotation_categories():
            values = np.asarray(getattr(extended, annotation), dtype="U16")
            values[virtual] = virtual_names
            extended.set_annotation(annotation, values)

    if "charge" in extended.get_annotation_categories():
        charge = np.asarray(extended.charge).copy()
        charge[virtual] = 0
        extended.set_annotation("charge", charge)

    if "leaving_atom_flag" in extended.get_annotation_categories():
        flags = np.asarray(extended.leaving_atom_flag).copy()
        flags[virtual] = False  # never strip the slots we just added
        extended.set_annotation("leaving_atom_flag", flags)

    # A designed residue has no leaving atoms by construction -- the only one
    # UNK had (OXT) is dropped above. An empty mapping is what
    # ``remove_leaving_atoms`` expects: it calls ``.get(centre_name)`` and
    # skips on None, whereas a bare None here would be an AttributeError.
    extended.central_to_leaving_groups = {}
    return extended


def extend_rdkit_mol(mol: Any) -> Any:
    """Pad an RDKit mol out to :data:`N_DESIGN_SLOTS` atoms.

    Mirrors :func:`extend_component_atom_array` on the reference-conformer
    side, extending the custom attributes ``atom_map``, ``ref_mask`` and
    ``ref_conf_id`` that ``get_ccd_ref_info`` reads.

    Virtual atoms get ``ref_mask`` from :data:`VIRTUAL_REF_MASK`. The atom
    encoder multiplies the per-atom embedding by ``ref_mask``
    (``mol/model/modules/transformer.py``), so ``False`` makes a virtual slot
    contribute a zero embedding while still occupying an index and still
    receiving a predicted coordinate, and ``True`` makes it visible -- which
    is what BoltzGen does. See :data:`VIRTUAL_REF_MASK` for why that matters
    and which is the default.
    """
    from rdkit import Chem

    if mol is None:
        return None

    atom_map = dict(getattr(mol, "atom_map", {}))

    # Size is decided by the virtual slot NAMES, not by mol.GetNumAtoms().
    # The RDKit mol is a name-keyed lookup table -- ``add_ref_feat_info`` does
    # ``atom_map[atom_name]`` with a bare subscript -- and it still carries
    # hydrogens and OXT, so its length (16 for UNK) has nothing to do with the
    # 14-slot layout. Any virtual name missing from atom_map is a hard KeyError
    # during featurization of a designed residue.
    missing_names = [name for name in VIRTUAL_ATOM_NAMES if name not in atom_map]
    if not missing_names:
        return mol
    n_missing = len(missing_names)

    editable = Chem.RWMol(mol)
    for _ in range(n_missing):
        editable.AddAtom(Chem.Atom(VIRTUAL_ATOMIC_NUMBER))
    extended = editable.GetMol()

    base_n = mol.GetNumAtoms()
    assert extended.GetNumAtoms() == base_n + n_missing

    # AddAtom already widened every existing conformer, filling the new rows
    # with the origin. Overwrite those rows in place -- adding a fresh
    # conformer with the same id instead would leave GetConformer() returning
    # the zero-filled original. Virtual atoms are parked on CA for the reason
    # given in extend_component_atom_array.
    ca_index = atom_map.get("CA", 0)
    for conformer in extended.GetConformers():
        ca_position = conformer.GetAtomPosition(ca_index)
        for i in range(base_n, extended.GetNumAtoms()):
            conformer.SetAtomPosition(i, ca_position)

    for offset, name in enumerate(missing_names):
        atom_map[name] = base_n + offset
    extended.atom_map = atom_map
    extended.ref_conf_id = getattr(mol, "ref_conf_id", -1)

    ref_mask = np.asarray(getattr(mol, "ref_mask", np.ones(base_n, dtype=bool)))
    fill = np.ones if VIRTUAL_REF_MASK else np.zeros
    extended.ref_mask = np.concatenate(
        [ref_mask, fill(n_missing, dtype=ref_mask.dtype)]
    )
    return extended


#: Observed amino-acid composition of designed (CDR) residues, measured by
#: decoding 4,572 designed residues out of 118 prepared structures in
#: ``design_data_v2``. Used only by the ``balanced`` codeword strategy, to
#: weight the N/O split by how often each residue is actually supervised
#: rather than treating all twenty as equally common -- SER alone is 14% of
#: designed positions and CYS is 0.04%.
CDR_COMPOSITION: Mapping[str, float] = MappingProxyType({
    "SER": 0.142, "TYR": 0.113, "THR": 0.093, "GLY": 0.083, "ASN": 0.077,
    "ASP": 0.065, "LEU": 0.058, "PHE": 0.051, "ALA": 0.050, "ARG": 0.047,
    "GLN": 0.042, "PRO": 0.035, "HIS": 0.029, "ILE": 0.025, "LYS": 0.023,
    "GLU": 0.022, "VAL": 0.022, "TRP": 0.015, "MET": 0.008, "CYS": 0.001,
})

#: Whether a virtual slot is visible to the atom encoder.
#:
#: ``False`` is what this fork shipped: ``ref_mask = 0``, and Mol-1 multiplies
#: the per-atom embedding by it (``transformer.py:953,965``) *after* the
#: element and atom-name contributions, so the whole embedding is zeroed. A
#: virtual slot then holds an index and receives a predicted coordinate while
#: contributing nothing -- which was the right call for Phase 0, where the
#: goal was to get a 14-slot residue through the pipeline without touching
#: the architecture.
#:
#: It is NOT what BoltzGen does, and the difference falls exactly on the
#: slots that carry the identity code. BoltzGen gives a fake atom a real
#: element row (``const.fake_element_id``), a real reference position (CA's),
#: and full passage through the atom encoder --
#: ``data/feature/featurizer.py:1160-1165``; its ``atom_pad_mask`` is tensor
#: padding and does not gate fakes out. Under ``False`` our eight code-bearing
#: slots enter the structure module with identical, zero representations, all
#: their reference positions are CA copies so the pair features are degenerate
#: too, and ``atom_to_tokatom_idx`` reaches only the confidence head. The
#: model cannot tell slot 6 from slot 13, yet the per-index MSE demands a
#: different coordinate at each.
#:
#: ``True`` restores the BoltzGen behaviour: the slot keeps its distinct
#: :data:`VIRTUAL_ELEMENT` and its CA reference position, and is no longer
#: zeroed. No architectural change, and no data rebuild -- ``ref_mask`` comes
#: from the CCD component at featurization time, while the prepared CIFs hold
#: only coordinates.
#:
#: Default is the historical behaviour so existing runs reproduce; flip it
#: with ``MOL_DESIGN_VIRTUAL_REF_MASK=1`` for the A/B.
VIRTUAL_REF_MASK: bool = os.environ.get(
    "MOL_DESIGN_VIRTUAL_REF_MASK", "0"
).lower() in ("1", "true", "yes")


#: How surplus slots are split between the two anchors.
#:
#: ``pure_o_first``
#:     What Mol-1 shipped. The greedy tries ``on_n = 0`` first, so the first
#:     residues to claim take the pure-O codewords ``(0, k)``. Ten of twenty
#:     end up pure-O and 87.9% of label mass (weighted by
#:     :data:`CDR_COMPOSITION`) sits on ``O``. Under a coordinate MSE, whose
#:     minimiser under uncertainty is the conditional mean, that makes "park
#:     everything on O" the risk-minimising hedge -- and it scores 49% of
#:     designed residues correct for free. Scoring xid 284 found the model
#:     doing exactly this: 77.6% of slots that should park on N were inside
#:     O's capture zone, 0.0% the reverse.
#:
#: ``balanced``
#:     Chooses the split that brings weighted N mass closest to O mass, giving
#:     50.7% / 49.3%, while leaving both the pure-O and the pure-N columns
#:     empty. Neither anchor is then a useful hedge, and neither degenerate
#:     strategy produces a decodable codeword.
#:
#: CHANGING THIS INVALIDATES PREPARED DATA AND CHECKPOINTS. The codeword *is*
#: the label geometry, so a dataset prepared under one strategy and trained
#: under the other is silently mislabelled in every designed residue. Stamp
#: :func:`codebook_fingerprint` into prepared data and check it.
CODEBOOK_STRATEGY: str = os.environ.get("MOL_CODEBOOK_STRATEGY", "pure_o_first")

_STRATEGIES = ("pure_o_first", "balanced")


def _claim_pure_o_first(
    groups: Mapping[int, Sequence[str]],
) -> dict[str, tuple[int, int]]:
    """Mol-1 as shipped: first free split, counting ``on_n`` up from zero."""
    claimed: set[tuple[int, int]] = set()
    out: dict[str, tuple[int, int]] = {}
    for k in sorted(groups):
        for res in groups[k]:
            for on_n in range(k + 1):
                if (on_n, k - on_n) not in claimed:
                    claimed.add((on_n, k - on_n))
                    out[res] = (on_n, k - on_n)
                    break
            else:  # pragma: no cover - unreachable for the standard 20
                raise ValueError(f"no free (N, O) split for {res}")
    return out


def _claim_balanced(
    groups: Mapping[int, Sequence[str]],
    composition: Mapping[str, float],
) -> dict[str, tuple[int, int]]:
    """Split so the frequency-weighted N and O mass come out even.

    Residues sharing a surplus total compete for the same cells -- ``on_n +
    on_o`` is fixed by the atom inventory -- so the choice is made per group
    by exhaustive search over assignments of distinct ``on_n`` values. The
    largest group has five members over nine cells, so this is a few thousand
    evaluations and needs no heuristic to be trusted.

    **Both extreme columns are excluded.** A codeword with ``on_n = 0`` is one
    a model that has learned only "park on O" produces for free, and because
    the achievable surplus totals are ``{0, 2, 3, ... 10}``, an occupied
    ``(0, k)`` also means that model's *mistakes* land on a valid codeword and
    decode silently. ``on_o = 0`` is the mirror of the same problem. Excluding
    both costs 0.2 percentage points of balance (50.7% vs 50.5% on O) and
    takes the silent-failure count for either degenerate strategy from six to
    zero.

    A zero-surplus residue has exactly one possible codeword, ``(0, 0)``, and
    is passed through. That is TRP, and it is why the decode sink cannot be
    closed here -- it needs a separate occupancy guard.
    """
    out: dict[str, tuple[int, int]] = {}
    for k in sorted(groups):
        members = list(groups[k])
        if k == 0:
            for res in members:
                out[res] = (0, 0)
            continue
        cells = list(range(1, k)) if k >= 2 else list(range(k + 1))
        if len(cells) < len(members):
            raise ValueError(
                f"{len(members)} residues share surplus {k} but only "
                f"{len(cells)} interior codewords exist for it")
        best, best_cost = None, None
        for combo in itertools.permutations(cells, len(members)):
            cost = abs(sum(composition.get(m, 0.0) * (2 * a - k)
                           for m, a in zip(members, combo)))
            if best_cost is None or cost < best_cost:
                best, best_cost = combo, cost
        for res, on_n in zip(members, best):
            out[res] = (on_n, k - on_n)
    return out


def build_codebook(
    res_atoms_dict: Optional[Mapping[str, Mapping[str, int]]] = None,
    residues: Sequence[str] = STANDARD_AA,
    strategy: Optional[str] = None,
) -> Mapping[str, tuple[tuple[str, ...], tuple[int, int]]]:
    """Derive the geometric residue-type codebook from Mol-1's atom inventory.

    For every standard residue, returns its real atoms in slot order and the
    codeword ``(count_on_N, count_on_O)`` saying where its surplus slots park.
    Decoding counts how many surplus slots sit within a threshold of ``N`` and
    of ``O``, so the codewords must be unique -- which this asserts.

    Surplus slots are assigned greedily, **fewest free slots first**: TRP fills
    all 14 and has exactly one possible codeword, so it must claim it before
    roomier residues crowd it out. Ties break alphabetically for reproducibility.

    This is derived rather than hard-coded because the inventory differs
    between models -- Mol-1's table includes ``OXT``, BoltzGen's does not -- so
    a codebook copied from another model is simply wrong here.
    """
    strategy = CODEBOOK_STRATEGY if strategy is None else strategy
    if strategy not in _STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}, expected one of "
                         f"{_STRATEGIES}")

    # The no-argument call is the hot path -- `decode_residue` makes it once
    # per residue when a codebook is not passed in, and deriving the table
    # costs ~0.4 ms. Only that case is cached: the arguments are a Mapping and
    # a Sequence, neither hashable, so `lru_cache` cannot wrap the function
    # itself. Callers that supply their own inventory still derive fresh.
    if res_atoms_dict is None and tuple(residues) == tuple(STANDARD_AA):
        return _default_codebook(strategy)

    if res_atoms_dict is None:
        from mol.data.constants import RES_ATOMS_DICT as res_atoms_dict

    atom_order: dict[str, tuple[str, ...]] = {}
    for res in residues:
        names = [
            name
            for name, _ in sorted(res_atoms_dict[res].items(), key=lambda kv: kv[1])
            if name not in EXCLUDED_ATOMS
        ]
        missing = [a for a in BACKBONE_ATOMS_4 if a not in names]
        if missing:
            raise ValueError(f"{res} is missing backbone atom(s) {missing}")
        if len(names) > N_DESIGN_SLOTS:
            raise ValueError(
                f"{res} has {len(names)} atoms, which overflows "
                f"N_DESIGN_SLOTS={N_DESIGN_SLOTS}"
            )
        sidechain = [a for a in names if a not in BACKBONE_ATOMS_4]
        atom_order[res] = tuple(BACKBONE_ATOMS_4) + tuple(sidechain)

    # Group by surplus count. Residues only ever compete with others of the
    # same count -- `on_n + on_o` is fixed by the atom inventory -- so the
    # claim is independent per group, which is what makes the exhaustive
    # search in the balanced strategy cheap. Ties break alphabetically for
    # reproducibility.
    groups: dict[int, list[str]] = {}
    for res in sorted(residues):
        groups.setdefault(N_DESIGN_SLOTS - len(atom_order[res]), []).append(res)

    if strategy == "balanced":
        words = _claim_balanced(groups, CDR_COMPOSITION)
    else:
        words = _claim_pure_o_first(groups)

    codebook = {res: (atom_order[res], words[res]) for res in sorted(residues)}
    assert len({c for _, c in codebook.values()}) == len(codebook), "codewords collide"
    return MappingProxyType(codebook)


@lru_cache(maxsize=len(_STRATEGIES))
def _default_codebook(
    strategy: str = "pure_o_first",
) -> Mapping[str, tuple[tuple[str, ...], tuple[int, int]]]:
    """The standard-20 codebook, derived once.

    Returns a read-only ``MappingProxyType`` rather than a dict. The cache
    hands every caller the SAME object, so a mutable one would let any of
    them silently redefine the residue code for the whole process -- the kind
    of shared-mutable-default bug that surfaces as a wrong amino acid months
    later. The values are already tuples, so the proxy makes the whole
    structure immutable.
    """
    from mol.data.constants import RES_ATOMS_DICT

    return build_codebook(RES_ATOMS_DICT, STANDARD_AA, strategy=strategy)


def encode_residue(
    res_name: str,
    atom_coords: Mapping[str, Sequence[float]],
    codebook: Optional[Mapping[str, tuple[tuple[str, ...], tuple[int, int]]]] = None,
) -> np.ndarray:
    """Lay a native residue out in the fixed slots, carrying its identity.

    Returns ``(N_DESIGN_SLOTS, 3)``. Real atoms occupy the leading slots in
    codebook order; the surplus are placed *exactly* on this residue's own
    ``N`` or ``O`` per its codeword. "Exactly" matters -- decoding uses a 0.5 A
    threshold, and the tightest genuine sidechain-to-backbone distance measured
    over real structures is 1.444 A (a PRO), so jitter here eats that margin.

    Raises
    ------
    KeyError
        If the residue is unknown or an atom it needs is unresolved. Callers
        should skip such residues; they cannot be supervised.
    """
    if codebook is None:
        codebook = build_codebook()
    order, (on_n, on_o) = codebook[res_name]

    slots = np.zeros((N_DESIGN_SLOTS, 3), dtype=np.float32)
    for i, atom in enumerate(order):
        try:
            slots[i] = atom_coords[atom]
        except KeyError as exc:
            raise KeyError(f"{res_name} is missing atom {atom}") from exc

    cursor = len(order)
    slots[cursor : cursor + on_n] = slots[BACKBONE_ATOMS_4.index("N")]
    cursor += on_n
    slots[cursor : cursor + on_o] = slots[BACKBONE_ATOMS_4.index("O")]
    return slots


def decode_residue(
    slots: np.ndarray,
    codebook: Optional[Mapping[str, tuple[tuple[str, ...], tuple[int, int]]]] = None,
    threshold: float = 0.5,
    unknown: str = "UNK",
) -> str:
    """Read a residue type back out of its fixed slots.

    Counts how many surplus slots sit within ``threshold`` angstroms of ``N``
    and of ``O`` and looks the pair up. The inverse of :func:`encode_residue`.

    The 0.5 A default has a measured ~2.9x margin: over 52,199 fully-resolved
    residues the tightest genuine sidechain-to-backbone distance was 1.444 A,
    in a PRO (whose CD is bonded to N).

    Note the irreducible ambiguity: TRP fills all 14 slots and parks nothing,
    so its codeword is ``(0, 0)`` -- which is also what a residue whose surplus
    atoms went nowhere near the backbone produces. A parse failure therefore
    reads as tryptophan.
    """
    if codebook is None:
        codebook = build_codebook()
    by_codeword = {code: res for res, (_, code) in codebook.items()}

    slots = np.asarray(slots, dtype=np.float64)
    if slots.shape != (N_DESIGN_SLOTS, 3):
        raise ValueError(f"expected ({N_DESIGN_SLOTS}, 3), got {slots.shape}")

    backbone = slots[: len(BACKBONE_ATOMS_4)]
    surplus = slots[len(BACKBONE_ATOMS_4) :]
    dist = np.linalg.norm(surplus[None, :, :] - backbone[:, None, :], axis=-1)
    nearest, nearest_dist = dist.argmin(axis=0), dist.min(axis=0)

    counts = np.zeros(len(BACKBONE_ATOMS_4), dtype=int)
    for index in nearest[nearest_dist <= threshold]:
        counts[index] += 1

    codeword = (
        int(counts[BACKBONE_ATOMS_4.index("N")]),
        int(counts[BACKBONE_ATOMS_4.index("O")]),
    )
    return by_codeword.get(codeword, unknown)


def codebook_fingerprint(
    codebook: Optional[Mapping[str, tuple[tuple[str, ...], tuple[int, int]]]] = None,
) -> str:
    """Short stable hash of a codebook's residue -> codeword mapping.

    The codeword IS the label geometry: a dataset prepared under one strategy
    and trained under another is mislabelled in every designed residue, and
    nothing downstream would notice -- the shapes match, the losses are
    finite, and the model simply learns the wrong code. Stamp this into
    prepared data and compare it at load time so that failure is loud.

    Only the codewords enter the hash. Atom order is derived from the same
    inventory and cannot differ without the codewords differing too.
    """
    import hashlib

    if codebook is None:
        codebook = build_codebook()
    payload = ";".join(f"{res}:{code[0]},{code[1]}"
                       for res, (_, code) in sorted(codebook.items()))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def check_codebook_stamp(data_dir: Any, strict: bool = True) -> Optional[dict]:
    """Verify a prepared dataset was built with the codebook now in force.

    ``prepare_design_data.py`` writes ``codebook.json`` beside the data. A
    mismatch means every designed residue in that set carries the wrong
    codeword, which is invisible downstream -- shapes match, losses are
    finite, the model simply learns a different code. Raising is the point.

    Returns the stamp, or ``None`` if the directory has none (datasets built
    before stamping existed; a warning, not an error, since they predate the
    only strategy that was available).
    """
    import json
    from pathlib import Path as _Path

    stamp_path = _Path(data_dir) / "codebook.json"
    if not stamp_path.is_file():
        logger.warning(
            "%s has no codebook.json; cannot verify it matches strategy %r. "
            "Rebuild, or confirm by hand.", data_dir, CODEBOOK_STRATEGY)
        return None

    stamp = json.loads(stamp_path.read_text())
    mine = codebook_fingerprint()
    if stamp.get("fingerprint") != mine:
        message = (
            f"{data_dir} was prepared with codebook "
            f"{stamp.get('fingerprint')} ({stamp.get('strategy')}) but this "
            f"process uses {mine} ({CODEBOOK_STRATEGY}). Every designed "
            f"residue would be mislabelled. Set MOL_CODEBOOK_STRATEGY="
            f"{stamp.get('strategy')!r}, or rebuild the dataset."
        )
        if strict:
            raise ValueError(message)
        logger.error("%s", message)
    return stamp


def normalize_design_atom_elements(atom_array: Any) -> Any:
    """Force designed residues' virtual slots onto :data:`VIRTUAL_ELEMENT`.

    Both input paths must agree on what a designed residue looks like, and they
    do not by default:

    * the JSON/inference path builds residues from the padded CCD component, so
      virtual slots already carry ``VIRTUAL_ELEMENT``;
    * the mmCIF/training path reads ``_atom_site.type_symbol`` from the file,
      and a CIF cannot express ``UNK_ELEM_119`` -- the field is a two-character
      element symbol, which biotite would truncate to ``"UN"``.

    Without this, the model would be trained on carbon-flavoured virtual slots
    and run on spare-element ones. Applied in ``Featurizer.__init__`` so it
    covers both paths at a single point. A no-op when no designed residue is
    present.
    """
    if atom_array is None or len(atom_array) == 0:
        return atom_array

    res_name = np.asarray(atom_array.res_name)
    is_design = res_name == DESIGN_RESIDUE
    if not is_design.any():
        return atom_array

    atom_name = np.asarray(atom_array.atom_name)
    virtual = is_design & np.isin(atom_name, np.asarray(VIRTUAL_ATOM_NAMES))
    if not virtual.any():
        return atom_array

    elements = np.asarray(atom_array.element, dtype="U16")
    if (elements[virtual] == VIRTUAL_ELEMENT).all():
        return atom_array
    elements[virtual] = VIRTUAL_ELEMENT
    atom_array.set_annotation("element", elements)
    return atom_array


def identity_perm(n_atom: int) -> np.ndarray:
    """The only safe atom permutation for a designed residue.

    ``get_substructure_perms`` matches the molecular graph, and the virtual
    slots are mutually indistinguishable disconnected dummy atoms -- eight of
    them generate 8! candidate permutations (capped at ``MaxMatches``). Any
    non-identity permutation would reshuffle the surplus slots and destroy the
    geometric identity code they exist to carry.

    Shape matches ``get_ccd_ref_info``'s ``perm``: ``(n_atom, n_perm)``.
    """
    return np.arange(n_atom, dtype=int).reshape(-1, 1)
