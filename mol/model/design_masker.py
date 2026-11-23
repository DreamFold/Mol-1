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

"""LOCAL ADDITION: blank the inputs that would reveal a designed residue.

A design model must not be shown the amino acid it is being asked to invent.
This module is the single choke point where those inputs are erased, applied
as the first thing ``Mol.forward`` does -- before ``generate_relp`` and
``update_input_feature_dict``, which derive ``relp``/``d_lm``/``v_lm`` from
features a masker could otherwise invalidate.

Why this is much smaller than BoltzGen's equivalent
---------------------------------------------------
BoltzGen's ``BoltzMasker`` blanks 15 feature keys and copies 50 more through
an allow-list, because there a designed residue is still represented as its
*native* residue right up until the mask is applied. Here the representation
does that work instead: a designed residue is a ``UNK`` component with a fixed
14-slot layout (see ``mol/data/core/design_residue.py``), so by the time
features exist:

* ``restype`` is already ``UNK`` -- nothing to blank;
* every per-atom reference feature (``ref_pos``, ``ref_element``,
  ``ref_charge``, ``ref_atom_name_chars``, ``ref_mask``) comes from the *same*
  ``UNK`` component for every designed residue, so it is identical across
  residue types and carries no signal to erase;
* the atom *count* is fixed at 14, which is the leak the layout exists to close.

What genuinely still leaks is anything derived from the **sequence** rather
than the structure: a real MSA, whose query row carries the native residue at
that column, and templates, which can carry the native structure and type.
Those are what the rules below erase.

Note that ``make_dummy_feature`` (``mol/data/utils.py``) synthesises ``msa``
and ``profile`` *from* ``restype``, so the MSA-free path is already clean --
it inherits ``UNK`` rather than the native residue.

Do NOT transplant BoltzGen's constants
--------------------------------------
Its vocabulary is offset from Mol-1's: ``boltzgen.data.const.token_ids["UNK"]``
is 22 in a 33-wide vocabulary that reserves ``<pad>`` and ``-`` before the
amino acids, whereas Mol-1's ``UNK`` is 20 in a 32-wide one. Writing 22 into
Mol-1's ``restype`` would silently mean RNA guanine.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Mapping, Optional, Sequence

import torch

from mol.data.constants import STD_RESIDUES, STD_RESIDUES_WITH_GAP

logger = logging.getLogger(__name__)

#: Residue index a designed (and any unknown) protein token carries.
UNK_RESTYPE = STD_RESIDUES["UNK"]

#: Width of the restype / profile one-hot.
RESTYPE_WIDTH = len(STD_RESIDUES_WITH_GAP)

#: Fill kinds a rule can request.
ZERO = "zero"
UNK_ID = "unk_id"
UNK_ONEHOT = "unk_onehot"


@dataclass(frozen=True)
class MaskRule:
    """One feature to blank over the designed tokens.

    Attributes
    ----------
    key:
        Feature-dict key.
    token_axes:
        Axes that are indexed by token, given as **negative** indices so the
        rule is insensitive to a leading batch dimension.
    fill:
        One of :data:`ZERO`, :data:`UNK_ID`, :data:`UNK_ONEHOT`.
    """

    key: str
    token_axes: tuple[int, ...]
    fill: str


#: MSA-derived features. The query row of a real alignment carries the native
#: residue, which is the primary leak once the representation is fixed-size.
MSA_RULES: tuple[MaskRule, ...] = (
    MaskRule("msa", (-1,), UNK_ID),
    MaskRule("has_deletion", (-1,), ZERO),
    MaskRule("deletion_value", (-1,), ZERO),
    MaskRule("profile", (-2,), UNK_ONEHOT),
    MaskRule("deletion_mean", (-1,), ZERO),
)

#: Template features. A template can carry both the native residue type and
#: its native geometry, so designed rows *and* columns are erased.
TEMPLATE_RULES: tuple[MaskRule, ...] = (
    MaskRule("template_restype", (-1,), UNK_ID),
    MaskRule("template_aatype", (-1,), UNK_ID),  # the name the featurizer uses
    MaskRule("template_all_atom_mask", (-2,), ZERO),
    MaskRule("template_all_atom_positions", (-3,), ZERO),
    MaskRule("template_pseudo_beta_mask", (-1,), ZERO),
    MaskRule("template_backbone_frame_mask", (-1,), ZERO),
    MaskRule("template_distogram", (-3, -2), ZERO),
    MaskRule("template_unit_vector", (-3, -2), ZERO),
)

DESIGN_MASK_RULES: tuple[MaskRule, ...] = MSA_RULES + TEMPLATE_RULES


def infer_design_mask(
    feature_dict: Mapping[str, torch.Tensor]
) -> Optional[torch.Tensor]:
    """Which tokens are being designed.

    Uses an explicit ``design_mask`` feature when present. Otherwise falls back
    to ``restype == UNK``, which is exact under this representation: a designed
    residue *is* a ``UNK`` residue. The fallback also catches genuinely unknown
    deposited residues, which is the desired behaviour -- their identity is not
    known either, so showing the model a sequence-derived hint for them would
    be the same leak.

    Returns ``None`` when neither is available.
    """
    if "design_mask" in feature_dict:
        return feature_dict["design_mask"].bool()

    restype = feature_dict.get("restype")
    if restype is None:
        return None
    if restype.dim() >= 2 and restype.shape[-1] == RESTYPE_WIDTH:
        is_unk = restype.argmax(dim=-1) == UNK_RESTYPE
    else:
        is_unk = restype == UNK_RESTYPE

    # `restype == UNK` alone is NOT enough: Featurizer.restype_onehot_encoded
    # represents every ligand as "unknown amino acid", and a ligand is
    # atomized one token per atom, so each of its atoms would be flagged as a
    # designed residue. Require the token to be protein as well.
    protein = _token_is_protein(feature_dict, n_token=is_unk.shape[-1])
    if protein is not None:
        is_unk = is_unk & protein
    return is_unk


def _token_is_protein(
    feature_dict: Mapping[str, torch.Tensor], n_token: int
) -> Optional[torch.Tensor]:
    """Token-level protein mask, scattered up from the per-atom ``is_protein``.

    Returns ``None`` when the features needed are absent, in which case the
    caller falls back to the (over-broad) restype test.
    """
    is_protein = feature_dict.get("is_protein")
    atom_to_token = feature_dict.get("atom_to_token_idx")
    if is_protein is None or atom_to_token is None:
        return None

    flat_atom = atom_to_token.reshape(-1).long()
    flat_protein = is_protein.reshape(-1)
    if flat_atom.numel() != flat_protein.numel():
        return None

    # Accumulate in float32, not bool.
    #
    # The obvious form -- scatter_reduce_(..., reduce="amax") onto a bool
    # tensor -- works on CPU and raises on CUDA:
    #   NotImplementedError: "cuda_scatter_gather_base_kernel_func"
    #                        not implemented for 'Bool'
    # which is invisible to a CPU-only test suite. index_add_ on float32 is
    # supported on both, and "any atom of this token is protein" is just a
    # positive sum.
    acc = torch.zeros(n_token, dtype=torch.float32, device=flat_protein.device)
    acc.index_add_(
        0, flat_atom.clamp(max=n_token - 1), flat_protein.float()
    )
    return acc > 0


def _fill_value(fill: str, tensor: torch.Tensor) -> Any:
    if fill == ZERO:
        return torch.zeros((), dtype=tensor.dtype, device=tensor.device)
    if fill == UNK_ID:
        return torch.full((), UNK_RESTYPE, dtype=tensor.dtype, device=tensor.device)
    raise ValueError(f"unhandled fill {fill!r}")


def _apply_rule(
    tensor: torch.Tensor, rule: MaskRule, design_mask: torch.Tensor
) -> torch.Tensor:
    """Blank ``tensor`` wherever any of its token axes is a designed token."""
    out = tensor.clone()
    n_token = design_mask.shape[-1]

    for axis in rule.token_axes:
        if tensor.dim() < -axis or tensor.shape[axis] != n_token:
            logger.warning(
                "design masker: %s axis %d has size %s, expected %d tokens; "
                "skipping this axis",
                rule.key,
                axis,
                tensor.shape[axis] if tensor.dim() >= -axis else "n/a",
                n_token,
            )
            continue

        # Broadcast the token mask up to the tensor's rank by giving it a
        # trailing shape of 1s for every axis after the token axis.
        shape = [1] * tensor.dim()
        shape[axis] = n_token
        mask = design_mask.reshape(-1)[: n_token].reshape(shape)

        if rule.fill == UNK_ONEHOT:
            # Replace the whole one-hot row, not just scale it.
            onehot = torch.zeros(
                tensor.shape[-1], dtype=tensor.dtype, device=tensor.device
            )
            onehot[UNK_RESTYPE] = 1
            out = torch.where(mask.bool(), onehot.expand_as(out), out)
        else:
            out = torch.where(mask.bool(), _fill_value(rule.fill, tensor), out)

    return out


def apply_design_mask(
    feature_dict: dict[str, torch.Tensor],
    design_mask: Optional[torch.Tensor] = None,
    rules: Sequence[MaskRule] = DESIGN_MASK_RULES,
    inplace: bool = False,
) -> dict[str, torch.Tensor]:
    """Erase sequence-derived evidence about the designed residues.

    Parameters
    ----------
    feature_dict:
        Mol-1 input features.
    design_mask:
        Boolean ``[..., N_token]``. Inferred via :func:`infer_design_mask`
        when omitted.
    rules:
        Defaults to :data:`DESIGN_MASK_RULES`. Pass a subset to mask only the
        MSA or only the templates.
    inplace:
        Mutate ``feature_dict`` rather than returning a shallow copy. The
        copy is shallow by design -- untouched tensors are shared, and touched
        ones are replaced wholesale rather than written through.

    Returns
    -------
    The feature dict with the designed tokens blanked. A no-op when nothing is
    designed, which keeps pure folding batches on exactly the original path.
    """
    if design_mask is None:
        design_mask = infer_design_mask(feature_dict)
    if design_mask is None or not bool(design_mask.any()):
        return feature_dict

    out = feature_dict if inplace else dict(feature_dict)
    for rule in rules:
        tensor = out.get(rule.key)
        if tensor is None or not torch.is_tensor(tensor):
            continue  # feature not in use on this path (e.g. templates off)
        out[rule.key] = _apply_rule(tensor, rule, design_mask)
    return out


def assert_no_identity_leak(
    feature_dict: Mapping[str, torch.Tensor],
    design_mask: Optional[torch.Tensor] = None,
) -> None:
    """Raise if any designed token still carries its residue identity.

    Cheap enough to call in tests and in a training sanity check. It verifies
    the invariants the representation is supposed to guarantee, so it also
    catches a data pipeline that forgot to convert a residue to the designed
    component.
    """
    if design_mask is None:
        design_mask = infer_design_mask(feature_dict)
    if design_mask is None or not bool(design_mask.any()):
        return

    mask = design_mask.reshape(-1).bool()

    restype = feature_dict.get("restype")
    if restype is not None:
        ids = restype.argmax(dim=-1) if restype.dim() >= 2 else restype
        offenders = ids.reshape(-1)[mask]
        if offenders.numel() and not bool((offenders == UNK_RESTYPE).all()):
            raise AssertionError(
                f"designed tokens carry restype {sorted(set(offenders.tolist()))}, "
                f"expected all {UNK_RESTYPE} (UNK)"
            )

    for key in ("msa", "template_restype", "template_aatype"):
        tensor = feature_dict.get(key)
        if tensor is None or not torch.is_tensor(tensor):
            continue
        if tensor.shape[-1] != mask.shape[-1]:
            continue
        offenders = tensor[..., mask]
        if offenders.numel() and not bool((offenders == UNK_RESTYPE).all()):
            raise AssertionError(
                f"{key} still distinguishes designed tokens: "
                f"found ids {sorted(set(offenders.reshape(-1).tolist()))[:8]}"
            )
