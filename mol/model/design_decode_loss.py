"""Differentiable surrogate for the geometric codebook's decode rule.

LOCAL ADDITION. Not part of upstream Mol-1 or AF3.

Why this exists
---------------
A designed residue's amino-acid identity is stored in *where* its surplus atom
slots sit: each is parked exactly on the residue's own backbone ``N`` or ``O``,
and the pair ``(count near N, count near O)`` is the codeword. Decoding counts
slots inside a hard 0.5 A radius.

Training, until now, supervised that only through the coordinate MSE. Two
measured consequences:

1. **The MSE gradient is weakest exactly where the decision is made.**
   ``d(x^2)/dx = 2x``, so the pull on a slot 3 A from its anchor is 6x the pull
   on one at 0.5 A -- and 0.5 A is the threshold at which the decoded residue
   flips. The loss is steepest where decoding does not care and flattest where
   it does.
2. **Under uncertainty, MSE asks for the conditional mean.** Weighted by real
   CDR composition, 87.9% of surplus-slot label mass sits on ``O``, so a model
   that does not yet know which residue it is placing minimises expected MSE by
   putting every slot on ``O``. Scoring xid 284 found exactly that: 77.6% of
   the slots that should park on N were inside O's capture zone, and 0.0% the
   other way round.

This term supervises the codeword itself. It replaces the hard radius with a
sigmoid and the nearest-atom argmin with a softmax, so the count becomes
differentiable, and regresses it against the count the label implies.

It does **not** replace the MSE. The sigmoid saturates a few Angstrom out, so
this term contributes almost nothing at long range -- MSE still does the
pulling. The two are complementary by construction: MSE for transport, this
for the decision.

Reading the temperature
-----------------------
``tau`` sets how far from the threshold the term can still feel. Small values
track the hard rule closely but vanish quickly, which matters more than it
sounds: a slot sitting *on the wrong anchor* needs gradient to leave a point
where the sigmoid is saturated at 1, and that derivative is ``s(1-s)/tau``.
At ``tau = 0.15`` it is 0.003; at ``tau = 0.5`` it is 0.39. The default is 0.5
for that reason, not because it approximates the decode rule best.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)

#: Slots per designed residue. Must match ``design_residue.N_DESIGN_SLOTS``;
#: imported lazily in ``forward`` so this module stays importable without the
#: data package.
_DEFAULT_SLOTS = 14

#: Indices of N, CA, C, O inside a designed residue's slot layout. Only N and
#: O can carry codeword mass; CA and C are candidates for the nearest-atom
#: search and are deliberately counted toward nothing.
_IDX_N, _IDX_CA, _IDX_C, _IDX_O = 0, 1, 2, 3


def gather_residue_slots(
    values: torch.Tensor,
    design_token_mask: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    atom_to_tokatom_idx: torch.Tensor,
    n_slots: int = _DEFAULT_SLOTS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape a per-atom quantity into ``[..., N_designed, n_slots, C]``.

    ``values`` is ``[..., N_atom, C]`` -- coordinates with ``C = 3``, or any
    per-atom scalar with ``C = 1``.

    Keyed off ``atom_to_tokatom_idx`` -- each atom's slot number within its own
    token -- rather than on atoms of a token being contiguous in the array.
    They are, today, but that is an invariant of the tokenizer and not one this
    function should inherit silently.

    Returns the gathered coordinates and a ``[N_designed]`` boolean saying
    which residues had all ``n_slots`` present. An incomplete residue cannot be
    decoded, so it must not contribute.
    """
    tok = atom_to_token_idx.reshape(-1).long()
    slot = atom_to_tokatom_idx.reshape(-1).long()
    design_flat = design_token_mask.reshape(-1).bool()

    designed = design_flat.nonzero().flatten()
    n_des = int(designed.numel())
    channels = values.shape[-1]
    if n_des == 0:
        empty = values.new_zeros(*values.shape[:-2], 0, n_slots, channels)
        return empty, torch.zeros(0, dtype=torch.bool, device=tok.device)

    # token id -> row in the gathered tensor, -1 for tokens we do not want
    rank = torch.full((design_flat.numel(),), -1, dtype=torch.long,
                      device=tok.device)
    rank[designed] = torch.arange(n_des, device=tok.device)

    keep_atom = design_flat[tok] & (slot < n_slots)
    rows = rank[tok[keep_atom]]
    cols = slot[keep_atom]

    out = values.new_zeros(*values.shape[:-2], n_des, n_slots, channels)
    out[..., rows, cols, :] = values[..., keep_atom, :]

    filled = torch.zeros(n_des, n_slots, dtype=torch.bool, device=tok.device)
    filled[rows, cols] = True
    return out, filled.all(dim=-1)


def codeword_counts(
    slots: torch.Tensor,
    threshold: float = 0.5,
    tau: Optional[float] = None,
) -> torch.Tensor:
    """Count surplus slots near N and near O, softly or exactly.

    ``tau is None`` reproduces ``design_residue.decode_residue`` exactly: hard
    argmin over the four backbone slots, hard 0.5 A cut. Use that on labels.

    With a ``tau``, the cut becomes ``sigmoid((threshold - d) / tau)`` and the
    argmin becomes ``softmax(-d / tau)``; their product is the differentiable
    stand-in for "inside the zone *and* nearest to this anchor". Use that on
    predictions.

    Returns ``[..., 2]`` -- counts on N and on O, in that order. CA and C are
    scored by the nearest-atom step and then dropped, exactly as the decoder
    drops them.
    """
    backbone = slots[..., :4, :]
    surplus = slots[..., 4:, :]
    # [..., n_surplus, 4]
    dist = torch.cdist(surplus, backbone)

    if tau is None:
        nearest = torch.zeros_like(dist).scatter_(
            -1, dist.argmin(dim=-1, keepdim=True), 1.0)
        inside = (dist <= threshold).to(dist.dtype)
    else:
        nearest = torch.softmax(-dist / tau, dim=-1)
        inside = torch.sigmoid((threshold - dist) / tau)

    per_anchor = (inside * nearest).sum(dim=-2)        # [..., 4]
    return torch.stack([per_anchor[..., _IDX_N], per_anchor[..., _IDX_O]],
                       dim=-1)


class SoftDecodeLoss(nn.Module):
    """Regress the predicted codeword onto the one the label implies.

    Returns ``(loss, metrics)``; the loss is 0.0 (and the metrics empty) on any
    batch with no complete designed residue, so the caller can add it
    unconditionally.
    """

    def __init__(
        self,
        tau: float = 0.5,
        threshold: float = 0.5,
        n_slots: int = _DEFAULT_SLOTS,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        self.tau = tau
        self.threshold = threshold
        self.n_slots = n_slots
        self.eps = eps

    def forward(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: torch.Tensor,
        design_token_mask: torch.Tensor,
        atom_to_token_idx: torch.Tensor,
        atom_to_tokatom_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            pred_coordinate: ``[..., N_sample, N_atom, 3]`` denoised coordinates.
            true_coordinate: ``[..., N_atom, 3]`` ground truth.
            coordinate_mask: ``[N_atom]`` or ``[..., N_atom]``, whether a label
                coordinate exists. A designed residue needs all of its slots.
            design_token_mask: ``[N_token]`` boolean.
            atom_to_token_idx: ``[N_atom]``.
            atom_to_tokatom_idx: ``[N_atom]`` slot number within the token.
        """
        zero = pred_coordinate.sum() * 0.0

        true_slots, complete = gather_residue_slots(
            true_coordinate, design_token_mask, atom_to_token_idx,
            atom_to_tokatom_idx, self.n_slots)
        if complete.numel() == 0 or not bool(complete.any()):
            return zero, {}

        # A residue with an unresolved slot has no codeword to learn from.
        # The mask may carry leading batch dims; collapse them, since a slot
        # unresolved in any of them is unusable in all of them.
        flat_mask = coordinate_mask.reshape(-1, coordinate_mask.shape[-1])
        flat_mask = flat_mask.amin(dim=0).to(true_coordinate.dtype)[:, None]
        mask_slots, _ = gather_residue_slots(
            flat_mask, design_token_mask, atom_to_token_idx,
            atom_to_tokatom_idx, self.n_slots)
        resolved = (mask_slots[..., 0] > 0).all(dim=-1)
        keep = complete & resolved
        if not bool(keep.any()):
            return zero, {}

        # Target: the label's codeword measured through the SAME kernel.
        #
        # Not the exact integer codeword. The soft kernel is biased -- real
        # sidechain atoms sit 1.5-3 A out and the sigmoid still gives them a
        # little weight, so on perfect coordinates the soft count misses the
        # integer by up to 3.3 at tau = 0.5. Regressing onto the integer would
        # therefore push the geometry *away* from the truth to chase a number
        # the kernel cannot produce. Passing the label through the same kernel
        # cancels that bias exactly: at pred == true the loss is 0 for any
        # tau, which frees tau to be chosen for its gradient rather than for
        # how closely it mimics the hard rule.
        true_kept = true_slots[..., keep, :, :]
        with torch.no_grad():
            target = codeword_counts(true_kept, self.threshold, tau=self.tau)
            exact = codeword_counts(true_kept, self.threshold, tau=None)

        pred_slots, _ = gather_residue_slots(
            pred_coordinate, design_token_mask, atom_to_token_idx,
            atom_to_tokatom_idx, self.n_slots)
        pred_kept = pred_slots[..., keep, :, :]
        soft = codeword_counts(pred_kept, self.threshold, tau=self.tau)

        err = soft - target
        loss = (err**2).sum(dim=-1).mean()

        with torch.no_grad():
            # Reported against the EXACT codeword, not the calibrated target:
            # the loss is a surrogate but the number a reader wants is "how
            # many slots short is this residue", in whole atoms.
            hard_err = codeword_counts(pred_kept, self.threshold, tau=None) - exact
            metrics = {
                "n_residues": torch.as_tensor(
                    float(int(keep.sum())), device=loss.device),
                "mae_on_n": hard_err[..., 0].abs().mean(),
                "mae_on_o": hard_err[..., 1].abs().mean(),
                # Signed, because the sign is the diagnosis: a model parking
                # everything on O shows a negative N error against a positive
                # O error of the same size.
                "bias_on_n": hard_err[..., 0].mean(),
                "bias_on_o": hard_err[..., 1].mean(),
                "exact_frac": (hard_err.abs().sum(dim=-1) == 0).to(
                    loss.dtype).mean(),
            }
        return loss, metrics
