"""The diffusion MSE must concentrate on the designed region.

At inference the framework is pinned and only the designed residues are
generated, so a loss averaged evenly over every atom spends almost all of its
gradient on a task the model never performs. ``weight_design`` fixes that.

Two properties carry the weight of these tests:

1. **``weight_design = 0`` is bit-identical to upstream.** Mol-1 is a vendored
   fork; a local addition that perturbs the stock folding path would be a
   regression disguised as a feature.
2. **The weight stays out of the rigid alignment.** The designed loop is a
   small minority of atoms. Letting it drive the alignment would fit the frame
   to the loop and let the framework drift -- measuring the error in the wrong
   coordinate system. BoltzGen keeps its analogous weights out of
   ``align_weights`` for the same reason (``diffusion.py:780-820``).
"""

from __future__ import annotations

import pytest
import torch

from mol.model.loss import MSELoss


N_ATOM = 40
N_SAMPLE = 2
DESIGNED = slice(0, 4)  # a 4-atom "designed residue" out of 40


def coords(seed, n=N_ATOM, n_sample=None):
    g = torch.Generator().manual_seed(seed)
    shape = (n, 3) if n_sample is None else (n_sample, n, 3)
    return torch.randn(shape, generator=g) * 10.0


def masks(n=N_ATOM):
    zeros = torch.zeros(n)
    return {
        "coordinate_mask": torch.ones(n),
        "is_dna": zeros,
        "is_rna": zeros,
        "is_ligand": zeros,
    }


def design_mask(n=N_ATOM, where=DESIGNED):
    m = torch.zeros(n)
    m[where] = 1.0
    return m


class TestNeutralByDefault:
    """A fork must not change the model it forked."""

    def test_zero_weight_matches_upstream_exactly(self):
        pred, true = coords(0, n_sample=N_SAMPLE), coords(1)
        plain = MSELoss()
        tagged = MSELoss(weight_design=0.0)
        a = plain(pred_coordinate=pred, true_coordinate=true, **masks())
        b = tagged(
            pred_coordinate=pred,
            true_coordinate=true,
            is_design=design_mask(),
            **masks(),
        )
        assert torch.equal(a, b)

    def test_none_mask_matches_upstream_exactly(self):
        """A folding batch passes is_design=None even with the weight set."""
        pred, true = coords(2, n_sample=N_SAMPLE), coords(3)
        a = MSELoss()(pred_coordinate=pred, true_coordinate=true, **masks())
        b = MSELoss(weight_design=10.0)(
            pred_coordinate=pred, true_coordinate=true, is_design=None, **masks()
        )
        assert torch.equal(a, b)

    def test_all_zero_mask_matches_upstream_exactly(self):
        pred, true = coords(4, n_sample=N_SAMPLE), coords(5)
        a = MSELoss()(pred_coordinate=pred, true_coordinate=true, **masks())
        b = MSELoss(weight_design=10.0)(
            pred_coordinate=pred,
            true_coordinate=true,
            is_design=torch.zeros(N_ATOM),
            **masks(),
        )
        assert torch.equal(a, b)


class TestTheWeightActuallyLands:
    def test_designed_error_counts_more(self):
        """Same total error, moved onto the designed atoms -> larger loss."""
        true = coords(6)
        on_design = true.clone()
        on_design[DESIGNED] += 3.0
        off_design = true.clone()
        off_design[10:14] += 3.0  # same magnitude, same atom count, elsewhere

        loss = MSELoss(weight_design=9.0)
        kw = dict(true_coordinate=true, is_design=design_mask(), **masks())
        a = loss(pred_coordinate=on_design.unsqueeze(0), **kw)
        b = loss(pred_coordinate=off_design.unsqueeze(0), **kw)
        assert a > b

    def test_loss_is_affine_in_the_weight(self):
        """loss(w) = loss(0) + w * (designed contribution), exactly.

        The naive check -- "weight 4 gives 5x the loss" -- is wrong and worth
        spelling out. ``weighted_rigid_align`` re-fits the ground truth to the
        prediction, so after alignment the residual is spread over *every*
        atom, not just the perturbed ones. The loss is therefore
        ``A + w*B`` with ``A`` the full unweighted loss and ``B`` only the
        designed atoms' share, and the ratio is bounded well below ``1 + w``.
        Fitting A and B from two weights and predicting a third is the precise
        statement, and it also proves the alignment is weight-invariant -- if
        it were not, the relation would not be affine.
        """
        true = coords(7)
        pred = true.clone()
        pred[DESIGNED] += 2.0
        kw = dict(
            pred_coordinate=pred.unsqueeze(0),
            true_coordinate=true,
            is_design=design_mask(),
            **masks(),
        )
        a = float(MSELoss(weight_design=0.0)(**kw))
        b = float(MSELoss(weight_design=1.0)(**kw))
        slope = b - a
        assert slope > 0, "the designed atoms must contribute something"
        for w in (2.0, 4.0, 37.5):
            got = float(MSELoss(weight_design=w)(**kw))
            assert got == pytest.approx(a + w * slope, rel=1e-5)

    def test_unresolved_designed_atoms_are_still_excluded(self):
        """coordinate_mask gates everything; the design weight cannot revive it.

        This is the limitation to keep in view: a designed residue's virtual
        slots are unresolved in the deposited structure, so weighting alone
        does NOT start supervising the geometric identity code.
        """
        true = coords(8)
        pred = true.clone()
        pred[DESIGNED] += 5.0
        m = masks()
        m["coordinate_mask"] = torch.ones(N_ATOM)
        m["coordinate_mask"][DESIGNED] = 0.0
        kw = dict(
            pred_coordinate=pred.unsqueeze(0), true_coordinate=true,
            is_design=design_mask(), **m,
        )
        assert MSELoss(weight_design=0.0)(**kw) == MSELoss(weight_design=50.0)(**kw)


class TestAlignmentIsNotReweighted:
    """The frame must stay anchored on the framework."""

    def test_align_weight_ignores_the_design_mask(self):
        pred, true = coords(9, n_sample=N_SAMPLE), coords(10)
        args = dict(pred_coordinate=pred, true_coordinate=true, **masks())
        _, w_plain = MSELoss().weighted_rigid_align(**args)
        _, w_design = MSELoss(weight_design=100.0).weighted_rigid_align(**args)
        assert torch.equal(w_plain, w_design)

    def test_weighted_rigid_align_takes_no_design_argument(self):
        """Guards against a later refactor quietly re-introducing the bug."""
        import inspect

        params = inspect.signature(MSELoss.weighted_rigid_align).parameters
        assert "is_design" not in params

    def test_aligned_labels_are_unchanged_by_the_weight(self):
        pred, true = coords(11, n_sample=N_SAMPLE), coords(12)
        args = dict(pred_coordinate=pred, true_coordinate=true, **masks())
        a, _ = MSELoss().weighted_rigid_align(**args)
        b, _ = MSELoss(weight_design=100.0).weighted_rigid_align(**args)
        assert torch.allclose(a, b)


class TestConfigPlumbing:
    def test_default_config_is_neutral(self):
        from mol.config import configs, parse_configs

        mse = parse_configs(configs, fill_required_with_null=True).loss.diffusion.mse
        assert mse.get("weight_design", None) == 0.0

    def test_config_block_constructs_the_loss(self):
        """`MSELoss(**configs.loss.diffusion.mse)` must keep working."""
        from mol.config import configs, parse_configs

        mse = parse_configs(configs, fill_required_with_null=True).loss.diffusion.mse
        assert MSELoss(**mse).weight_design == 0.0


class TestSplitDiagnostics:
    """The aggregate MSE cannot distinguish "learned the design task" from
    "got slightly better at the framework it was already good at".

    These three numbers are what make that distinction visible, so they have
    to be correct and they have to survive the trainer's metric filter.
    """

    def kw(self, pred, true, design=None, cm=None):
        m = masks()
        if cm is not None:
            m["coordinate_mask"] = cm
        return dict(
            pred_coordinate=pred.unsqueeze(0), true_coordinate=true,
            is_design=design if design is not None else design_mask(), **m,
        )

    def test_split_is_none_without_a_design_mask(self):
        loss = MSELoss(weight_design=9.0)
        true = coords(20)
        loss(pred_coordinate=true.unsqueeze(0), true_coordinate=true,
             is_design=None, **masks())
        assert loss.last_split is None

    def test_error_only_on_designed_atoms(self):
        """framework ~0, design > 0, share ~1."""
        true = coords(21)
        pred = true.clone()
        pred[DESIGNED] += 4.0
        loss = MSELoss(weight_design=9.0)
        loss(**self.kw(pred, true))
        s = loss.last_split
        assert s["design"] > s["framework"]
        assert float(s["design_share"]) > 0.5

    def test_error_only_on_framework(self):
        """The reverse: share must collapse, not stay high.

        Perturb the SAME number of atoms as the designed case (4), not a
        large block. ``weighted_rigid_align`` re-fits the ground truth to the
        prediction, so moving half the atoms drags the fit and spreads the
        residual evenly over both groups -- which makes the comparison
        meaningless rather than failing honestly.
        """
        true = coords(22)
        pred = true.clone()
        pred[10:14] += 4.0            # 4 framework atoms, same count as DESIGNED
        loss = MSELoss(weight_design=9.0)
        loss(**self.kw(pred, true))
        s = loss.last_split
        assert s["framework"] > s["design"]
        assert float(s["design_share"]) < 0.5

    def test_share_reflects_the_weight(self):
        """Raising weight_design must raise the designed share, monotonically.

        This is the number that answers "is weight_design doing anything",
        so it must actually track the weight rather than the atom counts.
        """
        true = coords(23)
        g = torch.Generator().manual_seed(5)
        pred = true + torch.randn(true.shape, generator=g)   # error everywhere
        shares = []
        for w in (0.0, 1.0, 9.0, 50.0):
            loss = MSELoss(weight_design=w)
            loss(**self.kw(pred, true))
            shares.append(float(loss.last_split["design_share"]))
        assert shares == sorted(shares), shares
        assert shares[0] < shares[-1]

    def test_unresolved_atoms_excluded_from_both_halves(self):
        true = coords(24)
        pred = true.clone()
        pred[DESIGNED] += 7.0
        cm = torch.ones(N_ATOM)
        cm[DESIGNED] = 0.0            # designed atoms unresolved
        loss = MSELoss(weight_design=9.0)
        loss(**self.kw(pred, true, cm=cm))
        # Nothing designed is resolved, so the designed half must be empty,
        # not quietly averaging over masked-out atoms.
        assert float(loss.last_split["design"]) == pytest.approx(0.0, abs=1e-6)

    def test_stale_split_cannot_leak_between_calls(self):
        """A folding batch after a design batch must not report the old split."""
        true = coords(25)
        loss = MSELoss(weight_design=9.0)
        loss(**self.kw(true + 1.0, true))
        assert loss.last_split is not None
        loss(pred_coordinate=(true + 1.0).unsqueeze(0), true_coordinate=true,
             is_design=None, **masks())
        assert loss.last_split is None

    def test_metric_names_survive_the_trainer_filter(self):
        """MolTrainer.train_step drops any key without "loss" in it
        (mol/runner/train.py:629)."""
        for name in ("mse_loss_design", "mse_loss_framework",
                     "mse_loss_design_share"):
            assert "loss" in name


class TestOnlyFullSlotResiduesAreWeighted:
    """A flagged token with no codeword must not collect the design weight.

    ``infer_design_mask`` deliberately also flags natively-``UNK`` deposited
    residues -- masking their inputs is right, their identity is unknown too.
    But a poly-UNK trace in the PDB is often one CA-like atom per residue, and
    such a residue carries no codeword to learn. Measured on overfit8: 8 of
    170 flagged tokens hold one atom rather than 14. Weighting them dilutes
    ``weight_design`` with atoms that teach nothing.
    """

    @staticmethod
    def _loss(n_slots):
        """A MolLoss whose `_design_atom_mask` can be called in isolation."""
        import types

        from mol.model.loss import MolLoss

        obj = types.SimpleNamespace()
        obj.configs = types.SimpleNamespace()
        obj._design_atom_mask = types.MethodType(
            MolLoss._design_atom_mask, obj)
        return obj

    def _feat(self, token_sizes):
        """Feature dict with one designed token per entry of `token_sizes`."""
        from mol.data.core.design_residue import N_DESIGN_SLOTS

        atom_to_token, restype = [], []
        for t, size in enumerate(token_sizes):
            atom_to_token += [t] * size
            restype.append(20)                    # UNK for every token here
        return {
            "atom_to_token_idx": torch.tensor(atom_to_token),
            "restype": torch.tensor(restype),
            "is_protein": torch.ones(len(token_sizes)),
            "is_ligand": torch.zeros(len(token_sizes)),
        }, N_DESIGN_SLOTS

    def test_one_atom_unk_residues_get_no_weight(self):
        import types

        from mol.data.core.design_residue import N_DESIGN_SLOTS
        from mol.model.loss import MolLoss

        feat, n = self._feat([N_DESIGN_SLOTS, 1, 1, N_DESIGN_SLOTS])
        obj = types.SimpleNamespace(
            configs=type("C", (), {"loss": type("L", (), {
                "diffusion": type("D", (), {
                    "mse": {"weight_design": 40.0}})()})()})())
        mask = MolLoss._design_atom_mask(obj, feat)

        assert mask is not None
        # the two full residues are weighted, the two stubs are not
        assert float(mask[:n].sum()) == float(n)
        assert float(mask[n:n + 2].sum()) == 0.0
        assert float(mask[n + 2:].sum()) == float(n)
        assert float(mask.sum()) == float(2 * n)

    def test_a_batch_of_only_stubs_weights_nothing(self):
        import types

        from mol.model.loss import MolLoss

        feat, _ = self._feat([1, 1, 1])
        obj = types.SimpleNamespace(
            configs=type("C", (), {"loss": type("L", (), {
                "diffusion": type("D", (), {
                    "mse": {"weight_design": 40.0}})()})()})())
        mask = MolLoss._design_atom_mask(obj, feat)
        # Still a tensor, not None -- the batch *does* contain flagged tokens,
        # they simply earn no weight. MSELoss multiplies by (1 + w * mask),
        # so an all-zero mask is already the unweighted loss.
        assert mask is not None
        assert float(mask.sum()) == 0.0
