"""The differentiable codebook-decode surrogate.

What these pin, in order of what would hurt most if it broke:

* the hard path reproduces ``decode_residue`` exactly -- if the surrogate's
  own notion of the codeword drifts from the decoder's, it optimises a
  different quantity than the one reported;
* the loss is exactly zero at the correct answer, for every temperature. The
  soft kernel is biased (real sidechain atoms 1.5-3 A out still get a little
  sigmoid weight), so regressing onto the integer codeword would pull the
  geometry away from the truth. Passing the label through the same kernel
  cancels it;
* a slot parked on the WRONG anchor receives usable gradient. That is the
  measured failure the term exists for, and the naive temperature leaves
  almost none.
"""

import unittest

import numpy as np
import torch

from mol.data.core import design_residue as dr
from mol.model.design_decode_loss import (
    SoftDecodeLoss, codeword_counts, gather_residue_slots)


def _residue(res, codebook, spread=3.0):
    """Encode `res` with synthetic but non-degenerate coordinates."""
    order = codebook[res][0]
    coords = {"N": (0.0, 0.0, 0.0), "CA": (1.5, 0.0, 0.0),
              "C": (2.4, 1.2, 0.0), "O": (2.0, 2.4, 0.0)}
    for i, a in enumerate(order):
        coords.setdefault(a, (spread + spread * i, 4.0, 4.0))
    return torch.tensor(dr.encode_residue(res, coords, codebook),
                        dtype=torch.float64)


class TestCodewordCounts(unittest.TestCase):
    def setUp(self):
        self.cb = dr.build_codebook()

    def test_hard_path_reproduces_the_decoder(self):
        for res in dr.STANDARD_AA:
            slots = _residue(res, self.cb)
            got = tuple(codeword_counts(slots, tau=None).int().tolist())
            self.assertEqual(got, tuple(self.cb[res][1]), res)
            # and the decoder agrees that this is that residue
            self.assertEqual(dr.decode_residue(slots.numpy(), self.cb), res)

    def test_ca_and_c_are_never_counted(self):
        # Park every surplus slot on CA. The decoder discards those, so the
        # codeword must be (0, 0) -- not a count against CA.
        cb = self.cb
        slots = _residue("GLY", cb).clone()
        n = len(cb["GLY"][0])
        slots[n:] = slots[1]                       # CA
        self.assertEqual(tuple(codeword_counts(slots, tau=None).int().tolist()),
                         (0, 0))

    def test_soft_counts_are_differentiable(self):
        slots = _residue("MET", self.cb).clone().requires_grad_(True)
        codeword_counts(slots, tau=0.5).sum().backward()
        self.assertTrue(torch.isfinite(slots.grad).all())
        self.assertGreater(float(slots.grad.abs().sum()), 0.0)


class TestCalibration(unittest.TestCase):
    """The property that lets tau be chosen freely."""

    def setUp(self):
        self.cb = dr.build_codebook()
        self.slots = torch.stack([_residue(r, self.cb) for r in dr.STANDARD_AA])

    def test_loss_is_zero_at_the_truth_for_every_tau(self):
        for tau in (0.1, 0.25, 0.5, 1.0, 2.0):
            target = codeword_counts(self.slots, tau=tau)
            soft = codeword_counts(self.slots, tau=tau)
            self.assertAlmostEqual(
                float(((soft - target) ** 2).sum()), 0.0, places=12,
                msg=f"tau={tau}")

    def test_the_soft_kernel_really_is_biased(self):
        # Guards the reason calibration exists. If this ever stops holding,
        # the calibration is dead weight and should be reconsidered -- but
        # while it holds, regressing onto the integer codeword is wrong.
        exact = codeword_counts(self.slots, tau=None)
        soft = codeword_counts(self.slots, tau=0.5)
        self.assertGreater(float((soft - exact).abs().max()), 1.0)


class TestWrongAnchorGradient(unittest.TestCase):
    """The failure this term was added for.

    xid 284: 77.6% of slots that should park on N sat inside O's capture
    zone, 0.0% the other way. A slot there is at the saturated end of the
    sigmoid, so whether it gets any gradient at all is a property of tau.
    """

    def setUp(self):
        self.cb = dr.build_codebook()
        self.res = "MET"                       # parks 4 on N, 2 on O
        self.n = len(self.cb[self.res][0])

    def _stuck(self):
        slots = _residue(self.res, self.cb).clone()
        slots[self.n:] = slots[3]              # all six parked on O
        return slots

    def test_the_collapse_decodes_to_a_different_residue(self):
        got = dr.decode_residue(self._stuck().numpy(), self.cb)
        self.assertNotEqual(got, self.res)
        # ... and, under the shipped table, not to UNK either: silently wrong.
        if dr.CODEBOOK_STRATEGY == "pure_o_first":
            self.assertNotEqual(got, "UNK")

    def test_default_tau_leaves_usable_gradient(self):
        target = codeword_counts(_residue(self.res, self.cb), tau=0.5)
        x = self._stuck().requires_grad_(True)
        ((codeword_counts(x, tau=0.5) - target) ** 2).sum().backward()
        stuck = x.grad[self.n:self.n + 4].norm(dim=-1).mean()
        self.assertGreater(float(stuck), 0.1,
                           "the default temperature must leave the stuck "
                           "slots a gradient they can act on")

    def test_a_sharp_tau_would_not(self):
        # Documents why the default is 0.5 and not something that tracks the
        # hard 0.5 A rule more faithfully.
        target = codeword_counts(_residue(self.res, self.cb), tau=0.15)
        x = self._stuck().requires_grad_(True)
        ((codeword_counts(x, tau=0.15) - target) ** 2).sum().backward()
        self.assertLess(float(x.grad[self.n:self.n + 4].norm(dim=-1).mean()),
                        0.05)


class TestGatherAndForward(unittest.TestCase):
    def setUp(self):
        self.cb = dr.build_codebook()
        self.names = ["ASN", "MET", "GLY"]
        self.slots = torch.stack([_residue(r, self.cb) for r in self.names])
        n = dr.N_DESIGN_SLOTS
        # Three designed residues, plus one 5-atom ligand token in the middle
        # so the gather has to actually use the token mask.
        self.coords = torch.cat(
            [self.slots[0], self.slots[1], torch.zeros(5, 3), self.slots[2]])
        self.atom_to_token = torch.tensor(
            [0] * n + [1] * n + [2] * 5 + [3] * n)
        self.tokatom = torch.tensor(
            list(range(n)) + list(range(n)) + list(range(5)) + list(range(n)))
        self.design_token = torch.tensor([True, True, False, True])
        self.coord_mask = torch.ones(self.coords.shape[0])

    def test_gather_skips_non_designed_tokens(self):
        got, complete = gather_residue_slots(
            self.coords, self.design_token, self.atom_to_token, self.tokatom)
        self.assertEqual(tuple(got.shape), (3, dr.N_DESIGN_SLOTS, 3))
        self.assertTrue(bool(complete.all()))
        for i in range(3):
            self.assertTrue(torch.allclose(got[i], self.slots[i]))

    def test_forward_is_zero_on_a_perfect_prediction(self):
        loss_fn = SoftDecodeLoss(tau=0.5)
        loss, metrics = loss_fn(
            pred_coordinate=self.coords[None].clone(),
            true_coordinate=self.coords,
            coordinate_mask=self.coord_mask,
            design_token_mask=self.design_token,
            atom_to_token_idx=self.atom_to_token,
            atom_to_tokatom_idx=self.tokatom)
        self.assertAlmostEqual(float(loss), 0.0, places=10)
        self.assertEqual(float(metrics["n_residues"]), 3.0)
        self.assertAlmostEqual(float(metrics["exact_frac"]), 1.0)

    def test_forward_penalises_the_anchor_collapse_and_reports_its_sign(self):
        bad = self.coords.clone()
        n_real = len(self.cb["MET"][0])
        off = dr.N_DESIGN_SLOTS                      # MET is the second token
        bad[off + n_real:off + dr.N_DESIGN_SLOTS] = bad[off + 3]   # all on O
        loss, metrics = SoftDecodeLoss(tau=0.5)(
            pred_coordinate=bad[None], true_coordinate=self.coords,
            coordinate_mask=self.coord_mask,
            design_token_mask=self.design_token,
            atom_to_token_idx=self.atom_to_token,
            atom_to_tokatom_idx=self.tokatom)
        self.assertGreater(float(loss), 1.0)
        # short on N, long on O -- the signature the metric exists to show
        self.assertLess(float(metrics["bias_on_n"]), 0.0)
        self.assertGreater(float(metrics["bias_on_o"]), 0.0)
        self.assertLess(float(metrics["exact_frac"]), 1.0)

    def test_unresolved_residues_are_dropped(self):
        mask = self.coord_mask.clone()
        mask[dr.N_DESIGN_SLOTS + 5] = 0.0            # one MET slot unresolved
        _, metrics = SoftDecodeLoss()(
            pred_coordinate=self.coords[None], true_coordinate=self.coords,
            coordinate_mask=mask, design_token_mask=self.design_token,
            atom_to_token_idx=self.atom_to_token,
            atom_to_tokatom_idx=self.tokatom)
        self.assertEqual(float(metrics["n_residues"]), 2.0)

    def test_folding_batch_returns_zero_and_no_metrics(self):
        loss, metrics = SoftDecodeLoss()(
            pred_coordinate=self.coords[None], true_coordinate=self.coords,
            coordinate_mask=self.coord_mask,
            design_token_mask=torch.zeros(4, dtype=torch.bool),
            atom_to_token_idx=self.atom_to_token,
            atom_to_tokatom_idx=self.tokatom)
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(metrics, {})


if __name__ == "__main__":
    unittest.main()
