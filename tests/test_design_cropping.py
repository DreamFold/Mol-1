"""The crop must contain the region being designed.

Mol-1 crops to a fixed token budget by taking the k nearest tokens to one
randomly chosen reference token. On an antibody-antigen complex (median 1232
tokens) a 512-token crop keeps ~42% of the structure, and an unconstrained
centre lands in the designed CDRs roughly as often as the CDRs are large --
a few percent. The other ~95% of the time the designed residues are cropped
away and the example silently degrades into plain folding, which is precisely
the failure these tests exist to prevent.

Everything here runs on synthetic arrays rather than a prepared CIF, so the
test suite does not depend on /mnt/shared_storage.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray

from mol.data.core.design_residue import DESIGN_RESIDUE
from mol.data.tokenizer import TokenArray
from mol.utils.cropping import CropData, get_spatial_crop_index


# Two chains, 60 residues each, one CA per residue (one token per residue).
# Chain 0 is the "antibody": residues 20-29 are designed.
# Chain 1 is the "antigen", placed so that it contacts the designed stretch.
N_PER_CHAIN = 60
DESIGN_SLICE = slice(20, 30)


def make_arrays(designed=True, n_per_chain=N_PER_CHAIN):
    n = 2 * n_per_chain
    arr = AtomArray(n)
    arr.coord = np.zeros((n, 3), dtype=np.float32)
    # Chain 0 along x; chain 1 parallel, 8 A away in y and offset so that its
    # nearest residues face the designed stretch.
    arr.coord[:n_per_chain, 0] = np.arange(n_per_chain) * 3.8
    arr.coord[n_per_chain:, 0] = np.arange(n_per_chain) * 3.8
    arr.coord[n_per_chain:, 1] = 8.0
    arr.atom_name = np.array(["CA"] * n)
    arr.element = np.array(["C"] * n)
    names = np.array(["ALA"] * n, dtype="<U5")
    if designed:
        names[DESIGN_SLICE] = DESIGN_RESIDUE
    arr.res_name = names
    arr.res_id = np.concatenate([np.arange(n_per_chain)] * 2) + 1
    arr.set_annotation("asym_id_int", np.repeat([0, 1], n_per_chain))
    arr.set_annotation("is_ligand", np.zeros(n, dtype=bool))
    arr.set_annotation("is_resolved", np.ones(n, dtype=bool))
    arr.set_annotation("ref_space_uid", np.arange(n))

    return _token_array(n), arr


def _token_array(n):
    """One token per atom -- the standard-residue case, one CA each."""
    from mol.data.tokenizer import Token

    return TokenArray([
        Token(i, atom_indices=[i], centre_atom_index=i) for i in range(n)
    ])


def crop(designed=True, crop_size=40, ref_chains=(0,), weights=(0.2, 0.4, 0.4)):
    tokens, arr = make_arrays(designed=designed)
    return CropData(
        crop_size=crop_size,
        ref_chain_indices=list(ref_chains),
        token_array=tokens,
        atom_array=arr,
        method_weights=list(weights),
    )


class TestDesignTokenMask:
    def test_finds_the_designed_residues(self):
        mask = crop().design_token_mask()
        assert mask is not None
        assert mask.nonzero().flatten().tolist() == list(range(20, 30))

    def test_none_on_a_folding_example(self):
        assert crop(designed=False).design_token_mask() is None

    def test_ligand_unk_is_not_a_designed_residue(self):
        """Mol-1 encodes every ligand atom as a one-atom UNK token.

        Without the is_ligand guard a structure with a ligand would scatter
        false anchors across it and the crop would centre on the ligand.
        """
        tokens, arr = make_arrays(designed=False)
        arr.res_name[0:5] = DESIGN_RESIDUE
        arr.set_annotation("is_ligand", np.isin(np.arange(len(arr)), range(5)))
        c = CropData(40, [0], tokens, arr)
        assert c.design_token_mask() is None

    def test_cached(self):
        c = crop()
        assert c.design_token_mask() is c.design_token_mask()


class TestCropMethodSelection:
    def test_contiguous_is_dropped_for_a_design_example(self):
        """It takes a sequence window and cannot be steered at the CDRs."""
        c = crop(weights=(1.0, 0.0, 0.0001))
        picked = {c.random_crop_method() for _ in range(200)}
        assert "ContiguousCropping" not in picked

    def test_contiguous_survives_on_a_folding_example(self):
        c = crop(designed=False, weights=(1.0, 0.0, 0.0))
        assert {c.random_crop_method() for _ in range(20)} == {"ContiguousCropping"}

    def test_contiguous_only_request_is_honoured(self):
        """Zeroing the only non-zero weight would make random.choices raise."""
        c = crop(weights=(1.0, 0.0, 0.0))
        assert c.random_crop_method() == "ContiguousCropping"


class TestReferenceChains:
    def test_designed_chain_is_added_when_absent(self):
        """The sampler often picks the antigen as the reference chain."""
        c = crop(ref_chains=(1,))
        assert set(c.ref_chain_indices) == {0, 1}

    def test_left_alone_when_already_present(self):
        c = crop(ref_chains=(0,))
        assert c.ref_chain_indices == [0]

    def test_left_alone_on_a_folding_example(self):
        c = crop(designed=False, ref_chains=(1,))
        assert c.ref_chain_indices == [1]


class TestTheCropContainsTheDesign:
    @pytest.mark.parametrize("method", ["SpatialCropping", "SpatialInterfaceCropping"])
    @pytest.mark.parametrize("ref", [(0,), (1,)])
    def test_every_crop_keeps_designed_residues(self, method, ref):
        designed = set(range(20, 30))
        for seed in range(40):
            torch.manual_seed(seed)
            c = crop(ref_chains=ref)
            selected, _ = c.get_crop_indices(crop_method=method)
            assert designed & set(selected.tolist()), (
                f"{method} ref={ref} seed={seed} dropped the designed region"
            )

    def test_without_the_anchor_most_crops_miss_it(self):
        """The baseline this fixes -- otherwise the tests above prove nothing."""
        designed = set(range(20, 30))
        hits = 0
        for seed in range(40):
            torch.manual_seed(seed)
            c = crop(designed=False)        # no design -> no anchoring
            selected, _ = c.get_crop_indices(crop_method="SpatialCropping")
            hits += bool(designed & set(selected.tolist()))
        assert hits < 30, f"{hits}/40 -- the unanchored baseline is not a baseline"

    def test_crop_size_is_respected(self):
        torch.manual_seed(0)
        selected, _ = crop(crop_size=40).get_crop_indices(
            crop_method="SpatialInterfaceCropping"
        )
        assert selected.shape[0] <= 40


class TestAnchorMaskPrimitive:
    def make(self, n=10):
        d = torch.cdist(torch.arange(n).float()[:, None], torch.arange(n).float()[:, None])
        return d, torch.ones(n, n)

    def test_centre_is_drawn_from_the_anchor(self):
        d, m = self.make()
        anchor = torch.zeros(10, dtype=torch.bool)
        anchor[7] = True
        for seed in range(20):
            torch.manual_seed(seed)
            _, ref = get_spatial_crop_index(
                tokens=torch.arange(10), chain_id=torch.zeros(10),
                token_distance=d, token_distance_mask=m,
                reference_chain_id=torch.tensor([0]),
                ref_space_uid_token=torch.arange(10).float(),
                crop_size=3, anchor_mask=anchor,
            )
            assert ref == 7

    def test_empty_anchor_falls_back_and_warns(self, caplog):
        d, m = self.make()
        torch.manual_seed(0)
        selected, _ = get_spatial_crop_index(
            tokens=torch.arange(10), chain_id=torch.zeros(10),
            token_distance=d, token_distance_mask=m,
            reference_chain_id=torch.tensor([0]),
            ref_space_uid_token=torch.arange(10).float(),
            crop_size=3, anchor_mask=torch.zeros(10, dtype=torch.bool),
        )
        assert selected.shape[0] == 3
        assert "no designed token" in caplog.text

    def test_none_anchor_is_the_stock_path(self):
        d, m = self.make()
        kw = dict(
            tokens=torch.arange(10), chain_id=torch.zeros(10),
            token_distance=d, token_distance_mask=m,
            reference_chain_id=torch.tensor([0]),
            ref_space_uid_token=torch.arange(10).float(), crop_size=3,
        )
        torch.manual_seed(1); a = get_spatial_crop_index(**kw)
        torch.manual_seed(1); b = get_spatial_crop_index(anchor_mask=None, **kw)
        assert torch.equal(a[0], b[0]) and a[1] == b[1]
