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

"""LOCAL ADDITION: tests for the designed-residue input masker."""

import os
import unittest

import numpy as np
import torch

from mol.data.constants import STD_RESIDUES
from mol.data.core import ccd
from mol.data.core import design_residue as dr
from mol.model import design_masker as dm

NEEDS_CCD = unittest.skipUnless(
    os.path.isfile(ccd.COMPONENTS_FILE), "needs the CCD from the reference data"
)

N_TOKEN = 6
DESIGNED = [2, 4]  # matches the "AGXWXV" fixture used below
ALA, TRP = STD_RESIDUES["ALA"], STD_RESIDUES["TRP"]


def restype_onehot(ids):
    out = torch.zeros(len(ids), dm.RESTYPE_WIDTH)
    out[torch.arange(len(ids)), torch.tensor(ids)] = 1
    return out


def fake_features(n_msa=3):
    """A feature dict whose MSA still carries the native residues."""
    ids = [ALA, STD_RESIDUES["GLY"], dm.UNK_RESTYPE, TRP, dm.UNK_RESTYPE,
           STD_RESIDUES["VAL"]]
    native = [ALA, STD_RESIDUES["GLY"], TRP, TRP, ALA, STD_RESIDUES["VAL"]]
    return {
        "restype": restype_onehot(ids),
        # row 0 is the query; a real alignment would carry the native residue
        # at the designed columns, which is the leak this module closes.
        "msa": torch.tensor([native] * n_msa, dtype=torch.long),
        "has_deletion": torch.ones(n_msa, N_TOKEN),
        "deletion_value": torch.ones(n_msa, N_TOKEN),
        "profile": restype_onehot(native),
        "deletion_mean": torch.ones(N_TOKEN),
        "template_restype": torch.tensor([native], dtype=torch.long),
        "template_all_atom_mask": torch.ones(1, N_TOKEN, 37),
        "template_all_atom_positions": torch.ones(1, N_TOKEN, 37, 3),
        "template_distogram": torch.ones(1, N_TOKEN, N_TOKEN, 39),
        "template_unit_vector": torch.ones(1, N_TOKEN, N_TOKEN, 3),
    }


class VocabularyGuardTest(unittest.TestCase):
    def test_unk_is_mol1s_index_not_boltzgens(self):
        # BoltzGen's token_ids["UNK"] is 22 in a 33-wide vocabulary; in Mol-1
        # index 22 is RNA guanine. Transplanting that constant would be silent.
        self.assertEqual(dm.UNK_RESTYPE, 20)
        self.assertEqual(dm.RESTYPE_WIDTH, 32)
        self.assertNotEqual(dm.UNK_RESTYPE, 22)


class RuleTableTest(unittest.TestCase):
    def test_rule_keys_are_unique(self):
        keys = [r.key for r in dm.DESIGN_MASK_RULES]
        self.assertEqual(len(keys), len(set(keys)))

    def test_fills_are_known(self):
        for rule in dm.DESIGN_MASK_RULES:
            self.assertIn(rule.fill, (dm.ZERO, dm.UNK_ID, dm.UNK_ONEHOT))

    def test_token_axes_are_negative(self):
        # Negative indices keep the rules correct with or without a leading
        # batch dimension.
        for rule in dm.DESIGN_MASK_RULES:
            for axis in rule.token_axes:
                self.assertLess(axis, 0, rule.key)

    def test_no_atom_level_rules(self):
        # Per-atom reference features come from the same UNK component for
        # every designed residue, so there is nothing to erase. If that stops
        # being true this test should fail loudly.
        atom_keys = {"ref_pos", "ref_element", "ref_charge",
                     "ref_atom_name_chars", "ref_mask"}
        self.assertFalse(atom_keys & {r.key for r in dm.DESIGN_MASK_RULES})


class InferDesignMaskTest(unittest.TestCase):
    def test_infers_from_restype(self):
        mask = dm.infer_design_mask(fake_features())
        self.assertEqual(mask.nonzero().flatten().tolist(), DESIGNED)

    def test_explicit_design_mask_wins(self):
        feats = fake_features()
        feats["design_mask"] = torch.zeros(N_TOKEN, dtype=torch.bool)
        feats["design_mask"][0] = True
        self.assertEqual(dm.infer_design_mask(feats).nonzero().flatten().tolist(), [0])

    def test_returns_none_without_restype(self):
        self.assertIsNone(dm.infer_design_mask({}))

    def test_ligand_atoms_are_not_mistaken_for_designed_residues(self):
        """Ligands are encoded as "unknown amino acid" and atomized.

        Featurizer.restype_onehot_encoded represents every ligand as UNK, and
        a ligand becomes one token per atom, so a bare `restype == UNK` test
        flags each of its atoms as a designed residue. Caught by pulling a real
        training batch: designed tokens came back with 1 atom instead of 14.
        """
        feats = fake_features()
        # tokens 0 and 1 become a two-atom ligand, both UNK by restype.
        ids = [dm.UNK_RESTYPE, dm.UNK_RESTYPE, dm.UNK_RESTYPE, TRP,
               dm.UNK_RESTYPE, STD_RESIDUES["VAL"]]
        feats["restype"] = restype_onehot(ids)
        # one atom per token for this fixture; first two are the ligand
        feats["atom_to_token_idx"] = torch.arange(N_TOKEN)
        feats["is_protein"] = torch.tensor([0, 0, 1, 1, 1, 1])

        mask = dm.infer_design_mask(feats)
        self.assertEqual(mask.nonzero().flatten().tolist(), DESIGNED)

    def test_is_protein_dtype_does_not_matter(self):
        """The featurizer may hand over bool, int or float; all must agree."""
        import itertools

        results = []
        for dtype in (torch.bool, torch.int64, torch.float32):
            feats = fake_features()
            feats["restype"] = restype_onehot(
                [dm.UNK_RESTYPE, dm.UNK_RESTYPE, dm.UNK_RESTYPE, TRP,
                 dm.UNK_RESTYPE, STD_RESIDUES["VAL"]]
            )
            feats["atom_to_token_idx"] = torch.arange(N_TOKEN)
            feats["is_protein"] = torch.tensor([0, 0, 1, 1, 1, 1]).to(dtype)
            results.append(dm.infer_design_mask(feats).tolist())
        for a, b in itertools.pairwise(results):
            self.assertEqual(a, b)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_runs_on_cuda(self):
        """Guards a CPU/CUDA divergence that a CPU-only suite cannot see.

        The first GPU run died here with
          NotImplementedError: "cuda_scatter_gather_base_kernel_func"
                               not implemented for 'Bool'
        because the token-level protein mask was accumulated with
        scatter_reduce_ onto a bool tensor -- fine on CPU, unimplemented on
        CUDA. Every tensor op in this module needs to work on both.
        """
        feats = {
            k: v.cuda() for k, v in fake_features().items() if torch.is_tensor(v)
        }
        feats["atom_to_token_idx"] = torch.arange(N_TOKEN, device="cuda")
        feats["is_protein"] = torch.ones(N_TOKEN, dtype=torch.bool, device="cuda")

        mask = dm.infer_design_mask(feats)
        self.assertEqual(mask.device.type, "cuda")
        self.assertEqual(mask.nonzero().flatten().tolist(), DESIGNED)

        out = dm.apply_design_mask(feats)
        self.assertTrue((out["msa"][:, DESIGNED] == dm.UNK_RESTYPE).all())
        dm.assert_no_identity_leak(out)

    def test_falls_back_when_molecule_types_are_absent(self):
        feats = fake_features()  # no is_protein / atom_to_token_idx
        self.assertEqual(
            dm.infer_design_mask(feats).nonzero().flatten().tolist(), DESIGNED
        )


class ApplyDesignMaskTest(unittest.TestCase):
    def setUp(self):
        self.before = fake_features()
        self.after = dm.apply_design_mask(self.before)

    def test_msa_is_blanked_only_at_designed_columns(self):
        msa = self.after["msa"]
        self.assertTrue((msa[:, DESIGNED] == dm.UNK_RESTYPE).all())
        keep = [i for i in range(N_TOKEN) if i not in DESIGNED]
        torch.testing.assert_close(msa[:, keep], self.before["msa"][:, keep])

    def test_profile_becomes_a_unk_one_hot(self):
        profile = self.after["profile"]
        for i in DESIGNED:
            self.assertEqual(profile[i].argmax().item(), dm.UNK_RESTYPE)
            self.assertEqual(profile[i].sum().item(), 1.0)

    def test_deletion_features_are_zeroed(self):
        for key in ("has_deletion", "deletion_value"):
            self.assertTrue((self.after[key][:, DESIGNED] == 0).all(), key)
        self.assertTrue((self.after["deletion_mean"][DESIGNED] == 0).all())

    def test_template_rows_and_columns_are_zeroed(self):
        # Pairwise template features leak through either axis.
        for key in ("template_distogram", "template_unit_vector"):
            tensor = self.after[key]
            self.assertTrue((tensor[:, DESIGNED, :] == 0).all(), f"{key} rows")
            self.assertTrue((tensor[:, :, DESIGNED] == 0).all(), f"{key} cols")

    def test_template_per_token_features_are_zeroed(self):
        self.assertTrue((self.after["template_all_atom_mask"][:, DESIGNED] == 0).all())
        self.assertTrue(
            (self.after["template_all_atom_positions"][:, DESIGNED] == 0).all()
        )
        self.assertTrue(
            (self.after["template_restype"][:, DESIGNED] == dm.UNK_RESTYPE).all()
        )

    def test_input_is_not_mutated(self):
        torch.testing.assert_close(
            self.before["msa"], fake_features()["msa"]
        )

    def test_missing_features_are_skipped(self):
        feats = {"restype": fake_features()["restype"]}
        dm.apply_design_mask(feats)  # templates / MSA absent -> no crash

    def test_is_a_noop_when_nothing_is_designed(self):
        feats = fake_features()
        feats["restype"] = restype_onehot([ALA] * N_TOKEN)
        self.assertIs(dm.apply_design_mask(feats), feats)

    def test_rule_subset_masks_only_that_family(self):
        feats = fake_features()
        out = dm.apply_design_mask(feats, rules=dm.MSA_RULES)
        self.assertTrue((out["msa"][:, DESIGNED] == dm.UNK_RESTYPE).all())
        torch.testing.assert_close(
            out["template_distogram"], feats["template_distogram"]
        )

    def test_tolerates_a_leading_batch_dimension(self):
        feats = fake_features()
        batched = {
            "restype": feats["restype"].unsqueeze(0),
            "msa": feats["msa"].unsqueeze(0),
            "profile": feats["profile"].unsqueeze(0),
        }
        mask = torch.zeros(N_TOKEN, dtype=torch.bool)
        mask[DESIGNED] = True
        out = dm.apply_design_mask(batched, design_mask=mask)
        self.assertTrue((out["msa"][0][:, DESIGNED] == dm.UNK_RESTYPE).all())


class LeakAssertionTest(unittest.TestCase):
    def test_raises_on_an_unmasked_msa(self):
        with self.assertRaises(AssertionError) as caught:
            dm.assert_no_identity_leak(fake_features())
        self.assertIn("msa", str(caught.exception))

    def test_passes_after_masking(self):
        dm.assert_no_identity_leak(dm.apply_design_mask(fake_features()))

    def test_raises_when_a_designed_token_is_not_unk(self):
        # Catches a data pipeline that failed to convert a residue to the
        # designed component.
        feats = fake_features()
        mask = torch.zeros(N_TOKEN, dtype=torch.bool)
        mask[0] = True  # token 0 is ALA, not UNK
        with self.assertRaises(AssertionError) as caught:
            dm.assert_no_identity_leak(feats, design_mask=mask)
        self.assertIn("restype", str(caught.exception))

    def test_is_a_noop_when_nothing_is_designed(self):
        feats = fake_features()
        feats["restype"] = restype_onehot([ALA] * N_TOKEN)
        dm.assert_no_identity_leak(feats)


@NEEDS_CCD
class EndToEndTest(unittest.TestCase):
    """Real featurizer output, masked, then checked for leaks."""

    @staticmethod
    def featurize(sequence):
        from mol.data.inference.json_to_feature import SampleDictToFeatures
        from mol.data.utils import make_dummy_feature

        s2f = SampleDictToFeatures(
            {
                "name": "masker",
                "modelSeeds": [1],
                "sequences": [{"proteinChain": {"sequence": sequence, "count": 1}}],
            }
        )
        feature_dict, atom_array, _ = s2f.get_feature_dict()
        feature_dict = make_dummy_feature(feature_dict, dummy_feats=["msa"])
        return feature_dict, atom_array

    def test_dummy_msa_path_is_already_clean(self):
        # make_dummy_feature derives msa and profile FROM restype, which is
        # already UNK for a designed residue, so there is nothing to leak.
        feature_dict, _ = self.featurize("AGXWXV")
        dm.assert_no_identity_leak(feature_dict)

    def test_design_mask_matches_the_designed_residues(self):
        feature_dict, atom_array = self.featurize("AGXWXV")
        mask = dm.infer_design_mask(feature_dict)
        self.assertEqual(mask.sum().item(), 2)
        res_name = np.asarray(atom_array.res_name)
        atom_to_token = np.asarray(feature_dict["atom_to_token_idx"])
        designed_tokens = set(mask.nonzero().flatten().tolist())
        for token in designed_tokens:
            names = set(res_name[atom_to_token == token])
            self.assertEqual(names, {dr.DESIGN_RESIDUE})

    def test_masking_real_features_leaves_them_clean(self):
        feature_dict, _ = self.featurize("AGXWXV")
        n_token = feature_dict["restype"].shape[0]
        # Simulate a real alignment that knows the native CDR residues.
        feature_dict["msa"] = torch.full((4, n_token), TRP, dtype=torch.long)
        with self.assertRaises(AssertionError):
            dm.assert_no_identity_leak(feature_dict)
        dm.assert_no_identity_leak(dm.apply_design_mask(feature_dict))


if __name__ == "__main__":
    unittest.main()
