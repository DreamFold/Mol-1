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

"""LOCAL ADDITION: the designed-residue fixed-slot layout.

``VocabularyUnchangedTest`` is the one that protects the released checkpoint:
widening ``UNK``'s layout must not touch the residue vocabulary, because
``restype`` width, the ``Featurizer.encoder`` continuity assert and the
``ELEMS`` offset are all derived from it and ``load_strict`` is True.
"""

import pytest
import os
import unittest

import numpy as np

from mol.data.constants import (
    ELEMS,
    RES_ATOMS_DICT,
    STD_RESIDUES,
    STD_RESIDUES_WITH_GAP,
)
from mol.data.core import ccd
from mol.data.core import design_residue as dr

NEEDS_CCD = unittest.skipUnless(
    os.path.isfile(ccd.COMPONENTS_FILE), "needs the CCD from the reference data"
)

#: Mirrors mol/model/modules/confidence.py's table sizing.
MAX_ATOMS_PER_TOKEN = 24


class VocabularyUnchangedTest(unittest.TestCase):
    """Guards against a change that would silently reject the checkpoint."""

    def test_restype_width_is_unchanged(self):
        self.assertEqual(len(STD_RESIDUES_WITH_GAP), 32)

    def test_encoder_continuity_assert_would_pass(self):
        # Featurizer.encoder: "Do not use discontinuous number".
        self.assertEqual(
            len(STD_RESIDUES_WITH_GAP),
            max(STD_RESIDUES_WITH_GAP.values()) + 1,
        )

    def test_element_token_offset_is_unchanged(self):
        # ELEMS is built as len(STD_RESIDUES) + idx, so a new residue would
        # shift every ligand element token away from the trained ids.
        self.assertEqual(len(STD_RESIDUES), 31)
        self.assertEqual(ELEMS["C"], 36)

    def test_design_residue_is_not_a_new_vocabulary_entry(self):
        self.assertIn(dr.DESIGN_RESIDUE, STD_RESIDUES)
        self.assertEqual(STD_RESIDUES[dr.DESIGN_RESIDUE], 20)


class ResAtomsDictTest(unittest.TestCase):
    def test_design_slots_are_registered_in_order(self):
        table = RES_ATOMS_DICT[dr.DESIGN_RESIDUE]
        for slot, name in enumerate(dr.design_slot_names()):
            self.assertEqual(table[name], slot, f"{name} is not at slot {slot}")

    def test_backbone_occupies_the_first_four_slots(self):
        self.assertEqual(dr.design_slot_names()[:4], ("N", "CA", "C", "O"))

    def test_oxt_is_kept_outside_the_design_layout(self):
        # A deposited C-terminal UNK must still resolve a tokatom_idx rather
        # than raising KeyError in add_tokatom_idx.
        table = RES_ATOMS_DICT[dr.DESIGN_RESIDUE]
        self.assertIn("OXT", table)
        self.assertGreaterEqual(table["OXT"], dr.N_DESIGN_SLOTS)

    def test_slot_indices_fit_the_confidence_head_tables(self):
        table = RES_ATOMS_DICT[dr.DESIGN_RESIDUE]
        self.assertLess(max(table.values()), MAX_ATOMS_PER_TOKEN)


@NEEDS_CCD
class ComponentAtomArrayTest(unittest.TestCase):
    def test_design_component_has_a_fixed_slot_count(self):
        comp = ccd.get_component_atom_array(dr.DESIGN_RESIDUE)
        self.assertIsNotNone(comp)
        self.assertEqual(len(comp), dr.N_DESIGN_SLOTS)
        self.assertEqual(tuple(comp.atom_name), dr.design_slot_names())

    def test_virtual_atoms_use_a_spare_element_slot(self):
        comp = ccd.get_component_atom_array(dr.DESIGN_RESIDUE)
        elements = list(comp.element)
        for element in elements[len(dr.BASE_ATOMS) :]:
            self.assertEqual(element, dr.VIRTUAL_ELEMENT)
        # Narrow biotite dtypes would truncate this to "UN" and break one-hot
        # encoding, so assert the full string survived.
        self.assertTrue(all(len(e) <= 16 for e in elements))

    def test_virtual_atoms_are_unbonded(self):
        comp = ccd.get_component_atom_array(dr.DESIGN_RESIDUE)
        if comp.bonds is None:
            self.skipTest("component carries no bonds")
        first_virtual = len(dr.BASE_ATOMS)
        for i, j, _ in comp.bonds.as_array():
            self.assertLess(i, first_virtual)
            self.assertLess(j, first_virtual)

    def test_other_residues_are_untouched(self):
        ala = ccd.get_component_atom_array("ALA")
        self.assertEqual(tuple(ala.atom_name), ("N", "CA", "C", "O", "CB"))

    def test_both_leaving_atom_paths_are_padded(self):
        # The polymer builder asks for keep_leaving_atoms=True and strips per
        # position, so padding only the False path left designed residues at
        # their native 6 atoms all the way through featurization.
        for keep in (False, True):
            with self.subTest(keep_leaving_atoms=keep):
                comp = ccd.get_component_atom_array(
                    dr.DESIGN_RESIDUE, keep_leaving_atoms=keep
                )
                self.assertEqual(len(comp), dr.N_DESIGN_SLOTS)
                self.assertNotIn("OXT", list(comp.atom_name))

    def test_design_component_has_no_leaving_groups(self):
        # remove_leaving_atoms does central_to_leaving_groups.get(...), so a
        # bare None would raise AttributeError.
        comp = ccd.get_component_atom_array(
            dr.DESIGN_RESIDUE, keep_leaving_atoms=True
        )
        self.assertEqual(comp.central_to_leaving_groups, {})


@NEEDS_CCD
class ReferenceInfoTest(unittest.TestCase):
    def test_every_design_slot_resolves_in_atom_map(self):
        # add_ref_feat_info does atom_map[atom_name] with a bare subscript, so
        # a missing slot name is a hard KeyError on any designed residue.
        info = ccd.get_ccd_ref_info(dr.DESIGN_RESIDUE)
        for name in dr.design_slot_names():
            self.assertIn(name, info["atom_map"])

    def test_virtual_slot_visibility_follows_the_switch(self):
        # ref_mask gates the per-atom embedding (c_l *= ref_mask). Which way
        # it is set is the whole VIRTUAL_REF_MASK question -- see the constant
        # for why BoltzGen sets it visible -- so assert the switch rather than
        # a fixed answer, or this test pins whichever arm happens to be the
        # default and silently fails the other.
        info = ccd.get_ccd_ref_info(dr.DESIGN_RESIDUE)
        mask = np.asarray(info["mask"])
        want = 1 if dr.VIRTUAL_REF_MASK else 0
        for name in dr.VIRTUAL_ATOM_NAMES:
            self.assertEqual(mask[info["atom_map"][name]], want, name)

    def test_real_slots_keep_their_reference_conformer(self):
        info = ccd.get_ccd_ref_info(dr.DESIGN_RESIDUE)
        mask = np.asarray(info["mask"])
        for name in dr.BASE_ATOMS:
            self.assertEqual(mask[info["atom_map"][name]], 1)

    def test_virtual_slots_are_parked_on_ca(self):
        info = ccd.get_ccd_ref_info(dr.DESIGN_RESIDUE)
        coord = np.asarray(info["coord"])
        ca = coord[info["atom_map"]["CA"]]
        for name in dr.VIRTUAL_ATOM_NAMES:
            np.testing.assert_allclose(coord[info["atom_map"][name]], ca)

    def test_permutation_is_forced_to_identity(self):
        # Eight indistinguishable disconnected dummy atoms would otherwise
        # generate permutations that reshuffle the identity code.
        info = ccd.get_ccd_ref_info(dr.DESIGN_RESIDUE)
        perm = info["perm"]
        self.assertEqual(perm.shape[1], 1)
        np.testing.assert_array_equal(perm[:, 0], np.arange(perm.shape[0]))

    def test_other_residues_keep_their_permutations(self):
        # ALA has a symmetric group; forcing identity everywhere would be a bug.
        perm = ccd.get_ccd_ref_info("ALA")["perm"]
        self.assertGreater(perm.shape[1], 1)

    def test_reference_arrays_agree_in_length(self):
        info = ccd.get_ccd_ref_info(dr.DESIGN_RESIDUE)
        n = len(info["atom_map"])
        for key in ("coord", "charge", "mask", "perm"):
            self.assertEqual(len(np.asarray(info[key])), n, key)


@NEEDS_CCD
class FeaturizationGateTest(unittest.TestCase):
    """End-to-end: do the fixed slots survive the whole featurizer?

    This is the Phase 0 go/no-go. Component-level checks are not enough -- the
    polymer builder rebuilds residues through its own leaving-atom path, and an
    earlier version of this change passed every component test while still
    producing 6-atom designed residues here.
    """

    @staticmethod
    def featurize(sequence):
        from mol.data.inference.json_to_feature import SampleDictToFeatures

        s2f = SampleDictToFeatures(
            {
                "name": "design-gate",
                "modelSeeds": [1],
                "sequences": [{"proteinChain": {"sequence": sequence, "count": 1}}],
            }
        )
        feature_dict, atom_array, _ = s2f.get_feature_dict()
        return feature_dict, atom_array

    def designed_residue_ids(self, atom_array):
        res_id = np.asarray(atom_array.res_id)
        res_name = np.asarray(atom_array.res_name)
        return sorted(
            {int(r) for r in res_id[res_name == dr.DESIGN_RESIDUE].tolist()}
        )

    def test_designed_residues_occupy_exactly_the_fixed_slots(self):
        feature_dict, atom_array = self.featurize("AGXWXV")
        res_id = np.asarray(atom_array.res_id)
        atom_name = np.asarray(atom_array.atom_name)
        designed = self.designed_residue_ids(atom_array)
        self.assertEqual(len(designed), 2)
        for rid in designed:
            mask = res_id == rid
            self.assertEqual(mask.sum(), dr.N_DESIGN_SLOTS)
            self.assertEqual(tuple(atom_name[mask]), dr.design_slot_names())

    def test_designed_residue_is_one_token_not_fourteen(self):
        # Atomization would mean res_name fell out of STD_RESIDUES.
        feature_dict, atom_array = self.featurize("AGXWXV")
        res_id = np.asarray(atom_array.res_id)
        atom_to_token = np.asarray(feature_dict["atom_to_token_idx"])
        restype = np.asarray(feature_dict["restype"]).argmax(-1)
        for rid in self.designed_residue_ids(atom_array):
            tokens = set(atom_to_token[res_id == rid].tolist())
            self.assertEqual(len(tokens), 1)
            self.assertEqual(restype[tokens.pop()], STD_RESIDUES["UNK"])

    def test_tokatom_idx_covers_every_slot(self):
        feature_dict, atom_array = self.featurize("AGXWXV")
        res_id = np.asarray(atom_array.res_id)
        tokatom = np.asarray(feature_dict["atom_to_tokatom_idx"])
        for rid in self.designed_residue_ids(atom_array):
            self.assertEqual(
                tokatom[res_id == rid].tolist(), list(range(dr.N_DESIGN_SLOTS))
            )

    def test_virtual_slots_are_gated_off_and_flagged(self):
        from mol.data.constants import get_all_elems

        feature_dict, atom_array = self.featurize("AGXWXV")
        res_id = np.asarray(atom_array.res_id)
        ref_mask = np.asarray(feature_dict["ref_mask"])
        ref_element = np.asarray(feature_dict["ref_element"]).argmax(-1)
        spare = get_all_elems().index(dr.VIRTUAL_ELEMENT)
        n_real = len(dr.BASE_ATOMS)
        for rid in self.designed_residue_ids(atom_array):
            mask = res_id == rid
            self.assertTrue(ref_mask[mask][:n_real].all())
            if dr.VIRTUAL_REF_MASK:
                self.assertTrue(ref_mask[mask][n_real:].all())
            else:
                self.assertFalse(ref_mask[mask][n_real:].any())
            # The distinct element is what tells a virtual slot apart from a
            # real one, and it must hold under BOTH settings -- when the slot
            # is visible it is the only thing distinguishing it.
            self.assertTrue((ref_element[mask][n_real:] == spare).all())

    def test_fixed_size_holds_at_both_termini(self):
        # The C-terminus is the case that breaks: the polymer builder keeps OXT
        # there, which would make a designed residue 15 atoms instead of 14.
        for sequence in ("XAG", "AGX", "X", "XX", "AXGXV"):
            with self.subTest(sequence=sequence):
                _, atom_array = self.featurize(sequence)
                res_id = np.asarray(atom_array.res_id)
                atom_name = np.asarray(atom_array.atom_name)
                designed = self.designed_residue_ids(atom_array)
                self.assertTrue(designed, f"no designed residue in {sequence}")
                for rid in designed:
                    mask = res_id == rid
                    self.assertEqual(mask.sum(), dr.N_DESIGN_SLOTS)
                    self.assertNotIn("OXT", list(atom_name[mask]))

    def test_real_residues_are_unaffected(self):
        _, atom_array = self.featurize("AGXWXV")
        res_id = np.asarray(atom_array.res_id)
        res_name = np.asarray(atom_array.res_name)
        atom_name = np.asarray(atom_array.atom_name)
        sizes = {}
        for rid in sorted({int(r) for r in res_id.tolist()}):
            mask = res_id == rid
            sizes[str(res_name[mask][0])] = int(mask.sum())
        self.assertEqual(sizes["ALA"], 5)
        self.assertEqual(sizes["GLY"], 4)
        self.assertEqual(sizes["TRP"], 14)  # natively fills the layout
        self.assertEqual(sizes["VAL"], 8)  # C-terminal, keeps its OXT
        trp_mask = res_name == "TRP"
        self.assertFalse(dr.is_extended(atom_name[trp_mask]))


if __name__ == "__main__":
    unittest.main()


class TestCodebookCaching:
    """The default codebook is cached and must be immutable.

    The cache hands every caller the SAME object. A mutable one would let any
    caller silently redefine the residue code for the whole process -- a bug
    that would surface as a wrong amino acid, far from its cause.
    """

    def test_default_is_cached(self):
        from mol.data.core.design_residue import build_codebook

        assert build_codebook() is build_codebook()

    def test_cached_result_is_read_only(self):
        from mol.data.core.design_residue import build_codebook

        cb = build_codebook()
        with pytest.raises(TypeError):
            cb["ALA"] = (("N",), (0, 0))
        with pytest.raises(AttributeError):
            cb.pop("ALA")

    def test_explicit_arguments_bypass_the_cache(self):
        """A caller with its own inventory must not get the default table."""
        from mol.data.constants import RES_ATOMS_DICT
        from mol.data.core.design_residue import STANDARD_AA, build_codebook

        fresh = build_codebook(RES_ATOMS_DICT, STANDARD_AA)
        assert fresh is not build_codebook()
        assert dict(fresh) == dict(build_codebook())

    def test_a_different_inventory_gives_a_different_table(self):
        """Guards against the cache leaking across inventories."""
        from mol.data.constants import RES_ATOMS_DICT
        from mol.data.core.design_residue import build_codebook

        subset = ["ALA", "GLY", "SER", "TRP"]
        small = build_codebook(RES_ATOMS_DICT, subset)
        assert sorted(small) == sorted(subset)
        assert len(build_codebook()) == 20

    def test_values_are_already_immutable(self):
        """The proxy only freezes the outer mapping; the values must be
        tuples, or they would still be mutable through it."""
        from mol.data.core.design_residue import build_codebook

        for atoms, codeword in build_codebook().values():
            assert isinstance(atoms, tuple)
            assert isinstance(codeword, tuple)


class TestCodebookStrategy(unittest.TestCase):
    """The two codeword allocation rules, and the guard against mixing them.

    The codeword IS the label geometry, so a dataset prepared under one rule
    and trained under the other is mislabelled in every designed residue and
    nothing downstream would notice. That is what the fingerprint is for.
    """

    def test_default_is_unchanged(self):
        # Flipping the shipped default would silently invalidate every
        # prepared dataset and checkpoint. It must be an explicit opt-in.
        self.assertEqual(dr.CODEBOOK_STRATEGY, "pure_o_first")
        self.assertEqual(dr.build_codebook(),
                         dr.build_codebook(strategy="pure_o_first"))

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            dr.build_codebook(strategy="whatever")

    def test_both_strategies_round_trip(self):
        for strategy in ("pure_o_first", "balanced"):
            cb = dr.build_codebook(strategy=strategy)
            self.assertEqual(len(cb), 20, strategy)
            self.assertEqual(len({c for _, c in cb.values()}), 20,
                             f"{strategy}: codewords collide")
            for res, (order, (on_n, on_o)) in cb.items():
                self.assertEqual(len(order) + on_n + on_o, dr.N_DESIGN_SLOTS,
                                 f"{strategy}/{res}")

    def test_balanced_empties_both_extreme_columns(self):
        cb = dr.build_codebook(strategy="balanced")
        for res, (_, (on_n, on_o)) in cb.items():
            if on_n + on_o == 0:
                self.assertEqual(res, "TRP")   # forced, nothing else can be
                continue
            self.assertGreater(on_n, 0, f"{res} is pure-O")
            self.assertGreater(on_o, 0, f"{res} is pure-N")

    def test_balanced_actually_balances(self):
        comp = dr.CDR_COMPOSITION
        def mass(cb):
            n = sum(comp.get(r, 0.0) * c[0] for r, (_, c) in cb.items())
            o = sum(comp.get(r, 0.0) * c[1] for r, (_, c) in cb.items())
            return o / (n + o)
        shipped = mass(dr.build_codebook(strategy="pure_o_first"))
        balanced = mass(dr.build_codebook(strategy="balanced"))
        self.assertGreater(shipped, 0.85)      # the problem: 87.9% on O
        self.assertLess(abs(balanced - 0.5), 0.05)

    def test_neither_degenerate_strategy_decodes_under_balanced(self):
        # A model that parks everything on one anchor must produce codewords
        # that are NOT in the table, so the failure is loud.
        cb = dr.build_codebook(strategy="balanced")
        by_code = {tuple(c): r for r, (_, c) in cb.items()}
        for collapse in (lambda k: (0, k), lambda k: (k, 0)):
            silent = [r for r, (_, c) in cb.items()
                      if by_code.get(collapse(c[0] + c[1])) not in (r, None)]
            self.assertEqual(silent, [], f"silently mis-decodes: {silent}")

    def test_fingerprint_separates_the_strategies(self):
        a = dr.codebook_fingerprint(dr.build_codebook(strategy="pure_o_first"))
        b = dr.codebook_fingerprint(dr.build_codebook(strategy="balanced"))
        self.assertNotEqual(a, b)
        self.assertEqual(len(a), 12)
        # stable across calls, so it can be written into prepared data
        self.assertEqual(
            a, dr.codebook_fingerprint(dr.build_codebook(strategy="pure_o_first")))

    def test_encode_decode_agree_within_a_strategy(self):
        coords = {"N": (0.0, 0.0, 0.0), "CA": (1.5, 0.0, 0.0),
                  "C": (2.4, 1.2, 0.0), "O": (2.0, 2.4, 0.0)}
        for strategy in ("pure_o_first", "balanced"):
            cb = dr.build_codebook(strategy=strategy)
            for res in dr.STANDARD_AA:
                c = dict(coords)
                for i, a in enumerate(cb[res][0]):
                    c.setdefault(a, (6.0 + 3.0 * i, 5.0, 5.0))
                slots = dr.encode_residue(res, c, cb)
                self.assertEqual(dr.decode_residue(slots, cb), res,
                                 f"{strategy}/{res}")

    def test_cross_strategy_decode_is_wrong(self):
        # The exact failure the fingerprint guards against, made concrete.
        coords = {"N": (0.0, 0.0, 0.0), "CA": (1.5, 0.0, 0.0),
                  "C": (2.4, 1.2, 0.0), "O": (2.0, 2.4, 0.0)}
        a = dr.build_codebook(strategy="pure_o_first")
        b = dr.build_codebook(strategy="balanced")
        c = dict(coords)
        for i, atom in enumerate(a["ASN"][0]):
            c.setdefault(atom, (6.0 + 3.0 * i, 5.0, 5.0))
        slots = dr.encode_residue("ASN", c, a)      # encoded one way
        self.assertNotEqual(dr.decode_residue(slots, b), "ASN")  # read another

    def test_stamp_guard_catches_a_mismatched_dataset(self):
        import json
        import tempfile
        from pathlib import Path

        d = Path(tempfile.mkdtemp())
        other = dr.codebook_fingerprint(dr.build_codebook(strategy="balanced"))
        (d / "codebook.json").write_text(
            json.dumps({"strategy": "balanced", "fingerprint": other}))
        with self.assertRaises(ValueError):
            dr.check_codebook_stamp(d)

        (d / "codebook.json").write_text(json.dumps(
            {"strategy": "pure_o_first",
             "fingerprint": dr.codebook_fingerprint()}))
        self.assertEqual(dr.check_codebook_stamp(d)["strategy"], "pure_o_first")

    def test_missing_stamp_warns_rather_than_raising(self):
        # Datasets built before stamping existed predate the only strategy
        # that was available, so they are not actually ambiguous.
        import tempfile
        from pathlib import Path

        self.assertIsNone(dr.check_codebook_stamp(Path(tempfile.mkdtemp())))
