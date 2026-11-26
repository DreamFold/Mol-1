"""Every copy of a designed chain must actually be converted.

The bug this guards against is silent and expensive. ``entity_poly_seq`` has
no per-chain granularity, so marking a position designed marks it for every
subchain of that entity. If only the annotated chain's atoms are rewritten,
the other copies end up with a sequence saying "UNK" and atom records saying
"TYR". Mol-1 keeps only the atoms the 14-slot UNK component has names for
(N/CA/C/O, sometimes CB/CG) and marks every virtual slot unresolved -- so the
residue is masked on the input side, counted on the loss side, and carries no
identity codeword to learn.

Measured before the fix, on ``design_data_mix``: 8EB2's two designed entities
own three subchains each, making two thirds of its designed residues phantoms;
affected structures showed designed-atom supervision of 0.38 rather than 1.00.
"""

from __future__ import annotations

import pytest

gemmi = pytest.importorskip("gemmi")

from scripts.prepare_design_data import expand_targets_to_entity_copies


class FakeEntity:
    def __init__(self, name, subchains):
        self.name = name
        self.subchains = subchains


class FakeStructure:
    def __init__(self, entities):
        self.entities = entities


def structure(*entities):
    return FakeStructure([FakeEntity(n, s) for n, s in entities])


class TestExpansion:
    def test_copies_of_the_annotated_chain_are_added(self):
        s = structure(("1", ["A"]), ("2", ["B", "E"]), ("3", ["C", "F"]))
        out = expand_targets_to_entity_copies(s, {"E": {50, 51}})
        assert out == {"B": {50, 51}, "E": {50, 51}}

    def test_three_copies(self):
        """8EB2's shape: entity 4 owns J, L and N; only J was annotated."""
        s = structure(("4", ["J", "L", "N"]))
        out = expand_targets_to_entity_copies(s, {"J": {52}})
        assert out == {"J": {52}, "L": {52}, "N": {52}}

    def test_single_copy_entity_is_unchanged(self):
        """7DAA's shape -- one chain per entity -- was never affected."""
        s = structure(("2", ["B"]), ("3", ["C"]))
        targets = {"B": {25}, "C": {101}}
        assert expand_targets_to_entity_copies(s, targets) == targets

    def test_unrelated_entities_untouched(self):
        s = structure(("1", ["A", "D"]), ("2", ["B", "E"]))
        out = expand_targets_to_entity_copies(s, {"B": {7}})
        assert "A" not in out and "D" not in out

    def test_two_entities_both_designed(self):
        """A Fab: heavy and light are separate entities, each with copies."""
        s = structure(("4", ["J", "L"]), ("5", ["K", "M"]))
        out = expand_targets_to_entity_copies(s, {"J": {52}, "K": {61}})
        assert out == {"J": {52}, "L": {52}, "K": {61}, "M": {61}}

    def test_union_when_several_copies_are_annotated_differently(self):
        """Copies are the same chain, so the designed set must agree.

        Taking the union rather than letting each copy keep its own set is the
        only choice consistent with a single entity sequence: the patched
        sequence is the union either way, so the atoms must be too.
        """
        s = structure(("2", ["B", "E"]))
        out = expand_targets_to_entity_copies(s, {"B": {1}, "E": {2}})
        assert out == {"B": {1, 2}, "E": {1, 2}}

    def test_no_targets_is_a_no_op(self):
        s = structure(("1", ["A", "D"]))
        assert expand_targets_to_entity_copies(s, {}) == {}

    def test_input_is_not_mutated(self):
        s = structure(("2", ["B", "E"]))
        targets = {"E": {50}}
        expand_targets_to_entity_copies(s, targets)
        assert targets == {"E": {50}}

    def test_target_on_a_chain_with_no_entity_survives(self):
        """Defensive: a target naming an unknown subchain must not vanish."""
        s = structure(("1", ["A"]))
        out = expand_targets_to_entity_copies(s, {"Z": {3}})
        assert out == {"Z": {3}}


class FakeResidue:
    def __init__(self, label_seq, codeable=True):
        self.label_seq = label_seq
        self.codeable = codeable


class FakeSubchain(list):
    def __init__(self, sid, residues):
        super().__init__(residues)
        self._sid = sid

    def subchain_id(self):
        return self._sid


class FakeChain(list):
    def subchains(self):
        return list(self)


class FakeModelStructure:
    def __init__(self, entities, chains):
        self.entities = [FakeEntity(n, s) for n, s in entities]
        self._chains = chains

    def __getitem__(self, i):
        return self._chains


def model_structure(entities, subchains):
    """subchains: {sid: [(label_seq, codeable), ...]}"""
    chain = FakeChain(
        FakeSubchain(sid, [FakeResidue(s, c) for s, c in residues])
        for sid, residues in subchains.items()
    )
    return FakeModelStructure(entities, [chain])


@pytest.fixture
def codebook():
    return object()  # only identity matters; encoding is monkeypatched


@pytest.fixture(autouse=True)
def fake_encoder(monkeypatch):
    """Stand in for encode_residue: a residue codes iff it says it does."""
    import scripts.prepare_design_data as mod

    monkeypatch.setattr(
        mod, "encoded_slots",
        lambda residue, codebook, stats: ["slot"] * 14 if residue.codeable else None,
    )


class TestConvertiblePerEntity:
    """Copies of one chain must agree on which positions are designed.

    A sidechain can be ordered in one Fab of the asymmetric unit and
    disordered in the next. The entity carries a single sequence for all its
    copies, so a position converted in only some of them leaves the others
    claiming UNK while holding native atoms -- the same inconsistency, one
    residue at a time.
    """

    def stats(self):
        from scripts.prepare_design_data import DesignStats

        return DesignStats()

    def call(self, entities, subchains, targets):
        from scripts.prepare_design_data import convertible_per_entity

        st = self.stats()
        out = convertible_per_entity(
            model_structure(entities, subchains), targets, object(), st
        )
        return out, st

    def test_intersection_across_copies(self):
        out, st = self.call(
            [("2", ["B", "E"])],
            {"B": [(1, True), (2, True)], "E": [(1, True), (2, False)]},
            {"B": {1, 2}, "E": {1, 2}},
        )
        assert out == {"B": {1}, "E": {1}}
        assert st.residues_dropped_copy_mismatch == 1

    def test_agreement_drops_nothing(self):
        out, st = self.call(
            [("2", ["B", "E"])],
            {"B": [(1, True), (2, True)], "E": [(1, True), (2, True)]},
            {"B": {1, 2}, "E": {1, 2}},
        )
        assert out == {"B": {1, 2}, "E": {1, 2}}
        assert st.residues_dropped_copy_mismatch == 0

    def test_single_copy_keeps_everything_codeable(self):
        out, st = self.call(
            [("2", ["B"])], {"B": [(1, True), (2, False)]}, {"B": {1, 2}}
        )
        assert out == {"B": {1}}
        assert st.residues_dropped_copy_mismatch == 0

    def test_entities_are_independent(self):
        out, _ = self.call(
            [("2", ["B", "E"]), ("3", ["C"])],
            {"B": [(1, True)], "E": [(1, False)], "C": [(9, True)]},
            {"B": {1}, "E": {1}, "C": {9}},
        )
        assert out == {"B": set(), "E": set(), "C": {9}}

    def test_three_copies_need_unanimity(self):
        out, st = self.call(
            [("4", ["J", "L", "N"])],
            {"J": [(5, True)], "L": [(5, True)], "N": [(5, False)]},
            {"J": {5}, "L": {5}, "N": {5}},
        )
        assert out == {"J": set(), "L": set(), "N": set()}
        assert st.residues_dropped_copy_mismatch == 2

    def test_uncodeable_residues_are_counted_once_per_copy(self):
        """The dry run is the only pass that sees them, so it must count."""
        import scripts.prepare_design_data as mod

        st = self.stats()
        # Undo the autouse monkeypatch for this one check by counting via a
        # stub that increments the real counter.
        def enc(residue, codebook, stats):
            if residue.codeable:
                return ["slot"] * 14
            stats.residues_skipped_unresolved += 1
            return None

        real = mod.encoded_slots
        mod.encoded_slots = enc
        try:
            mod.convertible_per_entity(
                model_structure(
                    [("2", ["B", "E"])],
                    {"B": [(1, False)], "E": [(1, False)]},
                ),
                {"B": {1}, "E": {1}},
                object(),
                st,
            )
        finally:
            mod.encoded_slots = real
        assert st.residues_skipped_unresolved == 2


class TestVariantIdentity:
    """Each variant must declare its own ``_entry.id``.

    ``prepare_training_data.py`` names its output from the *parsed*
    ``_entry.id`` (``mol/data/core/parser.py:165``), not from the file stem.
    Two variants both declaring ``1A4J`` therefore write to one
    ``1a4j.pkl.gz`` and the last worker to finish wins -- so a fold variant
    can overwrite a design variant and nothing reports it. On the first build
    this discarded every ``_v1`` file: 5270 CIFs became 847 bioassemblies, and
    37% of those had no designed residue.
    """

    def test_suffix_is_empty_for_the_first_variant(self):
        from scripts.prepare_design_data import suffix_for

        assert suffix_for(0) == ""
        assert suffix_for(1) == "_v1"
        assert suffix_for(2) == "_v2"

    def test_entry_id_tracks_the_filename(self, tmp_path):
        """The round trip that actually matters: write, reread, compare."""
        source = "/mnt/shared_storage/datasets/representative_ab/cifs/7DAA.cif"
        if not __import__("pathlib").Path(source).is_file():
            pytest.skip("representative_ab CIFs absent")

        st = gemmi.read_structure(source)
        st.setup_entities()
        st.name = "7DAA_V1"
        st.info["_entry.id"] = "7DAA_V1"
        out = tmp_path / "7daa_v1.cif"
        st.make_mmcif_document().write_file(str(out))

        block = gemmi.cif.read(str(out)).sole_block()
        # gemmi quotes an id containing an underscore; the parser reads it
        # through biotite, which unquotes. as_string does the same here.
        value = gemmi.cif.as_string(block.find_value("_entry.id"))
        assert value.lower() == "7daa_v1"

    def test_structure_name_alone_is_not_enough(self, tmp_path):
        """Guards the specific trap: .name renames the block, not the entry."""
        source = "/mnt/shared_storage/datasets/representative_ab/cifs/7DAA.cif"
        if not __import__("pathlib").Path(source).is_file():
            pytest.skip("representative_ab CIFs absent")

        st = gemmi.read_structure(source)
        st.setup_entities()
        st.name = "7DAA_V1"          # deliberately NOT setting info
        out = tmp_path / "x.cif"
        st.make_mmcif_document().write_file(str(out))
        block = gemmi.cif.read(str(out)).sole_block()
        assert block.name == "7DAA_V1"
        assert gemmi.cif.as_string(block.find_value("_entry.id")) == "7DAA"
