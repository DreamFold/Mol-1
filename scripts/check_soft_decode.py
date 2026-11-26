"""Integration gate for the soft-decode loss, on real prepared data.

The unit tests in ``tests/test_design_decode_loss.py`` build their tensors by
hand, so they prove the arithmetic but not the plumbing. Everything that
actually breaks lives in the gap between them and a real batch:

* is ``atom_to_tokatom_idx`` present in the feature dict at all;
* does ``infer_design_mask`` find the designed residues in prepared data;
* does the gather reassemble a residue's 14 slots in the right order, so that
  the codewords it reads back match the codebook;
* is the loss actually ~0 when the prediction IS the label.

No GPU and no model forward -- the label is fed in as the prediction, which is
the one input whose correct answer is known exactly. Cheap enough to run
before every training submission.

    python scripts/check_soft_decode.py --data-dir ... --indices ...
"""

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",
                    default="/mnt/shared_storage/.cache/mol1/design_data_v2")
    ap.add_argument("--indices", default="indices_overfit8.csv")
    ap.add_argument("--crop-size", type=int, default=512)
    ap.add_argument("--n-batches", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.5)
    args = ap.parse_args(argv)

    from mol.config.base import configs as base
    from mol.config.data import data_configs
    from mol.config.model_type import model_configs
    from mol.config.parser import parse_configs
    from mol.data.core import design_residue as dr
    from mol.data.pipeline.dataloader import get_dataloaders
    from mol.model.design_decode_loss import SoftDecodeLoss, codeword_counts, \
        gather_residue_slots
    from mol.model.design_masker import infer_design_mask

    print(f"codebook   : {dr.codebook_fingerprint()} ({dr.CODEBOOK_STRATEGY})")
    dr.check_codebook_stamp(args.data_dir, strict=True)

    def deep(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                deep(d[k], v)
            else:
                d[k] = v
        return d

    ts = "weightedPDB_before210930_v20260101"
    argv_cfg = [
        "--train_crop_size", str(args.crop_size), "--data.train_sets", ts,
        f"--data.{ts}.base_info.bioassembly_dict_dir", f"{args.data_dir}/bioassembly",
        f"--data.{ts}.base_info.indices_fpath", f"{args.data_dir}/{args.indices}",
        "--data.msa.enable_prot_msa", "false", "--data.msa.enable_rna_msa", "false",
        "--data.template.enable_prot_template", "false", "--data.test_sets", "''",
        "--eval_interval", "-1", "--seed", "0",
    ]
    cfg_dict = {**base, **{"data": data_configs}}
    deep(cfg_dict, model_configs["mol-1-base"])
    cfg = parse_configs(cfg_dict, " ".join(argv_cfg), fill_required_with_null=True)

    loader, _ = get_dataloaders(cfg, 1, 0)
    loss_fn = SoftDecodeLoss(tau=args.tau)
    codebook = dr.build_codebook()
    by_code = {tuple(c): r for r, (_, c) in codebook.items()}

    seen, failures = 0, []
    tally = collections.Counter()
    for batch in loader:
        if seen >= args.n_batches:
            break
        feat, label = batch["input_feature_dict"], batch["label_dict"]
        design = infer_design_mask(feat)
        if design is None or not bool(design.any()):
            continue
        seen += 1

        for key in ("atom_to_token_idx", "atom_to_tokatom_idx"):
            if feat.get(key) is None:
                failures.append(f"batch {seen}: {key} missing from feat_dict")
                continue

        coords = label["coordinate"]
        slots, complete = gather_residue_slots(
            coords, design, feat["atom_to_token_idx"],
            feat["atom_to_tokatom_idx"])

        # The gather is only right if the codewords it reads back are in the
        # table. A transposed or misordered slot axis would show up here as
        # UNK, not as an exception.
        exact = codeword_counts(slots[complete], tau=None).int()
        for row in exact.tolist():
            tally[by_code.get(tuple(row), "UNK")] += 1

        loss, metrics = loss_fn(
            pred_coordinate=coords[None], true_coordinate=coords,
            coordinate_mask=label["coordinate_mask"], design_token_mask=design,
            atom_to_token_idx=feat["atom_to_token_idx"],
            atom_to_tokatom_idx=feat["atom_to_tokatom_idx"])

        print(f"  batch {seen}: {int(design.sum())} designed tokens, "
              f"{int(complete.sum())}/{int(complete.numel())} complete, "
              f"n_scored={int(metrics['n_residues'])}, "
              f"loss(pred=label)={float(loss):.3e}, "
              f"exact_frac={float(metrics['exact_frac']):.4f}")
        if float(loss) > 1e-8:
            failures.append(f"batch {seen}: loss {float(loss):.3e} on an "
                            f"exact prediction, expected ~0")
        if float(metrics["exact_frac"]) < 1.0:
            failures.append(f"batch {seen}: exact_frac "
                            f"{float(metrics['exact_frac']):.4f} < 1.0 -- the "
                            f"label's own codewords do not decode")

        # And it must actually punish the failure we are trying to fix.
        bad = coords.clone()
        tok = feat["atom_to_token_idx"].reshape(-1).long()
        slot = feat["atom_to_tokatom_idx"].reshape(-1).long()
        is_des = design.reshape(-1).bool()[tok]
        # move every slot from index 4 on onto its residue's O
        o_of_atom = torch.zeros_like(coords)
        for t in design.reshape(-1).nonzero().flatten().tolist():
            idx = ((tok == t) & (slot == 3)).nonzero().flatten()
            if idx.numel():
                o_of_atom[(tok == t)] = coords[idx[0]]
        move = is_des & (slot >= 4)
        bad[move] = o_of_atom[move]
        bad_loss, bad_metrics = loss_fn(
            pred_coordinate=bad[None], true_coordinate=coords,
            coordinate_mask=label["coordinate_mask"], design_token_mask=design,
            atom_to_token_idx=feat["atom_to_token_idx"],
            atom_to_tokatom_idx=feat["atom_to_tokatom_idx"])
        print(f"            all-on-O collapse: loss={float(bad_loss):.3f}, "
              f"bias_on_n={float(bad_metrics['bias_on_n']):+.3f}, "
              f"bias_on_o={float(bad_metrics['bias_on_o']):+.3f}")
        if float(bad_loss) <= float(loss):
            failures.append(f"batch {seen}: the anchor collapse is not "
                            f"penalised ({float(bad_loss):.3e})")

    if seen == 0:
        print("FAIL: no batch contained a designed residue")
        return 1
    print(f"\nlabel codewords decoded over {sum(tally.values())} residues:")
    for res, n in tally.most_common():
        print(f"   {res:<5}{n:>6}")
    if tally.get("UNK"):
        failures.append(f"{tally['UNK']} label residues decode to UNK -- the "
                        f"gather or the codebook is wrong")

    print()
    for f in failures:
        print("FAIL:", f)
    print("PASS" if not failures else f"{len(failures)} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
