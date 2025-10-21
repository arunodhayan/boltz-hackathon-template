#!/usr/bin/env python3
# ==============================================================
# 🧬 Boltz-2 Hackathon Full Pipeline (PLDDT-based Postprocess)
# ==============================================================

import os
import json
import random
import pickle
import urllib.request
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import List, Optional, Literal
import torch
import click
from tqdm import tqdm
from rdkit import Chem
from Bio.PDB import PDBParser, PDBIO
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# --- Boltz2 imports ---
from boltz.data.parse.yaml import parse_yaml
from boltz.data.parse.csv import parse_csv
from boltz.data.types import Manifest
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.write.writer import BoltzWriter
from boltz.model.models import Boltz2

# ==============================================================
# Utilities
# ==============================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

# ==============================================================
# Boltz Resources
# ==============================================================
MOLS_URL = "https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar"
MODEL_URL = "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt"

@rank_zero_only
def download_resources(cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    mols_tar = cache / "mols.tar"
    mols_dir = cache / "mols"
    ckpt = cache / "boltz2_conf.ckpt"

    # --- Molecules ---
    if not mols_dir.exists() or not any(mols_dir.glob("*.pkl")):
        if not mols_tar.exists():
            click.echo("📦 Downloading mols.tar ...")
            urllib.request.urlretrieve(MOLS_URL, str(mols_tar))
        import tarfile
        click.echo(f"📂 Extracting {mols_tar} → {mols_dir}")
        mols_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(mols_tar, "r") as tar:
            tar.extractall(mols_dir)
        nested = mols_dir / "mols"
        if nested.exists():
            import shutil
            click.echo("⚙️ Fixing nested mols/ structure ...")
            for f in nested.glob("*"):
                shutil.move(str(f), str(mols_dir))
            shutil.rmtree(nested)
        click.echo(f"✅ Molecule library ready ({len(list(mols_dir.glob('*.pkl')))} entries).")
    else:
        click.echo("✅ Molecule library already exists.")

    # --- Model ---
    if not ckpt.exists():
        click.echo("📥 Downloading Boltz-2 checkpoint ...")
        urllib.request.urlretrieve(MODEL_URL, str(ckpt))
    click.echo("✅ Model checkpoint ready.")

# ==============================================================
# 1️⃣ YAML Creation
# ==============================================================
def create_yamls(jsonl_path: Path, msa_dir: Path, intermediate_root: Path) -> Path:
    dataset = jsonl_path.stem
    yaml_dir = intermediate_root / dataset / "input"
    yaml_dir.mkdir(parents=True, exist_ok=True)

    import yaml
    with open(jsonl_path) as f:
        lines = [json.loads(l) for l in f if l.strip()]

    for dp in lines:
        dp_id = dp["datapoint_id"]
        seqs = []
        for p in dp["proteins"]:
            msa_path = str(msa_dir / p["msa"]) if p.get("msa") else None
            seqs.append({"protein": {"id": p["id"], "sequence": p["sequence"], "msa": msa_path}})
        for l in dp.get("ligands", []):
            seqs.append({"ligand": {"id": l["id"], "smiles": l["smiles"]}})
        with open(yaml_dir / f"{dp_id}_config_0.yaml", "w") as f:
            yaml.safe_dump({"version": 1, "sequences": seqs}, f, sort_keys=False)

    click.echo(f"✅ Created YAMLs under {yaml_dir}")
    return yaml_dir

# ==============================================================
# 2️⃣ Preprocess (MSA + structure only)
# ==============================================================
@rank_zero_only
def process_inputs(yaml_files: List[Path], dataset_dir: Path, mol_dir: Path):
    click.echo(f"🚀 Preprocessing dataset under {dataset_dir}")
    predictions_root = dataset_dir / "predictions"
    predictions_root.mkdir(parents=True, exist_ok=True)

    mols_root = mol_dir / "mols" if (mol_dir / "mols").exists() else mol_dir
    ccd = {}
    for mol_path in mols_root.glob("*.pkl"):
        try:
            with open(mol_path, "rb") as f:
                mol = pickle.load(f)
            mol = Chem.MolFromMolBlock(mol.decode()) if isinstance(mol, bytes) else mol
            if mol:
                ccd[mol_path.stem.upper()] = mol
        except Exception as e:
            print(f"⚠️ Skipping {mol_path}: {e}")

    for path in tqdm(yaml_files, desc="YAML preprocessing"):
        target = parse_yaml(path, ccd=ccd, mol_dir=mols_root, boltz2=True)
        tid = target.record.id
        result_dir = predictions_root / f"boltz_results_{tid}_config_0"
        processed_dir = result_dir / "processed"
        for sub in ["mols", "msa", "records", "structures", "templates"]:
            (processed_dir / sub).mkdir(parents=True, exist_ok=True)

        for chain in target.record.chains:
            msa_p = Path(chain.msa_id)
            if msa_p.exists():
                msa = parse_csv(msa_p, max_seqs=4096)
                msa.dump(processed_dir / "msa" / f"{msa_p.stem}.npz")
                chain.msa_id = msa_p.stem

        target.structure.dump(processed_dir / "structures" / f"{tid}.npz")
        target.record.dump(processed_dir / "records" / f"{tid}.json")
        Manifest([target.record]).dump(processed_dir / "manifest.json")

    click.echo("✅ Finished preprocessing — structure-only mode.")

# ==============================================================
# 3️⃣ Prediction (Unified Diffusion + Steering)
# ==============================================================
@dataclass
class BoltzDiffusionParams:
    gamma_0: float = 0.607
    gamma_min: float = 1.109
    noise_scale: float = 0.901
    rho: float = 7.8
    step_scale: float = 1.638
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True

@dataclass
class BoltzProcessedInput:
    manifest: Manifest
    targets_dir: Path
    msa_dir: Path

def predict(
    data: str,
    out_dir: str,
    cache: str = "~/.boltz",
    checkpoint: Optional[str] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 20,
    step_scale: float = 1.638,
    write_full_pae: bool = True,
    write_full_pde: bool = True,
    output_format: Literal["pdb", "mmcif"] = "pdb",
    num_workers: int = 4,
):
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")

    cache = Path(cache).expanduser()
    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser()

    download_resources(cache)
    mols_dir = str(cache / "mols")

    processed_dir = data / "processed"
    manifest_path = processed_dir / "manifest.json"
    processed = BoltzProcessedInput(
        manifest=Manifest.load(manifest_path),
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
    )

    data_module = Boltz2InferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        mol_dir=mols_dir,
        num_workers=num_workers,
    )

    if checkpoint is None:
        checkpoint = cache / "boltz2_conf.ckpt"

    diffusion_params = BoltzDiffusionParams()
    diffusion_params.step_scale = step_scale

    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "max_parallel_samples": diffusion_samples,
        "write_confidence_summary": True,
        "write_full_pae": write_full_pae,
        "write_full_pde": write_full_pde,
    }

    steering_args = {
        "fk_steering": True,
        "fk_weight": 0.4,
        "directional_guidance": True,
        "steering_strength": 0.3,
        "steering_sigma_threshold": 0.1,
        "physical_guidance_update": False,
        "contact_guidance_update": False,
        "num_particles": 1,
        "fk_resampling_interval": 1,
        "fk_lambda": 0.8,
    }

    pairformer_args = {
        "num_blocks": 8,
        "num_heads": 16,
        "dropout": 0.25,
        "pairwise_head_width": 32,
        "pairwise_num_heads": 4,
        "post_layer_norm": False,
        "activation_checkpointing": False,
        "v2": True,
    }

    model = Boltz2.load_from_checkpoint(
        checkpoint,
        strict=False,
        map_location="cpu",
        predict_args=predict_args,
        diffusion_process_args=asdict(diffusion_params),
        steering_args=steering_args,
        pairformer_args=pairformer_args,
        ema=False,
    )
    model.eval()

    pred_writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=out_dir / "predictions",
        output_format=output_format,
        boltz2=True,
    )

    trainer = Trainer(
        default_root_dir=out_dir,
        accelerator=accelerator,
        devices=devices,
        callbacks=[pred_writer],
        precision=32,
    )

    click.echo("🚀 Running unified Boltz-2 inference ...")
    trainer.predict(model, datamodule=data_module, return_predictions=False)
    click.echo("✅ All predictions completed successfully.")

# ==============================================================
# 4️⃣ Post-processing (PLDDT-based Top-5)
# ==============================================================
def postprocess_predictions(intermediate_root, submission_root):
    submission_root.mkdir(parents=True, exist_ok=True)
    out_csv = submission_root / "submission_summary.csv"

    def safe_mean(x):
        return float(np.mean(x)) if len(x) > 0 else np.nan

    def load_npz_mean(path):
        if not path.exists():
            return np.nan
        try:
            npz = np.load(path, allow_pickle=True)
            vals = [safe_mean(npz[k]) for k in npz.files if npz[k].size > 0]
            return np.mean(vals) if vals else np.nan
        except Exception:
            return np.nan

    all_results = []

    for folder in tqdm(sorted(intermediate_root.glob("boltz_results_*_config_0_config_0")), desc="Ranking by PLDDT"):
        inner_pred = list((folder / "predictions").glob("*"))
        if not inner_pred:
            continue

        pred_dir = inner_pred[0]
        target_id = pred_dir.name.replace("_config_0", "")
        print(f"\n📂 Evaluating {target_id}")

        records = []
        for i in range(20):
            base = f"{target_id}_config_0_model_{i}"
            rec = {
                "target": target_id,
                "model_index": i,
                "plddt": load_npz_mean(pred_dir / f"plddt_{base}.npz"),
                "model_path": pred_dir / f"{base}.pdb"
            }
            records.append(rec)

        df = pd.DataFrame(records)
        if df.empty:
            continue

        df = df.sort_values("plddt", ascending=False).reset_index(drop=True)
        top_models = df.head(5)

        sub_dir = submission_root / target_id
        sub_dir.mkdir(parents=True, exist_ok=True)

        for rank, row in enumerate(top_models.itertuples()):
            src = row.model_path
            dst = sub_dir / f"model_{rank}.pdb"
            if src.exists():
                import shutil
                shutil.copy2(src, dst)

        best = top_models.iloc[0]
        all_results.append({
            "target": target_id,
            "best_plddt": best.plddt,
            "avg_top5_plddt": float(top_models.plddt.mean()),
            "std_top5_plddt": float(top_models.plddt.std())
        })

    summary = pd.DataFrame(all_results)
    summary.to_csv(out_csv, index=False)
    print(f"\n💾 Saved → {out_csv}")
    print(f"✅ Top-5 models copied into {submission_root}")

# ==============================================================
# 5️⃣ Main
# ==============================================================
def main():
    parser = argparse.ArgumentParser(description="Boltz-2 Hackathon Full Pipeline (PLDDT only)")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--msa-dir", required=True)
    parser.add_argument("--submission-dir", required=True)
    parser.add_argument("--intermediate-dir", required=True)
    parser.add_argument("--result-folder", required=True)
    parser.add_argument("--devices", type=int, default=1)
    args = parser.parse_args()

    seed_everything(42)
    jsonl = Path(args.input_jsonl).resolve()
    msa = Path(args.msa_dir).resolve()
    inter_root = Path(args.intermediate_dir).resolve()
    submission_root = Path(args.submission_dir).resolve()
    result_root = Path(args.result_folder).resolve()
    cache = Path("boltz-2").resolve()

    download_resources(cache)
    yaml_dir = create_yamls(jsonl, msa, inter_root)
    process_inputs(list(yaml_dir.glob("*.yaml")), inter_root, cache / "mols")

    for result_dir in (inter_root / "predictions").glob("boltz_results_*"):
        predict(data=result_dir, out_dir=result_root, cache=cache, devices=args.devices)

    click.echo("🎯 Inference done — starting PLDDT post-processing.")
    postprocess_predictions(intermediate_root=result_root, submission_root=submission_root)

if __name__ == "__main__":
    main()
