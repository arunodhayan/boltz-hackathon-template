# 🧬 Boltz-2 Hackathon Full Pipeline  
**Agentic Diffusion Search Tree + PLDDT-Based Post-Processing**

This repository provides a complete **end-to-end structure prediction pipeline** using **Boltz-2**, a diffusion-based generative model for biomolecular complexes.  
It performs dataset preprocessing, multi-agent diffusion inference, and **PLDDT-driven agentic search tree selection** to produce top-ranked structures.

---

## 🚀 Quick Start

```bash
python3 boltz2_pipeline.py \
  --input-jsonl datasets/abag_public.jsonl \
  --msa-dir datasets/abag_public/msa \
  --submission-dir hackathon_data/submission \
  --intermediate-dir hackathon_data/intermediate_files/abag_public \
  --result-folder hackathon_data/intermediate_files/abag_public/predictions \
  --devices 1
```

---

## ⚙️ Key Parameters

| Parameter | Default | Description |
|------------|----------|-------------|
| `--input-jsonl` | *required* | Input dataset describing proteins & ligands. |
| `--msa-dir` | *required* | Folder with MSA CSV files. |
| `--submission-dir` | *required* | Output folder for final Top-5 PDBs. |
| `--intermediate-dir` | *required* | Stores YAMLs, manifests, and processed data. |
| `--result-folder` | *required* | Destination for inference predictions. |
| `--devices` | `1` | Number of GPUs to use. |

**Internal model settings**
- `sampling_steps = 200`  
- `diffusion_samples = 20`  
- `recycling_steps = 3`  
- `step_scale = 1.638`  
- `pairformer_args.v2 = True`  
- Steering: `fk_steering=True`, `directional_guidance=True`, `fk_lambda=0.8`

---

## 🌳 Agentic Post-Processing

After inference, **20 diffusion samples** per target are evaluated via **mean PLDDT**.  
This step forms an **agentic search tree**, where each node represents a generated conformation:

1. Compute PLDDT mean for each model (`plddt_*.npz`).  
2. Rank all 20 conformations by confidence.  
3. Select **Top-5** (beam-search-style pruning).  
4. Copy best models to submission folder.  
5. Save summary metrics in `submission_summary.csv`.

**Output structure**

```
submission/
 ├── 8BK2/
 │    ├── model_0.pdb
 │    ├── model_1.pdb
 │    ├── model_2.pdb
 │    ├── model_3.pdb
 │    └── model_4.pdb
 └── submission_summary.csv
```

---

## 📊 Output Metrics

| Column | Meaning |
|---------|----------|
| `best_plddt` | PLDDT of highest-confidence model |
| `avg_top5_plddt` | Mean PLDDT across selected Top-5 |
| `std_top5_plddt` | Spread indicating ensemble stability |

---

## 🧠 Notes
- **Checkpoint:** auto-downloads from Hugging Face (`boltz2_conf.ckpt`).  
- **Molecule DB:** cached under `~/.boltz/mols`.  
- **Deterministic:** full seed control for reproducibility.  
- **Hardware:** NVIDIA A100/H100 (≥ 20 GB VRAM) recommended.  

---

**Author:** Arunodhayan Sampathkumar M.Sc.  
**Version:** Boltz-2 Hackathon 2025
