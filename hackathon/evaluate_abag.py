#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import pandas as pd

# =========================================================
# ⚙️ Folder Configuration
# =========================================================
BASE_DIR = Path("hackathon_data")
GT_DIR = BASE_DIR / "datasets" / "abag_public" / "ground_truth"
SUBMISSION_DIR = BASE_DIR / "submission" / "abag_public"
RESULTS_DIR = BASE_DIR / "evaluation" / "abag_public"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Number of models per target (e.g., model_0.pdb … model_4.pdb)
NSAMPLES = 5
NTHREADS = 20  # parallel CAPRI-Q containers

# =========================================================
# 🧮 CAPRI-Q Docker evaluation
# =========================================================
def run_evaluation(target_id: str, model_index: int) -> Optional[pd.DataFrame]:
    """Run CAPRI-Q evaluation for one model."""
    output_subdir = RESULTS_DIR / f"{target_id}_{model_index}"
    output_subdir.mkdir(parents=True, exist_ok=True)
    prediction_file = SUBMISSION_DIR / target_id / f"model_{model_index}.pdb"

    if not prediction_file.exists():
        print(f"⚠️ Prediction {prediction_file} missing.")
        return None

    # expected GT files
    gt_complex = GT_DIR / f"{target_id}_complex.pdb"
    gt_ab = GT_DIR / f"{target_id}_Ab.pdb"
    gt_lig = GT_DIR / f"{target_id}_ligand.pdb"

    if not gt_complex.exists() or not gt_ab.exists() or not gt_lig.exists():
        print(f"⚠️ Missing GT components for {target_id}.")
        return None

    capriq_cmd = [
        "/capri-q/bin/capriq",
        "-a", "--dontwrite",
        "-t", f"/app/ground_truth/{gt_complex.name}",
        "-u", f"/app/ground_truth/{gt_ab.name}",
        "-u", f"/app/ground_truth/{gt_lig.name}",
        "-z", "/app/outputs/",
        "-p", "65",
        "-o", f"/app/outputs/{target_id}_{model_index}_results.txt",
        "-l", f"/app/outputs/{target_id}_{model_index}_errors.txt",
        f"/app/predictions/prediction.pdb",
        "&&",
        "chown", "-R", f"{os.getuid()}:{os.getgid()}", "/app/outputs"
    ]

    docker_cmd = [
        "docker", "run", "--group-add", str(os.getgid()), "--rm", "--network", "none",
        "-v", f"{GT_DIR.absolute()}:/app/ground_truth/",
        "-v", f"{output_subdir.absolute()}:/app/outputs",
        "-v", f"{prediction_file.absolute()}:/app/predictions/prediction.pdb",
        "gitlab-registry.in2p3.fr/cmsb-public/capri-q",
        "/bin/bash", "-c", " ".join(capriq_cmd)
    ]

    print(f"🚀 Evaluating {target_id} model {model_index}")
    try:
        subprocess.run(docker_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ CAPRI-Q failed for {target_id} model {model_index}: {e}")
        return pd.DataFrame({
            "target": [target_id],
            "model_index": [model_index],
            "classification": ["error"],
            "error": [str(e)]
        })

    result_file = output_subdir / f"{target_id}_{model_index}_results.txt"
    if not result_file.exists():
        return pd.DataFrame({
            "target": [target_id],
            "model_index": [model_index],
            "classification": ["error"],
            "error": ["Result file not found"]
        })

    try:
        df = pd.read_csv(result_file, sep=r"\s+")
        df["target"] = target_id
        df["model_index"] = model_index
        return df
    except Exception as e:
        print(f"⚠️ Failed to parse results for {target_id} model {model_index}: {e}")
        return pd.DataFrame({
            "target": [target_id],
            "model_index": [model_index],
            "classification": ["error"],
            "error": [str(e)]
        })

# =========================================================
# 🧠 Parallel Evaluation Loop
# =========================================================
def main():
    targets = sorted([d.name for d in SUBMISSION_DIR.iterdir() if d.is_dir()])
    result_dfs = []

    with ThreadPoolExecutor(max_workers=NTHREADS) as executor:
        futures = []
        for target_id in targets:
            for i in range(NSAMPLES):
                futures.append(executor.submit(run_evaluation, target_id, i))

        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                result_dfs.append(res)

    if not result_dfs:
        print("❌ No evaluations completed.")
        return

    combined = pd.concat(result_dfs, ignore_index=True)
    combined.to_csv(RESULTS_DIR / "combined_results.csv", index=False)

    # Classification summary for top-1 models (model_0)
    n_successful = 0
    good = ["high", "medium", "acceptable"]
    bad = ["incorrect", "error"]

    print("\n===== CAPRI-Q Evaluation Summary =====")
    for label in good + bad:
        n = len(combined[(combined["model_index"] == 0) & (combined["classification"].str.contains(label, case=False, na=False))])
        print(f"Number of {label} classifications in top 1: {n}")
        if label in good:
            n_successful += n

    print(f"\n✅ Number of successful top-1 predictions: {n_successful} out of {len(targets)}")
    print("All evaluations completed.")
    print(f"💾 Results saved to {RESULTS_DIR/'combined_results.csv'}")

if __name__ == "__main__":
    main()

