#!/usr/bin/env python3
"""
==============================================================
🧬 Boltz-2 + CAPRI-Q Hackathon Orchestrator
==============================================================
1️⃣ Runs Boltz-2 inference (PLDDT-based Top-5 selection)
2️⃣ Optionally runs CAPRI-Q Docker evaluation
"""

import argparse
import subprocess
from pathlib import Path
from boltz2_hackathon_pipeline import main as boltz2_main
from evaluate_abag import main as capri_main

def main():
    parser = argparse.ArgumentParser(description="Boltz-2 + CAPRI-Q Combined Pipeline")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--msa-dir", required=True)
    parser.add_argument("--submission-dir", required=True)
    parser.add_argument("--intermediate-dir", required=True)
    parser.add_argument("--result-folder", required=True)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--run-eval", action="store_true",
                        help="Run CAPRI-Q evaluation after inference")
    args = parser.parse_args()

    # 1️⃣ Run Boltz-2 full inference + postprocessing
    print("\n🚀 Running Boltz-2 inference and PLDDT postprocessing ...")
    boltz2_args = [
        "--input-jsonl", args.input_jsonl,
        "--msa-dir", args.msa_dir,
        "--submission-dir", args.submission_dir,
        "--intermediate-dir", args.intermediate_dir,
        "--result-folder", args.result_folder,
        "--devices", str(args.devices),
    ]
    subprocess.run(["python3", "boltz2_hackathon_pipeline.py", *boltz2_args], check=True)

    # 2️⃣ Optional CAPRI-Q evaluation
    if args.run_eval:
        print("\n🧠 Running CAPRI-Q evaluation ...")
        subprocess.run(["python3", "evaluate_abag.py"], check=True)
        print("\n✅ CAPRI-Q evaluation completed successfully.")

if __name__ == "__main__":
    main()

