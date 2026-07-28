#!/usr/bin/env python3
"""
Evaluate a saved SAE using all SAEBench evaluations.
Uses EXACTLY the same metric names as train_and_saebench.py for wandb comparison.

Usage:
    python eval_sae.py --sae_path output/ --preset gpt2_l8
    python eval_sae.py --sae_path output/ --preset gpt2_l8 --wandb_project icml_polynomial_sae
"""

import argparse
import json
import os
import sys
from glob import glob
from typing import Dict, List
sys.path.insert(0, "SAEBench")

import numpy as np
import torch
import wandb
from sae_lens import SAE


def log_saebench_results_to_wandb(output_dir: str, sae_name: str) -> None:
    """
    Parse SAEBench JSON result files and log key metrics to wandb.
    EXACTLY the same as train_and_saebench.py for metric name consistency.
    
    Args:
        output_dir: Directory containing SAEBench results
        sae_name: Name of the SAE (used for finding result files)
    """
    if wandb.run is None:
        print("[wandb] No active run, skipping result logging")
        return
    
    print("\n[wandb] Logging SAEBench results...")
    print(f"[wandb] Looking for results in: {output_dir}")
    
    # Find and parse result files
    result_files = glob(f"{output_dir}/**/*_eval_results.json", recursive=True)
    
    print(f"[wandb] Found {len(result_files)} result files")
    
    if not result_files:
        print("[wandb] No result files found to log")
        return
    
    logged_metrics = {}
    
    for result_file in result_files:
        print(f"[wandb] Processing: {result_file}")
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            eval_type = data.get("eval_type_id", "unknown")
            metrics = data.get("eval_result_metrics", {})
            
            # Flatten nested metrics with prefix
            for category, category_metrics in metrics.items():
                if isinstance(category_metrics, dict):
                    for metric_name, value in category_metrics.items():
                        # Skip infinity and NaN values
                        if isinstance(value, float) and (value != value or abs(value) == float('inf')):
                            continue
                        if value is None:
                            continue
                        key = f"{eval_type}/{category}/{metric_name}"
                        logged_metrics[key] = value
                elif isinstance(category_metrics, (int, float)):
                    # Skip infinity and NaN values
                    if isinstance(category_metrics, float) and (category_metrics != category_metrics or abs(category_metrics) == float('inf')):
                        continue
                    key = f"{eval_type}/{category}"
                    logged_metrics[key] = category_metrics
            
            # For sparse_probing, also log per-dataset metrics from eval_result_details
            if eval_type == "sparse_probing":
                details = data.get("eval_result_details", [])
                for dataset_result in details:
                    # Extract dataset name (remove _results suffix if present)
                    dataset_name = dataset_result.get("dataset_name", "unknown")
                    dataset_name = dataset_name.replace("_results", "")
                    # Shorten common prefixes for cleaner metric names
                    dataset_name = dataset_name.replace("LabHC/bias_in_bios_", "bios_")
                    dataset_name = dataset_name.replace("canrager/amazon_reviews_mcauley_", "amazon_")
                    dataset_name = dataset_name.replace("codeparrot/github-code", "github_code")
                    dataset_name = dataset_name.replace("fancyzhx/ag_news", "ag_news")
                    dataset_name = dataset_name.replace("Helsinki-NLP/europarl", "europarl")
                    
                    # Log all accuracy metrics for this dataset
                    for metric_name, value in dataset_result.items():
                        if metric_name == "dataset_name":
                            continue
                        if value is None:
                            continue
                        if isinstance(value, float) and (value != value or abs(value) == float('inf')):
                            continue
                        key = f"{eval_type}/datasets/{dataset_name}/{metric_name}"
                        logged_metrics[key] = value
            
            # For TPP and SCR, log per-dataset metrics and compute median/std
            if eval_type in ("tpp", "scr"):
                details = data.get("eval_result_details", [])
                
                # Collect values per metric for median/std computation
                metric_values: Dict[str, List[float]] = {}
                
                for dataset_result in details:
                    # Extract dataset name
                    dataset_name = dataset_result.get("dataset_name", "unknown")
                    dataset_name = dataset_name.replace("_results", "")
                    dataset_name = dataset_name.replace("_tpp", "")
                    dataset_name = dataset_name.replace("_scr", "")
                    # Shorten common prefixes
                    dataset_name = dataset_name.replace("LabHC/bias_in_bios_class_set1_", "bios_")
                    dataset_name = dataset_name.replace("canrager/amazon_reviews_mcauley_1and5_", "amazon_")
                    
                    # Log all metrics for this dataset pair
                    for metric_name, value in dataset_result.items():
                        if metric_name == "dataset_name":
                            continue
                        if value is None:
                            continue
                        if isinstance(value, float) and (value != value or abs(value) == float('inf')):
                            continue
                        
                        # Log per-dataset value
                        key = f"{eval_type}/datasets/{dataset_name}/{metric_name}"
                        logged_metrics[key] = value
                        
                        # Collect for median/std
                        if metric_name not in metric_values:
                            metric_values[metric_name] = []
                        metric_values[metric_name].append(value)
                
                # Compute and log median and std for each metric
                for metric_name, values in metric_values.items():
                    if len(values) > 1:
                        median_val = float(np.median(values))
                        std_val = float(np.std(values))
                        logged_metrics[f"{eval_type}/median/{metric_name}"] = median_val
                        logged_metrics[f"{eval_type}/std/{metric_name}"] = std_val
            
            print(f"  ✓ Parsed {eval_type} results")
            
        except Exception as e:
            print(f"  ⚠ Failed to parse {result_file}: {e}")
    
    # Log all metrics to wandb
    if logged_metrics:
        print(f"[wandb] Logging {len(logged_metrics)} metrics...")
        for key, value in list(logged_metrics.items())[:5]:  # Show first 5 metrics
            print(f"  {key}: {value}")
        if len(logged_metrics) > 5:
            print(f"  ... and {len(logged_metrics) - 5} more")
        wandb.log(logged_metrics)
        print(f"[wandb] ✓ Logged {len(logged_metrics)} SAEBench metrics to wandb")
    else:
        print("[wandb] No metrics extracted from result files")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved SAE with SAEBench")
    parser.add_argument("--sae_path", required=True, help="Path to saved SAE directory")
    parser.add_argument("--preset", default="gpt2_l8", help="Model preset (gpt2_l8, pythia410m_l15)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", default="eval_results")
    parser.add_argument("--skip_eval", action="append", default=[],
                        help="Evals to skip: l0_loss, sparse_probing, tpp, scr, absorption")
    parser.add_argument("--wandb_project", default=None, help="Wandb project name (enables wandb logging)")
    parser.add_argument("--wandb_entity", default=None, help="Wandb entity/team name")
    parser.add_argument("--wandb_run_name", default=None, help="Wandb run name")
    args = parser.parse_args()

    # Model configs
    PRESETS = {
        "gpt2_l8": {"model_name": "gpt2", "dataset": "Skylion007/openwebtext", "context_size": 128},
        "pythia410m_l15": {"model_name": "EleutherAI/pythia-410m-deduped", "dataset": "monology/pile-uncopyrighted", "context_size": 1024},
    }
    preset = PRESETS.get(args.preset)
    if not preset:
        print(f"Unknown preset: {args.preset}. Available: {list(PRESETS.keys())}")
        return

    model_name = preset["model_name"]
    dataset = preset["dataset"]
    output_dir = args.output_dir
    device = args.device

    # Initialize wandb if requested
    sae_name = args.sae_path.rstrip("/").split("/")[-1]
    if args.wandb_project:
        run_name = args.wandb_run_name or f"eval_{sae_name}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "sae_path": args.sae_path,
                "preset": args.preset,
                "model_name": model_name,
            }
        )
        print(f"[wandb] Initialized run: {run_name}")

    # Load SAE
    print(f"Loading SAE from {args.sae_path}...")
    sae = SAE.load_from_disk(args.sae_path)
    sae.eval()
    sae.to(device)
    print(f"  Architecture: {sae.cfg.architecture()}")
    print(f"  d_sae: {sae.cfg.d_sae}, d_in: {sae.cfg.d_in}")

    # Try to fold decoder norms
    try:
        sae.fold_W_dec_norm()
        print("✓ Decoder norms folded into encoder")
    except (NotImplementedError, AttributeError):
        print("⚠ Warning: fold_W_dec_norm() not supported")

    # Determine dtype
    dtype = "float32"
    if hasattr(sae.cfg, "dtype"):
        dtype = sae.cfg.dtype

    # Import SAEBench modules
    import sae_bench.evals.core.main as saebench_core
    import sae_bench.evals.scr_and_tpp.main as saebench_scr_tpp
    import sae_bench.evals.sparse_probing.main as saebench_sparse_probing
    import sae_bench.evals.absorption.main as saebench_absorption
    
    selected_saes = [(sae_name, sae)]
    
    # Eval configurations - EXACTLY same as train_and_saebench.py
    eval_configs = {
        "l0_loss": {
            "name": "Core (L0/Loss Recovered)",
            "runner": lambda: saebench_core.multiple_evals(
                selected_saes=selected_saes,
                n_eval_reconstruction_batches=200,
                n_eval_sparsity_variance_batches=2000,
                eval_batch_size_prompts=16,
                compute_featurewise_density_statistics=True,
                compute_featurewise_weight_based_metrics=True,
                exclude_special_tokens_from_reconstruction=True,
                dataset=dataset,
                context_size=128,
                output_folder=f"{output_dir}/core",
                verbose=True,
                dtype=dtype,
                device=device,
            ),
        },
        "sparse_probing": {
            "name": "Sparse Probing",
            "runner": lambda: saebench_sparse_probing.run_eval(
                saebench_sparse_probing.SparseProbingEvalConfig(
                    model_name=model_name,
                    random_seed=42,
                    llm_dtype=dtype,
                    llm_batch_size=16,
                ),
                selected_saes,
                device,
                f"{output_dir}/sparse_probing",
                force_rerun=False,
                clean_up_activations=True,
                save_activations=False,
            ),
        },
        "tpp": {
            "name": "Targeted Probe Perturbation (TPP)",
            "runner": lambda: saebench_scr_tpp.run_eval(
                saebench_scr_tpp.ScrAndTppEvalConfig(
                    model_name=model_name,
                    random_seed=42,
                    perform_scr=False,
                    llm_dtype=dtype,
                    llm_batch_size=16,
                ),
                selected_saes,
                device,
                output_dir,
                force_rerun=False,
                clean_up_activations=True,
                save_activations=False,
            ),
        },
        "scr": {
            "name": "Spurious Correlation Removal (SCR)",
            "runner": lambda: saebench_scr_tpp.run_eval(
                saebench_scr_tpp.ScrAndTppEvalConfig(
                    model_name=model_name,
                    random_seed=42,
                    perform_scr=True,
                    llm_dtype=dtype,
                    llm_batch_size=16,
                ),
                selected_saes,
                device,
                output_dir,
                force_rerun=False,
                clean_up_activations=True,
                save_activations=False,
            ),
        },
        "absorption": {
            "name": "Feature Absorption",
            "runner": lambda: saebench_absorption.run_eval(
                saebench_absorption.AbsorptionEvalConfig(
                    model_name=model_name,
                    random_seed=42,
                    llm_dtype=dtype,
                ),
                selected_saes,
                device,
                f"{output_dir}/absorption",
                force_rerun=False,
            ),
        },
    }
    
    # Run evaluations
    skip_set = set(args.skip_eval)
    for eval_key, eval_info in eval_configs.items():
        if eval_key in skip_set:
            print(f"\n[SKIP] {eval_info['name']}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Running: {eval_info['name']}")
        print(f"{'='*70}")
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            eval_info["runner"]()
            print(f"✓ Completed: {eval_info['name']}")
        except Exception as e:
            print(f"\n[ERROR] {eval_info['name']} failed:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print(f"SAEBench evaluations completed!")
    print(f"Results saved to: {output_dir}")
    print("="*70)
    
    # Log to wandb - EXACTLY same function as train_and_saebench.py
    if args.wandb_project:
        log_saebench_results_to_wandb(output_dir, sae_name)
        wandb.finish()
        print("[wandb] Run finished")


if __name__ == "__main__":
    main()
