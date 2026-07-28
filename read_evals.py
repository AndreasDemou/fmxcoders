import wandb
import numpy as np
import argparse
import sys

# Configuration
WANDB_ENTITY = "facereenactment-team"
WANDB_PROJECT = "icml_sae"

DATASETS = [
    "europarl",
    "amazon_1and5_sentiment",
    "github_code",
    "ag_news",
    "amazon_1and5",
]

BIOS_CLASS_SETS = [
    "bios_class_set1",
    "bios_class_set2", 
    "bios_class_set3",
]
BIOS_PRIMARY_KEY = "bias_in_bios"

def print_ascii_bar(label, value, max_val=100, bar_len=40):
    val_clamped = max(0, min(max_val, value))
    bar_filled = int(round(val_clamped / max_val * bar_len))
    bar = "█" * bar_filled + "░" * (bar_len - bar_filled)
    print(f"{label:25} | {bar} {value:5.1f}%")

def get_run_metrics(run):
    """Extracts WandB run metrics and formats them as a LaTeX table row and ASCII plot."""
    summary = run.summary
    mse = summary.get("core/reconstruction_quality/mse")
    
    k = 1 # k=1 for Top-1 metric evaluation
    metrics = {}
    
    # Fetch F1 and Wasserstein scores for all desired datasets.
    # Note: Confirmed this matches exact WandB keys from fetch_wandb_tables.py.
    for ds in DATASETS + BIOS_CLASS_SETS + [BIOS_PRIMARY_KEY]:
        f1_val = summary.get(f"sparse_probing/datasets/{ds}/sae_top_{k}_test_f1")
        if f1_val is None:
            f1_val = summary.get(f"sparse_probing/datasets/{ds}/k_{k}/macro_f1")
            
        wass_val = summary.get(f"sparse_probing/datasets/{ds}/sae_top_{k}_wasserstein")
        if wass_val is None:
            wass_val = summary.get(f"sparse_probing/datasets/{ds}/k_{k}/median_wasserstein")
        
        metrics[ds] = {"f1": f1_val, "wass": wass_val}
        
    # Bios is evaluated across 3 sets typically, we average them
    bios_f1s = []
    bios_wass = []
    for bs in BIOS_CLASS_SETS:
        if metrics[bs]["f1"] is not None: bios_f1s.append(metrics[bs]["f1"])
        if metrics[bs]["wass"] is not None: bios_wass.append(metrics[bs]["wass"])

    if not bios_f1s and metrics[BIOS_PRIMARY_KEY]["f1"] is not None:
        bios_f1 = metrics[BIOS_PRIMARY_KEY]["f1"]
        bios_w = metrics[BIOS_PRIMARY_KEY]["wass"]
    else:
        bios_f1 = np.mean(bios_f1s) if bios_f1s else None
        bios_w = np.mean(bios_wass) if bios_wass else None

    # Gather F1s for all datasets to compute the overall mean and std
    all_f1s = []
    f1_dict_for_plot = {}
    for ds in DATASETS:
        if metrics[ds]["f1"] is not None:
            val = metrics[ds]["f1"] * 100
            all_f1s.append(val)
            f1_dict_for_plot[ds] = val
            
    if bios_f1 is not None:
        val = bios_f1 * 100
        all_f1s.append(val)
        f1_dict_for_plot["bias_in_bios (avg)"] = val

    mean_f1 = np.mean(all_f1s) if all_f1s else 0
    std_f1 = np.std(all_f1s) if all_f1s else 0

    def fmt(ds_key):
        """Format metrics as 'F1 / Wasserstein' scaled to percentages/x1000."""
        try:
            f = metrics[ds_key]["f1"] * 100
            w = metrics[ds_key]["wass"] * 1000
            return f"{f:.1f} / {w:.1f}"
        except:
            return "- / -"

    # Format entries
    mse_str = f"{mse:.2f}" if mse is not None else "-"
    f1_mean_std = f"{mean_f1:.1f} \\pm {std_f1:.1f}"
    
    euro = fmt('europarl')
    bios = f"{bios_f1 * 100:.1f} / {bios_w * 1000:.1f}" if bios_f1 is not None else "- / -"
    amzn_sent = fmt('amazon_1and5_sentiment')
    github = fmt('github_code')
    ag_news = fmt('ag_news')
    amzn_15 = fmt('amazon_1and5')
    
    print("\n" + "="*80)
    print("               ASCII TERMINAL PLOT (Top-1 F1 %)")
    print("="*80)
    for ds_label, f1_val in f1_dict_for_plot.items():
        print_ascii_bar(ds_label, f1_val)
    print_ascii_bar("OVERALL MEAN", mean_f1)
    
    print("\n" + "="*80)
    print("                       LATEX TABLE ROW")
    print("="*80)
    # LaTeX Row Generation with Header
    header = "MSE & Mean F1 & Europarl & Bias in Bios & Amazon Sent & Github & AG News & Amazon 1and5 \\\\"
    print(header)
    latex_row = f"{mse_str} & {f1_mean_std} & {euro} & {bios} & {amzn_sent} & {github} & {ag_news} & {amzn_15}"
    print(latex_row)
    
    return "\nFinished extracting metrics."

def main():
    parser = argparse.ArgumentParser(description="Fetch and format WandB run metrics into LaTeX table rows.")
    parser.add_argument("run", help="WandB Run name or ID")
    args = parser.parse_args()
    
    api = wandb.Api()
    run_identifier = args.run
    run = None
    
    # 1. Try fetching by ID first
    try:
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_identifier}")
    except:
        pass
        
    # 2. Try searching by display name
    if run is None:
        runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"display_name": run_identifier})
        if len(runs) > 0:
            run = runs[0]
    
    if run is None:
        print(f"Error: Run '{run_identifier}' not found in project {WANDB_ENTITY}/{WANDB_PROJECT}.", file=sys.stderr)
        sys.exit(1)
        
    print(f"% Found Run: {run.name} ({run.id})")
    print(get_run_metrics(run))

if __name__ == "__main__":
    main()

