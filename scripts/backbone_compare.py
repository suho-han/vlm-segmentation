import os
import json
from pathlib import Path
import argparse
import csv

def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate metrics from runs")
    parser.add_argument("--runs_dir", type=str, default="runs", help="Directory containing runs")
    parser.add_argument("--out_dir", type=str, default="runs/backbone_compare", help="Output directory")
    return parser.parse_args()

def aggregate_metrics(runs_dir):
    runs_path = Path(runs_dir)
    results = []
    
    if not runs_path.exists():
        return []

    # Traverse through dataset / model / exp_id
    for dataset_dir in sorted(runs_path.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name in ["analysis", "backbone_compare", "dummy"]:
            continue
            
        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
                
            for exp_dir in sorted(model_dir.iterdir()):
                if not exp_dir.is_dir():
                    continue
                    
                metrics_file = exp_dir / "metrics.json"
                if metrics_file.exists():
                    try:
                        with open(metrics_file, "r") as f:
                            data = json.load(f)
                        
                        agg = data.get("aggregate", data)
                        
                        res = {
                            "dataset": dataset_dir.name,
                            "model": model_dir.name,
                            "exp_id": exp_dir.name,
                            "Dice": agg.get("Dice", 0.0),
                            "IoU": agg.get("IoU", 0.0),
                            "hd95": agg.get("hd95", float("inf")),
                            "betti_beta0": agg.get("betti_beta0", 0.0),
                            "betti_beta1": agg.get("betti_beta1", 0.0),
                        }
                        results.append(res)
                    except Exception as e:
                        print(f"Error reading {metrics_file}: {e}")
    
    return results

def format_as_markdown_table(data):
    if not data:
        return ""
    headers = data[0].keys()
    # Remove 'dataset' from headers for the sub-table
    sub_headers = [h for h in headers if h != 'dataset']
    
    lines = []
    lines.append("| " + " | ".join(sub_headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(sub_headers)) + " |")
    
    for row in data:
        lines.append("| " + " | ".join(str(row[h]) for h in sub_headers) + " |")
    
    return "\n".join(lines)

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = aggregate_metrics(args.runs_dir)
    if not results:
        print("No results found.")
        return
        
    # Save CSV
    keys = results[0].keys()
    with open(out_dir / "summary.csv", "w", newline='') as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
    
    # Save Markdown
    with open(out_dir / "summary.md", "w") as f:
        f.write("# Backbone Comparison Summary\n\n")
        datasets = sorted(list(set(r["dataset"] for r in results)))
        for dataset in datasets:
            f.write(f"## {dataset}\n\n")
            subset = [r for r in results if r["dataset"] == dataset]
            f.write(format_as_markdown_table(subset))
            f.write("\n\n")
            
    print(f"Summary reports saved to {out_dir}")

if __name__ == "__main__":
    main()
