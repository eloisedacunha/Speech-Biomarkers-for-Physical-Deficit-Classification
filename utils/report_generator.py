import pandas as pd
import os
import json
from utils.config import DEFICIT_MAP

def generate_deficit_report(deficit_name):
    """Génère un rapport consolidé pour un déficit"""
    report = {
        "deficit": deficit_name,
        "models": {}
    }
    
    # Modèles individuels
    for data_type in ["POS", "NEG"]:
        metrics_path = f"results/{deficit_name}/{data_type}/final_metrics.csv"
        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path, index_col=0)
            report["models"][data_type] = metrics_df.to_dict()
    
    # Stacking
    stacking_path = f"results/{deficit_name}/stacking/stacking_metrics.json"
    if os.path.exists(stacking_path):
        with open(stacking_path) as f:
            report["stacking"] = json.load(f)
    
    return report

def main():
    all_reports = []
    
    for class_id, deficit_name in DEFICIT_MAP.items():
        print(f"Generating report for {deficit_name}...")
        report = generate_deficit_report(deficit_name)
        all_reports.append(report)
    
    # Sauvegarde
    report_dir = "results/summary"
    os.makedirs(report_dir, exist_ok=True)
    
    # Format JSON complet
    with open(f"{report_dir}/full_report.json", "w") as f:
        json.dump(all_reports, f, indent=2)
    
    # Format CSV simplifié
    summary_data = []
    for report in all_reports:
        row = {"deficit": report["deficit"]}
        for data_type, models in report.get("models", {}).items():
            for model, metrics in models.items():
                prefix = f"{model}_{data_type}_"
                for metric, value in metrics.items():
                    row[prefix + metric] = value
        
        if "stacking" in report:
            for metric, value in report["stacking"].items():
                row["stacking_" + metric] = value
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{report_dir}/summary_report.csv", index=False)
    
    print("\nReports generated:")
    print(f"- {report_dir}/full_report.json")
    print(f"- {report_dir}/summary_report.csv")

if __name__ == "__main__":
    main()