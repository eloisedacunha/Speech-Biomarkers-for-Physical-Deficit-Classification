import argparse
import subprocess
from utils.config import DEFICIT_MAP

def run_training(classification_id):
    for data_type in ['POS', 'NEG']:
        cmd = [
            "python", "models/train_model.py",
            "--classification", str(classification_id),
            "--type", data_type
        ]
        subprocess.run(cmd, check=True)

def run_explanation(classification_id):
    for data_type in ['POS', 'NEG']:
        cmd = [
            "python", "models/explain_model.py",
            "--classification", str(classification_id),
            "--type", data_type
        ]
        subprocess.run(cmd, check=True)

def run_stacking(classification_id):
    cmd = [
        "python", "models/stacking_model.py",
        "--classification", str(classification_id)
    ]
    subprocess.run(cmd, check=True)

def main(start_id=1, end_id=10):
    for class_id in range(start_id, end_id + 1):
        deficit_name = DEFICIT_MAP[class_id]
        print(f"\n{'='*40}")
        print(f"Processing {deficit_name} (ID {class_id})")
        print(f"{'='*40}")
        
        print("\nTraining models...")
        run_training(class_id)
        
        print("\nGenerating explanations...")
        run_explanation(class_id)
        
        print("\nCreating stacking model...")
        run_stacking(class_id)
        
        print(f"\nCompleted {deficit_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1, choices=range(1, 11))
    parser.add_argument("--end", type=int, default=10, choices=range(1, 11))
    args = parser.parse_args()
    main(args.start, args.end)