"""
Main entry point for the Moneyball 2.0 pipeline.

Usage:
    python main.py [--all] [--gather] [--clean] [--model] [--cluster]

Options:
    --all      Run the full pipeline (default if no flags given)
    --gather   Run data gathering (scraping)
    --clean    Run data cleaning and database creation
    --model    Run model training
    --cluster  Run clustering and value segmentation

Examples:
    python main.py --all
    python main.py --gather --clean
"""
import argparse
import subprocess
import sys

STEPS = [
    ("gather", "python src/data/load_data.py"),
    ("clean", "python src/data/clean_data.py"),
    ("model", "python src/models/train_model.py"),
    ("update_value_labels", "python src/models/get_predicted_war.py"),
    ("cluster", "python src/clustering/clustering.py"),
]

def run_step(name, cmd):
    print(f"\n=== Running step: {name} ===")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Step '{name}' failed. Exiting.")
        sys.exit(result.returncode)
    print(f"=== Step '{name}' completed ===\n")

def main():
    parser = argparse.ArgumentParser(description="Run the Moneyball 2.0 pipeline.")
    parser.add_argument('--all', action='store_true', help='Run all steps')
    parser.add_argument('--gather', action='store_true', help='Run data gathering')
    parser.add_argument('--clean', action='store_true', help='Run data cleaning')
    parser.add_argument('--model', action='store_true', help='Run model training')
    parser.add_argument('--cluster', action='store_true', help='Run clustering')
    args = parser.parse_args()

    # If no flags, run all
    if not any([args.all, args.gather, args.clean, args.model, args.cluster]):
        args.all = True

    steps_to_run = []
    if args.all or args.gather:
        steps_to_run.append(STEPS[0])
    if args.all or args.clean:
        steps_to_run.append(STEPS[1])
    if args.all or args.model:
        steps_to_run.append(STEPS[2])
        steps_to_run.append(STEPS[3])  # update_value_labels after model
    if args.all or args.cluster:
        steps_to_run.append(STEPS[4])

    for name, cmd in steps_to_run:
        run_step(name, cmd)

if __name__ == "__main__":
    main() 