import subprocess
import os
from pathlib import Path
from datetime import datetime

# Define paths
OCR_EVALUATION_SCRIPT = Path("ocr_evaluation.py")
GROUND_TRUTH_FILE = Path("lp_all_dataset/all_anotaciones.csv")
# List of prediction files and their corresponding output directories
# The key is the path to the predictions CSV, and the value is the output subdirectory name
EVALUATION_CONFIGS = {
    Path("eval_all_ocr") / "fpo" / "results.csv": "fpo",
    Path("eval_all_ocr") / "paddle" / "results.csv": "paddle",
}

def run_ocr_evaluation(predictions_file: Path, output_base_dir: Path, output_sub_dir: str):
    """Runs the ocr_evaluation.py script for a given predictions file."""
    output_dir = output_base_dir / output_sub_dir
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

    print(f"\n--- Running evaluation for {predictions_file.name} ---")
    print(f"Output will be saved to: {output_dir}")

    command = [
        "python",
        str(OCR_EVALUATION_SCRIPT),
        "--predictions", str(predictions_file),
        "--ground_truth", str(GROUND_TRUTH_FILE),
        "--output_dir", str(output_dir),
        "--max_confusion_classes", "100",  # Increased to show more characters
        "--no_normalize"  # Preserve distinction between O/0, I/1, S/5
    ]

    try:
        # Run the command
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        print("STDOUT:")
        print(process.stdout)
        if process.stderr:
            print("STDERR:")
            print(process.stderr)
        print(f"Evaluation for {predictions_file.name} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation for {predictions_file.name}:")
        print(f"Command: {e.cmd}")
        print(f"Return Code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
    except FileNotFoundError:
        print(f"Error: Python or {OCR_EVALUATION_SCRIPT} not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    print("Starting all OCR evaluations...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_run_output_dir = Path(f"eval_all_ocr_{timestamp}")
    current_run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"All evaluation results for this run will be saved in: {current_run_output_dir}")

    if not OCR_EVALUATION_SCRIPT.exists():
        print(f"Error: OCR evaluation script not found at {OCR_EVALUATION_SCRIPT}. Please ensure it's in the correct path.")
        return

    if not GROUND_TRUTH_FILE.exists():
        print(f"Error: Ground truth file not found at {GROUND_TRUTH_FILE}. Please check the path.")
        return

    for predictions_file, output_sub_dir in EVALUATION_CONFIGS.items():
        if not predictions_file.exists():
            print(f"Warning: Predictions file not found: {predictions_file}. Skipping this evaluation.")
            continue
        run_ocr_evaluation(predictions_file, current_run_output_dir, output_sub_dir)

    print("\nAll OCR evaluations finished.")

if __name__ == "__main__":
    main() 