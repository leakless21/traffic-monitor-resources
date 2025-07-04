# Traffic Monitor Resources

This repository contains tools and resources for evaluating and comparing different Optical Character Recognition (OCR) engines and object detection models (for license plates and vehicles) in the context of traffic monitoring. The project provides scripts for processing datasets, performing detailed OCR evaluations, comparing multiple OCR engines, and generating comprehensive visualization reports.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Dataset Processing](#dataset-processing)
  - [OCR Evaluation](#ocr-evaluation)
  - [OCR Engine Comparison](#ocr-engine-comparison)
  - [Running All Visualizations](#running-all-visualizations)
- [Output Structure](#output-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- **OCR Dataset Processing**: Process image datasets using various OCR engines (e.g., FastPlateOCR, PaddleOCR) and export results.
- **Comprehensive OCR Evaluation**: Calculate detailed metrics for OCR performance, including:
  - Character Error Rate (CER)
  - Plate-level accuracy, precision, recall, and F1 score
  - Detection rate, false positive rate, false negative rate
  - Latency (average, median, min, max) and throughput
  - Generation of character-level confusion matrices and error analysis reports.
- **OCR Engine Comparison**: Compare multiple OCR engines side-by-side using various metrics and generate visual comparison plots.
- **Automated Visualization Generation**: Scripts to automatically generate a wide range of performance visualizations for OCR, license plate detection, and vehicle detection models.
- **Structured Results Output**: Organized output directories for evaluation reports, raw data, and visualizations.

## Installation

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/traffic-monitor-resources.git
    cd traffic-monitor-resources
    ```

2.  **Create and activate a Python virtual environment** (recommended):

    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    The project uses `pixi` for environment management. If you have `pixi` installed, it will automatically manage the environment. Otherwise, you can install the dependencies using `pip`:
    ```bash
    pip install -r requirements.txt
    ```
    Additional dependencies for specific OCR engines might be required:
    - For `FastPlateOCR`: `pip install fast-plate-ocr`
    - For `PaddleOCR`: `pip install paddlepaddle paddleocr`

## Usage

The primary way to run the full evaluation suite is via the `run_evaluations.bat` script (for Windows). Individual scripts can also be run for specific tasks.

### Dataset Processing (`ocr_dataset_processor.py`)

This script processes images using a specified OCR engine and saves the results to a CSV file.

```bash
python ocr_dataset_processor.py --input_dir path/to/your/image/dataset --output_csv results.csv --engine fast_plate_ocr --engine_config "{\"lang\": \"en\", \"conf_threshold\": 0.5}"
```

- `--input_dir`: Path to the directory containing images to be processed.
- `--output_csv`: Path to save the OCR results CSV.
- `--engine`: Specify the OCR engine to use (e.g., `fast_plate_ocr`, `paddleocr`).
- `--engine_config`: (Optional) JSON string for engine-specific configuration (e.g., `{"lang": "en", "conf_threshold": 0.5}`).

### OCR Evaluation (`ocr_evaluation.py`)

This script evaluates the performance of an OCR engine against a ground truth dataset.

```bash
python ocr_evaluation.py --predictions path/to/predictions.csv --ground_truth path/to/ground_truth.csv --output_dir ocr_eval_results
```

- `--predictions`: CSV file containing OCR predictions.
- `--ground_truth`: CSV file containing ground truth annotations.
- `--output_dir`: Directory to save evaluation reports and visualizations.

### OCR Engine Comparison (`comparison_evaluation.py`)

This script compares the performance of multiple OCR engines.

```bash
python comparison_evaluation.py \
    --predictions_list results1.csv results2.csv \
    --ground_truth annotations.csv \
    --names Engine1 Engine2 \
    --output_dir comparison_results
```

- `--predictions_list`: Space-separated list of CSV files with predictions from different engines.
- `--ground_truth`: CSV file with ground truth annotations.
- `--names`: Space-separated list of names for each OCR engine, corresponding to the order in `--predictions_list`.
- `--output_dir`: Directory to save comparison reports and plots.

### Running All Visualizations (`evaluation_tools/run_all_visualizations.py`)

This script orchestrates the generation of all evaluation reports and visualizations. It is typically run via `run_evaluations.bat`.

```bash
python evaluation_tools/run_all_visualizations.py
```

This script will generate a comprehensive set of evaluation results, including performance metrics and visualizations for OCR, license plate detection, and vehicle detection, saved within the `Evaluation_Results/` and `eval_all_ocr/` directories.

## Output Structure

The project organizes its output into several key directories:

- `eval_all_ocr/`: Contains results from OCR engine comparisons and individual OCR evaluations.
  - `comparison/`: Reports and plots comparing different OCR engines.
  - `fpo/`: Evaluation results for the FastPlateOCR engine.
  - `paddle/`: Evaluation results for the PaddleOCR engine.
- `Evaluation_Results/`: Stores comprehensive tables and visualizations for various detection and OCR tasks.
  - `comprehensive_tables/`: CSV files containing aggregated performance metrics.
  - `ocr_visualizations/`: Plots and tables specifically for OCR performance.
  - `plate_detection_visualizations/`: Plots and tables related to license plate detection.
  - `vehicle_detection_visualizations/`: Plots and tables related to vehicle detection.
  - `THESIS_VISUALIZATION_SUMMARY.md`: A markdown summary of key visualizations.
- `Results/`: Contains raw detection and training results for different YOLO models (e.g., GhostYOLO, YOLO11n, YOLOv10n, YOLOv5u, YOLOv8n) across license plate and vehicle detection tasks.
  - `License Plate/`: Results for license plate detection models.
  - `Vehicles/`: Results for vehicle detection models.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
