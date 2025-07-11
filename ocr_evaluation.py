#!/usr/bin/env python3
"""
OCR Evaluation Script

This script evaluates OCR performance by comparing predicted results with ground truth
and generates comprehensive metrics including CER, plate-level accuracy, precision,
recall, F1 score, confusion matrix, and latency analysis.

Usage:
    python scripts/ocr_evaluation.py --predictions results.csv --ground_truth annotations.csv --output_dir evaluation_results
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from loguru import logger


@dataclass
class EvaluationMetrics:
    """Data class for storing evaluation metrics."""
    # Character-level metrics
    cer: float
    char_precision: float
    char_recall: float
    char_f1: float
    
    # Plate-level metrics
    plate_accuracy: float
    plate_precision: float
    plate_recall: float
    plate_f1: float
    
    # Detection metrics
    detection_rate: float
    false_positive_rate: float
    false_negative_rate: float
    
    # Performance metrics
    avg_latency: float
    median_latency: float
    min_latency: float
    max_latency: float
    throughput: float  # plates per second
    
    # Dataset statistics
    total_samples: int
    detected_samples: int
    correct_detections: int
    false_positives: int
    false_negatives: int
    
    # Character statistics
    total_characters: int
    correct_characters: int
    insertion_errors: int
    deletion_errors: int
    substitution_errors: int


class OCREvaluator:
    """Comprehensive OCR evaluation system."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config if config is not None else {}
        self.case_sensitive = self.config.get("case_sensitive", False)
        self.normalize_text = self.config.get("normalize_text", True)
        self.ignore_chars = set(self.config.get("ignore_chars", ["-", ".", "_", " "]))
        
    def normalize_plate_text(self, text: str) -> str:
        """Normalize plate text for comparison."""
        if not text:
            return ""
        
        # Remove ignored characters
        for char in self.ignore_chars:
            text = text.replace(char, "")
        
        # Case normalization
        if not self.case_sensitive:
            text = text.upper()
        
        # Additional normalization
        if self.normalize_text:
            # Replace similar looking characters
            text = text.replace("O", "0")  # Letter O to digit 0
            text = text.replace("I", "1")  # Letter I to digit 1
            text = text.replace("S", "5")  # Sometimes S looks like 5
            
        return text.strip()
    
    def calculate_cer(self, predicted: str, ground_truth: str) -> Tuple[float, Dict[str, int]]:
        """
        Calculate Character Error Rate (CER) using edit distance.
        
        Returns:
            Tuple of (CER, error_counts) where error_counts contains
            insertion, deletion, and substitution counts.
        """
        pred = self.normalize_plate_text(predicted)
        gt = self.normalize_plate_text(ground_truth)
        
        if len(gt) == 0:
            return 1.0 if len(pred) > 0 else 0.0, {"insertion": len(pred), "deletion": 0, "substitution": 0}
        
        # Dynamic programming for edit distance with traceback
        dp = np.zeros((len(pred) + 1, len(gt) + 1), dtype=int)
        
        # Initialize base cases
        for i in range(len(pred) + 1):
            dp[i][0] = i
        for j in range(len(gt) + 1):
            dp[0][j] = j
        
        # Fill the DP table
        for i in range(1, len(pred) + 1):
            for j in range(1, len(gt) + 1):
                if pred[i-1] == gt[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # deletion
                        dp[i][j-1],    # insertion
                        dp[i-1][j-1]   # substitution
                    )
        
        # Traceback to count error types
        i, j = len(pred), len(gt)
        insertions = deletions = substitutions = 0
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and pred[i-1] == gt[j-1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                substitutions += 1
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                deletions += 1
                i -= 1
            elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
                insertions += 1
                j -= 1
        
        cer = dp[len(pred)][len(gt)] / len(gt)
        error_counts = {
            "insertion": insertions,
            "deletion": deletions,
            "substitution": substitutions
        }
        
        return cer, error_counts
    
    def get_character_alignments(self, predicted: str, ground_truth: str) -> List[Tuple[Optional[str], Optional[str]]]:
        """
        Get character alignments using edit distance algorithm.
        
        Returns:
            List of (predicted_char, ground_truth_char) tuples.
            None values indicate insertions or deletions.
        """
        pred = self.normalize_plate_text(predicted)
        gt = self.normalize_plate_text(ground_truth)
        
        if len(gt) == 0:
            return [(char, None) for char in pred]
        if len(pred) == 0:
            return [(None, char) for char in gt]
        
        # Dynamic programming for edit distance
        dp = np.zeros((len(pred) + 1, len(gt) + 1), dtype=int)
        
        # Initialize base cases
        for i in range(len(pred) + 1):
            dp[i][0] = i
        for j in range(len(gt) + 1):
            dp[0][j] = j
        
        # Fill the DP table
        for i in range(1, len(pred) + 1):
            for j in range(1, len(gt) + 1):
                if pred[i-1] == gt[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # deletion
                        dp[i][j-1],    # insertion
                        dp[i-1][j-1]   # substitution
                    )
        
        # Traceback to get alignments
        alignments = []
        i, j = len(pred), len(gt)
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and pred[i-1] == gt[j-1]:
                # Match
                alignments.append((pred[i-1], gt[j-1]))
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                # Substitution
                alignments.append((pred[i-1], gt[j-1]))
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                # Deletion (character in predicted but not in ground truth)
                alignments.append((pred[i-1], None))
                i -= 1
            elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
                # Insertion (character in ground truth but not in predicted)
                alignments.append((None, gt[j-1]))
                j -= 1
        
        # Reverse because we built it backwards
        alignments.reverse()
        return alignments
    
    def is_exact_match(self, predicted: str, ground_truth: str) -> bool:
        """Check if prediction exactly matches ground truth."""
        return self.normalize_plate_text(predicted) == self.normalize_plate_text(ground_truth)
    
    def load_csv_data(self, csv_path: Path) -> List[Dict]:
        """Load data from CSV file."""
        data = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row)
            logger.info(f"Loaded {len(data)} records from {csv_path}")
        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}")
            raise
        
        return data
    
    def align_datasets(self, predictions: List[Dict], ground_truth: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """Align predictions with ground truth based on image path."""
        # Create lookup for ground truth with multiple keys for flexible matching
        gt_lookup = {}
        gt_filename_lookup = {}
        
        for gt_item in ground_truth:
            # Normalize path for comparison
            img_path = gt_item.get("image_path", "").replace("\\", "/")
            gt_lookup[img_path] = gt_item
            
            # Also create filename-only lookup
            filename = Path(img_path).name
            if filename not in gt_filename_lookup:
                gt_filename_lookup[filename] = []
            gt_filename_lookup[filename].append(gt_item)
            
            # Handle cases where ground truth has prefix like "valid/" or "CarTGMTCrop_"
            if "/" in img_path:
                filename_no_prefix = img_path.split("/")[-1]
                if filename_no_prefix.startswith("CarTGMTCrop_"):
                    base_filename = filename_no_prefix[12:]  # Remove "CarTGMTCrop_" prefix
                    if base_filename not in gt_filename_lookup:
                        gt_filename_lookup[base_filename] = []
                    gt_filename_lookup[base_filename].append(gt_item)
        
        aligned_pairs = []
        missing_gt = []
        
        for pred_item in predictions:
            pred_path = pred_item.get("image_path", "").replace("\\", "/")
            pred_filename = Path(pred_path).name
            
            found = False
            
            # Try exact path match first
            if pred_path in gt_lookup:
                aligned_pairs.append((pred_item, gt_lookup[pred_path]))
                found = True
            # Try exact filename match
            elif pred_filename in gt_filename_lookup:
                # If multiple matches, take the first one
                aligned_pairs.append((pred_item, gt_filename_lookup[pred_filename][0]))
                found = True
            # Try with CarTGMTCrop prefix
            elif f"CarTGMTCrop_{pred_filename}" in gt_filename_lookup:
                aligned_pairs.append((pred_item, gt_filename_lookup[f"CarTGMTCrop_{pred_filename}"][0]))
                found = True
            # Try partial matching for similar filenames
            else:
                # Look for ground truth files that contain the prediction filename (without extension)
                pred_base = Path(pred_filename).stem
                for gt_filename, gt_items in gt_filename_lookup.items():
                    if pred_base in gt_filename or Path(gt_filename).stem in pred_base:
                        aligned_pairs.append((pred_item, gt_items[0]))
                        found = True
                        break
            
            if not found:
                missing_gt.append(pred_path)
        
        logger.info(f"Aligned {len(aligned_pairs)} prediction-groundtruth pairs")
        if missing_gt:
            logger.warning(f"Missing ground truth for {len(missing_gt)} predictions")
            for path in missing_gt[:5]:  # Show first 5
                logger.warning(f"  Missing GT: {path}")
        
        return aligned_pairs
    
    def evaluate_predictions(self, predictions: List[Dict], ground_truth: List[Dict]) -> EvaluationMetrics:
        """Evaluate predictions against ground truth."""
        # Align datasets
        aligned_pairs = self.align_datasets(predictions, ground_truth)
        
        if not aligned_pairs:
            raise ValueError("No aligned prediction-groundtruth pairs found")
        
        # Initialize counters
        total_samples = len(aligned_pairs)
        detected_samples = 0
        correct_detections = 0
        false_positives = 0
        false_negatives = 0
        
        total_characters = 0
        correct_characters = 0
        total_insertions = 0
        total_deletions = 0
        total_substitutions = 0
        
        total_cer = 0.0
        latencies = []
        
        exact_matches = 0
        
        # Process each aligned pair
        for pred_item, gt_item in aligned_pairs:
            pred_text = pred_item.get("plate_text", "").strip()
            gt_text = gt_item.get("plate_text", "").strip()
            
            # Handle latency if available
            if "processing_time" in pred_item:
                try:
                    latency = float(pred_item["processing_time"])
                    latencies.append(latency)
                except (ValueError, TypeError):
                    pass
            
            # Classification: detection vs no detection
            has_prediction = bool(pred_text)
            has_ground_truth = bool(gt_text)
            
            if has_prediction:
                detected_samples += 1
            
            if has_ground_truth and has_prediction:
                # True positive case - calculate accuracy metrics
                cer, error_counts = self.calculate_cer(pred_text, gt_text)
                total_cer += cer
                
                total_insertions += error_counts["insertion"]
                total_deletions += error_counts["deletion"]
                total_substitutions += error_counts["substitution"]
                
                # Character-level accuracy
                gt_normalized = self.normalize_plate_text(gt_text)
                total_characters += len(gt_normalized)
                correct_characters += len(gt_normalized) - (
                    error_counts["insertion"] + error_counts["deletion"] + error_counts["substitution"]
                )
                
                # Exact match
                if self.is_exact_match(pred_text, gt_text):
                    correct_detections += 1
                    exact_matches += 1
                
            elif has_ground_truth and not has_prediction:
                # False negative
                false_negatives += 1
                total_cer += 1.0  # Maximum CER for missed detection
                total_characters += len(self.normalize_plate_text(gt_text))
                
            elif not has_ground_truth and has_prediction:
                # False positive
                false_positives += 1
                
            # If neither has text, it's a true negative (correct non-detection)
        
        # Calculate metrics
        avg_cer = total_cer / total_samples if total_samples > 0 else 1.0
        
        # Character-level metrics
        char_accuracy = correct_characters / total_characters if total_characters > 0 else 0.0
        char_precision = char_recall = char_f1 = char_accuracy  # Simplified for character level
        
        # Plate-level metrics
        plate_accuracy = exact_matches / total_samples if total_samples > 0 else 0.0
        
        # Detection metrics (treating detection as binary classification)
        true_positives = correct_detections
        detected_with_gt = sum(1 for _, gt in aligned_pairs if gt.get("plate_text", "").strip())
        
        plate_precision = true_positives / detected_samples if detected_samples > 0 else 0.0
        plate_recall = true_positives / detected_with_gt if detected_with_gt > 0 else 0.0
        plate_f1 = (2 * plate_precision * plate_recall) / (plate_precision + plate_recall) if (plate_precision + plate_recall) > 0 else 0.0
        
        detection_rate = detected_samples / total_samples if total_samples > 0 else 0.0
        false_positive_rate = false_positives / total_samples if total_samples > 0 else 0.0
        false_negative_rate = false_negatives / total_samples if total_samples > 0 else 0.0
        
        # Latency metrics
        if latencies:
            avg_latency = float(np.mean(latencies))
            median_latency = float(np.median(latencies))
            min_latency = float(np.min(latencies))
            max_latency = float(np.max(latencies))
            throughput = 1.0 / avg_latency if avg_latency > 0 else 0.0
        else:
            avg_latency = median_latency = min_latency = max_latency = throughput = 0.0
        
        return EvaluationMetrics(
            cer=avg_cer,
            char_precision=char_precision,
            char_recall=char_recall,
            char_f1=char_f1,
            plate_accuracy=plate_accuracy,
            plate_precision=plate_precision,
            plate_recall=plate_recall,
            plate_f1=plate_f1,
            detection_rate=detection_rate,
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
            avg_latency=avg_latency,
            median_latency=median_latency,
            min_latency=min_latency,
            max_latency=max_latency,
            throughput=throughput,
            total_samples=total_samples,
            detected_samples=detected_samples,
            correct_detections=correct_detections,
            false_positives=false_positives,
            false_negatives=false_negatives,
            total_characters=total_characters,
            correct_characters=correct_characters,
            insertion_errors=total_insertions,
            deletion_errors=total_deletions,
            substitution_errors=total_substitutions
        )
    
    def generate_confusion_matrix(self, predictions: List[Dict], ground_truth: List[Dict], 
                                output_dir: Path, max_classes: int = 50) -> None:
        """Generate confusion matrix for character-level analysis."""
        aligned_pairs = self.align_datasets(predictions, ground_truth)
        
        # Collect character substitutions using proper alignment
        pred_chars = []
        gt_chars = []
        
        for pred_item, gt_item in aligned_pairs:
            pred_text = pred_item.get("plate_text", "")
            gt_text = gt_item.get("plate_text", "")
            
            # Get character alignments using edit distance
            alignments = self.get_character_alignments(pred_text, gt_text)
            
            # Only include substitutions and matches (not insertions/deletions)
            for pred_char, gt_char in alignments:
                if pred_char is not None and gt_char is not None:
                    pred_chars.append(pred_char)
                    gt_chars.append(gt_char)
        
        if not gt_chars:
            logger.warning("No character alignments found for confusion matrix")
            return
        
        # Get unique characters and limit to most common
        char_counts = Counter(gt_chars)
        all_chars = set(pred_chars + gt_chars)
        most_common_chars = [char for char, _ in char_counts.most_common(max_classes)]
        
        # Filter to most common characters
        filtered_pred = []
        filtered_gt = []
        for p, g in zip(pred_chars, gt_chars):
            if g in most_common_chars:
                if p in most_common_chars:
                    filtered_pred.append(p)
                    filtered_gt.append(g)
                else:
                    # Skip unusual predicted characters instead of grouping as 'OTHER'
                    # This prevents cluttering the confusion matrix with rare OCR artifacts
                    continue
        
        if not filtered_gt:
            logger.warning("No characters found for confusion matrix after filtering")
            return
        
        # Create confusion matrix
        unique_chars = sorted(ch for ch in set(filtered_gt + filtered_pred) if ch not in {'', ' '})
        cm = confusion_matrix(filtered_gt, filtered_pred, labels=unique_chars)

        # Downsample for large class count if needed (Ultralytics logic)
        nc = len(unique_chars)
        if nc >= 100:
            k = max(2, nc // 60)  # step size for downsampling
            keep_idx = slice(None, None, k)  # create slice instead of array
            unique_chars_downsampled = [unique_chars[i] for i in range(0, nc, k)]
            cm = cm[keep_idx, :][:, keep_idx]
            nc = len(unique_chars_downsampled) # Update nc to the new number of classes
        else:
            unique_chars_downsampled = unique_chars


        # ------------------------------------------------------------------
        # Plotting setup – mirror Ultralytics best-practice aesthetics
        # ------------------------------------------------------------------
        # Always show annotations for non-zero values, but adjust font size based on matrix size
        annot_cells = True  # Always show annotations, we'll handle zero hiding separately

        # Apply a clean seaborn style
        sns.set(style="white", font_scale=0.8) # Keep original font_scale for general style

        # Dynamic font sizes and margins
        tick_fontsize = max(6, 15 - 0.1 * nc)
        label_fontsize = max(6, 12 - 0.1 * nc)
        title_fontsize = max(6, 12 - 0.1 * nc)
        annot_fontsize = max(4, 8 - 0.05 * nc)  # Dynamic annotation font size
        bottom_margin = max(0.1, 0.25 - 0.001 * nc) # dynamic bottom margin for x-axis labels

        # Normalize confusion matrix for better visualization of proportions (column-wise normalization)
        col_sum_normalized = cm.sum(axis=0, keepdims=True)
        cm_normalized = cm.astype('float') / np.where(col_sum_normalized == 0, 1, col_sum_normalized)
        cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero for columns with no true labels
        cm_normalized[cm_normalized < 0.005] = np.nan # Hide very small values for cleaner annotation

        # --- Improved visualization settings ---
        # 1. Unnormalized (count) confusion matrix ----------------------------------
        fig, ax = plt.subplots(figsize=(12, 10)) # Use fig, ax = plt.subplots() for more control

        # Use a logarithmic colour scale so small non-zero counts are still visible.
        # Mask true zeros so they stay white.
        mask_counts = cm == 0
        if cm.max() > 1:
            norm_counts = mcolors.LogNorm(vmin=1, vmax=cm.max())
            # For log scale, create nice tick values
            vmin_log, vmax_log = 1, cm.max()
            ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
            ticks = [t for t in ticks if vmin_log <= t <= vmax_log]
            if not ticks: # Fallback if no ticks within range
                ticks = [vmin_log, vmax_log]
            cbar_ticks_unnorm = ticks
        else:
            norm_counts = None  # Fallback to linear if all counts are <=1

            # If linear, ensure min and max are explicitly set for ticks, and add dynamic ticks for intermediate values
            # Ensure at least 2 ticks (min, max) and up to 5 evenly spaced for richer display
            if cm.max() > 0:
                # Generate ticks as integers, ensuring min and max are always included
                min_val = int(cm.min())
                max_val = int(cm.max())
                if min_val == max_val:
                    cbar_ticks_unnorm = [min_val]
                else:
                    # Generate up to 5 intermediate ticks, ensuring they are integers and unique
                    num_intermediate_ticks = min(3, max_val - min_val - 1) # At most 3 intermediate ticks for better spacing
                    if num_intermediate_ticks > 0:
                        intermediate_ticks = np.linspace(min_val, max_val, num_intermediate_ticks + 2)
                        # Convert to integers, remove duplicates and sort, ensuring min/max are present
                        intermediate_ticks = sorted(list(set([int(t) for t in intermediate_ticks])))
                        cbar_ticks_unnorm = [t for t in intermediate_ticks if min_val <= t <= max_val]
                    else:
                        cbar_ticks_unnorm = [min_val, max_val] # Only min and max if no space for intermediate

                # Ensure 0 is included if it's the minimum value
                if min_val == 0 and 0 not in cbar_ticks_unnorm:
                    cbar_ticks_unnorm.insert(0, 0)
            else:
                cbar_ticks_unnorm = [0] # If all zeros

        # Create a colormap that starts with white so zeros/background remain white
        # Use "Blues" colormap, following Ultralytics best practices for confusion matrices
        base_cmap = plt.colormaps.get_cmap('Blues').resampled(256)
        new_colors = base_cmap(np.linspace(0.25, 1, 256))  # Skip very light shades
        new_colors = np.vstack((np.array([1, 1, 1, 1]), new_colors))  # Prepend white
        white_cmap_counts = mcolors.ListedColormap(new_colors)

        # Custom colorbar formatting for regular numbers instead of scientific notation
        # The logic for ticks_list_unnorm is already handled above based on norm_counts
        # We will use cbar_kws for setting the ticks and then update_ticks()
        
        # Plot unnormalized heatmap - always show annotations for non-zero values
        sns.heatmap(
            cm,
            annot=True,  # Always show annotations
            fmt='d',
            cmap=white_cmap_counts,
            norm=norm_counts,
            mask=mask_counts,
            linewidths=.5,
            linecolor='lightgrey',
            cbar_kws={'label': 'Count', 'shrink': 0.8, 'pad': 0.02, 'ticks': cbar_ticks_unnorm},
            annot_kws={'fontsize': annot_fontsize},
            xticklabels=unique_chars_downsampled,
            yticklabels=unique_chars_downsampled,
            square=True,
            ax=ax # Pass the axes explicitly
        )

        cbar = ax.collections[0].colorbar

        cbar.ax.tick_params(labelsize=tick_fontsize, pad=5)  # Smaller font for ticks, increased pad
        cbar.ax.yaxis.set_ticks_position('right')  # Ensure ticks are on the right
        cbar.ax.yaxis.set_label_position('right')  # Ensure label is on the right
        cbar.update_ticks()  # Force update of ticks and labels
        
        # Hide all spines
        for s in ['left', 'right', 'top', 'bottom']:
            ax.spines[s].set_visible(False)
        for s in ['left', 'right', 'top', 'bottom']:
             cbar.ax.spines[s].set_visible(False) # Hide colorbar spines

        # Hide zero annotations while keeping non-zero values visible
        for text in ax.texts:
            if text.get_text() == '0':
                text.set_text('')

        # Force all tick labels to be displayed
        centers = np.arange(len(unique_chars_downsampled)) + 0.5
        ax.set_xticks(centers)
        ax.set_xticklabels(unique_chars_downsampled, rotation=45, ha='right', fontsize=tick_fontsize)
        ax.set_yticks(centers)
        ax.set_yticklabels(unique_chars_downsampled, rotation=0, ha='right', fontsize=tick_fontsize)

        ax.set_title('Character-level Confusion Matrix (Counts)', fontsize=title_fontsize, pad=20)
        ax.set_xlabel('Predicted Characters', fontsize=label_fontsize, labelpad=10)
        ax.set_ylabel('True Characters', fontsize=label_fontsize, labelpad=10)
        ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False, labelsize=tick_fontsize)
        ax.tick_params(axis="y", left=True, right=False, labelleft=True, labelright=False, labelsize=tick_fontsize)

        # Move tight_layout here to ensure it accounts for colorbar adjustments
        fig.subplots_adjust(left=0, right=0.84, top=0.94, bottom=bottom_margin) # Adjust layout to ensure equal margins

        output_path_unnormalized = output_dir / "confusion_matrix_characters_unnormalized.png"
        plt.savefig(output_path_unnormalized, dpi=300, bbox_inches='tight')
        plt.close(fig) # Close the specific figure

        # 2. Normalized confusion matrix -------------------------------------------
        fig_norm, ax_norm = plt.subplots(figsize=(12, 10)) # New figure and axes for normalized

        # Use a power-law normalisation (gamma<1) to emphasise low values but keep scale linear near the high end.
        norm_ratio = mcolors.PowerNorm(gamma=0.4, vmin=0.0, vmax=1.0)
        mask_norm = np.isnan(cm_normalized) # Mask NaN values that were set to hide 0.00 entries

        # Reuse the custom colormap for normalized heatmap
        white_cmap_norm = white_cmap_counts

        # Custom ticks for normalized matrix
        # Ensure 0.0 and 1.0 are always included in ticks and well-distributed
        norm_ticks = sorted(list(set([norm_ratio.vmin, 0.2, 0.4, 0.6, 0.8, norm_ratio.vmax])))
        # Ensure 0.0 and 1.0 are always included in ticks if they aren't already
        if 0.0 not in norm_ticks: norm_ticks.insert(0, 0.0)
        if 1.0 not in norm_ticks: norm_ticks.append(1.0)
        norm_ticks = sorted(list(set(norm_ticks))) # Remove duplicates and sort

        norm_cbar_kws = {'label': 'Proportion', 'shrink': 0.8, 'pad': 0.04, 'ticks': norm_ticks} # Increased pad

        sns.heatmap(
            cm_normalized,
            annot=True,  # Always show annotations
            fmt='.2f',
            cmap=white_cmap_norm,
            norm=norm_ratio,
            mask=mask_norm,
            linewidths=.5,
            linecolor='lightgrey',
            cbar_kws=norm_cbar_kws,
            annot_kws={'fontsize': annot_fontsize},
            xticklabels=unique_chars_downsampled,
            yticklabels=unique_chars_downsampled,
            square=True,
            ax=ax_norm # Pass the axes explicitly
        )

        cbar_norm = ax_norm.collections[0].colorbar # Get the colorbar for the normalized plot

        # Removed redundant explicit tick setting, rely on cbar_kws for ticks
        # Ensure the colorbar displays the full range from vmin to vmax
        ax_norm.collections[0].set_clim(norm_ratio.vmin, norm_ratio.vmax)

        cbar_norm.ax.tick_params(labelsize=tick_fontsize, pad=5)  # Smaller font for ticks, increased pad
        cbar_norm.ax.yaxis.set_ticks_position('right')  # Ensure ticks are on the right
        cbar_norm.ax.yaxis.set_label_position('right')  # Ensure label is on the right
        cbar_norm.update_ticks()  # Force update of ticks and labels
        
        # Hide all spines
        for s in ['left', 'right', 'top', 'bottom']:
            ax_norm.spines[s].set_visible(False)
        for s in ['left', 'right', 'top', 'bottom']:
             cbar_norm.ax.spines[s].set_visible(False) # Hide colorbar spines

        # Improve annotation readability – remove leading 0 and hide 0.00 entirely
        # Ultralytics uses 0.45 threshold relative to max_val (1.0 for normalized)
        color_threshold_norm = 0.45 * 1.0 # For normalized matrix, max value is 1.0
        for text in ax_norm.texts: # Iterate over texts on ax_norm
            txt = text.get_text()
            # Hide zeros and very small values that were set to NaN and thus not annotated by sns.heatmap
            if txt in ('0.00', '0', ''): # sns.heatmap handles NaNs by not annotating. This is for explicit 0.00
                text.set_text('')
                continue
            # Remove leading zero for compactness
            if txt.startswith('0.'):
                text.set_text(txt[1:])
                txt = txt[1:]
            # Dynamically adjust text colour for readability
            try:
                val = float(txt if not txt.startswith('.') else '0' + txt)
            except ValueError:
                val = 0.0 # Default if conversion fails
            text.set_color('white' if val > color_threshold_norm else 'black')

        # Force all tick labels to be displayed
        centers_norm = np.arange(len(unique_chars_downsampled)) + 0.5
        ax_norm.set_xticks(centers_norm)
        ax_norm.set_xticklabels(unique_chars_downsampled, rotation=45, ha='right', fontsize=tick_fontsize)
        ax_norm.set_yticks(centers_norm)
        ax_norm.set_yticklabels(unique_chars_downsampled, rotation=0, ha='right', fontsize=tick_fontsize)

        ax_norm.set_title('Character-level Confusion Matrix (Normalised)', fontsize=title_fontsize, pad=20)
        ax_norm.set_xlabel('Predicted Characters', fontsize=label_fontsize, labelpad=10)
        ax_norm.set_ylabel('True Characters', fontsize=label_fontsize, labelpad=10)
        ax_norm.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False, labelsize=tick_fontsize)
        ax_norm.tick_params(axis="y", left=True, right=False, labelleft=True, labelright=False, labelsize=tick_fontsize)

        plt.tight_layout() # Removed fig.subplots_adjust here to allow bbox_inches='tight'

        output_path_normalized = output_dir / "confusion_matrix_characters_normalized.png"
        plt.savefig(output_path_normalized, dpi=300, bbox_inches='tight')
        plt.close(fig_norm) # Close the specific figure
    
    def generate_error_analysis(self, predictions: List[Dict], ground_truth: List[Dict], 
                              output_dir: Path) -> None:
        """Generate detailed error analysis."""
        aligned_pairs = self.align_datasets(predictions, ground_truth)
        
        error_analysis = {
            "common_substitutions": Counter(),
            "common_insertions": Counter(),
            "common_deletions": Counter(),
            "error_examples": [],
            "by_plate_length": defaultdict(list),
            "by_confidence": defaultdict(list)
        }
        
        for pred_item, gt_item in aligned_pairs:
            pred_text = pred_item.get("plate_text", "").strip()
            gt_text = gt_item.get("plate_text", "").strip()
            confidence = float(pred_item.get("confidence", 0.0))
            
            if not gt_text:
                continue
                
            pred_norm = self.normalize_plate_text(pred_text)
            gt_norm = self.normalize_plate_text(gt_text)
            
            cer, error_counts = self.calculate_cer(pred_text, gt_text)
            
            # Categorize by plate length
            length_category = f"{len(gt_norm)} chars"
            error_analysis["by_plate_length"][length_category].append(cer)
            
            # Categorize by confidence
            conf_category = f"{int(confidence * 10) / 10:.1f}"
            error_analysis["by_confidence"][conf_category].append(cer)
            
            # Collect error examples
            if cer > 0:
                error_analysis["error_examples"].append({
                    "image_path": pred_item.get("image_path", ""),
                    "predicted": pred_text,
                    "ground_truth": gt_text,
                    "cer": cer,
                    "confidence": confidence,
                    "error_counts": error_counts
                })
            
            # Simple character-level analysis for common errors
            if pred_norm != gt_norm:
                # This is a simplified approach - for detailed analysis would need alignment
                for i, (p, g) in enumerate(zip(pred_norm, gt_norm)):
                    if p != g:
                        error_analysis["common_substitutions"][(g, p)] += 1
        
        # Save error analysis
        output_path = output_dir / "error_analysis.json"
        
        # Convert Counter objects to regular dicts for JSON serialization
        # Convert tuple keys to strings for JSON compatibility
        common_substitutions = {}
        for (orig, pred), count in error_analysis["common_substitutions"].items():
            common_substitutions[f"{orig}->{pred}"] = count
        
        serializable_analysis = {
            "common_substitutions": common_substitutions,
            "common_insertions": dict(error_analysis["common_insertions"]),
            "common_deletions": dict(error_analysis["common_deletions"]),
            "error_examples": error_analysis["error_examples"][:50],  # Limit to first 50
            "by_plate_length": {k: {"mean_cer": float(np.mean(v)), "std_cer": float(np.std(v)), "count": len(v)} 
                              for k, v in error_analysis["by_plate_length"].items()},
            "by_confidence": {k: {"mean_cer": float(np.mean(v)), "std_cer": float(np.std(v)), "count": len(v)} 
                            for k, v in error_analysis["by_confidence"].items()}
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_analysis, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Error analysis saved to {output_path}")
    
    def generate_performance_plots(self, metrics: EvaluationMetrics, latencies: List[float], 
                                 output_dir: Path) -> None:
        """Generate performance visualization plots."""
        # Metrics summary plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy metrics
        accuracy_metrics = ['Plate Accuracy', 'Char Precision', 'Char Recall', 'Char F1']
        accuracy_values = [metrics.plate_accuracy, metrics.char_precision, 
                          metrics.char_recall, metrics.char_f1]
        
        axes[0, 0].bar(accuracy_metrics, accuracy_values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        axes[0, 0].set_title('Accuracy Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(accuracy_values):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Error types
        error_types = ['CER', 'FP Rate', 'FN Rate']
        error_values = [metrics.cer, metrics.false_positive_rate, metrics.false_negative_rate]
        
        axes[0, 1].bar(error_types, error_values, color=['#E74C3C', '#E67E22', '#F39C12'])
        axes[0, 1].set_title('Error Rates')
        axes[0, 1].set_ylabel('Rate')
        axes[0, 1].set_ylim(0, max(error_values) * 1.1 if error_values else 1)
        for i, v in enumerate(error_values):
            axes[0, 1].text(i, v + max(error_values) * 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Latency distribution
        if latencies:
            axes[1, 0].hist(latencies, bins=30, alpha=0.7, color='#3498DB', edgecolor='black')
            axes[1, 0].axvline(metrics.avg_latency, color='red', linestyle='--', 
                              label=f'Mean: {metrics.avg_latency:.4f}s')
            axes[1, 0].axvline(metrics.median_latency, color='green', linestyle='--', 
                              label=f'Median: {metrics.median_latency:.4f}s')
            axes[1, 0].set_title('Latency Distribution')
            axes[1, 0].set_xlabel('Processing Time (seconds)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
        
        # Performance summary
        perf_metrics = ['Throughput\n(plates/s)', 'Detection\nRate', 'Avg Latency\n(ms)']
        perf_values = [metrics.throughput, metrics.detection_rate, metrics.avg_latency * 1000]
        
        # Normalize values for visualization
        normalized_values = [
            metrics.throughput / 1000 if metrics.throughput > 0 else 0,  # Scale down throughput
            metrics.detection_rate,
            min(metrics.avg_latency * 10, 1.0)  # Scale up latency but cap at 1
        ]
        
        bars = axes[1, 1].bar(perf_metrics, normalized_values, color=['#27AE60', '#8E44AD', '#D35400'])
        axes[1, 1].set_title('Performance Summary (Normalized)')
        axes[1, 1].set_ylabel('Normalized Score')
        
        # Add actual values as text
        for i, (bar, actual) in enumerate(zip(bars, perf_values)):
            height = bar.get_height()
            if i == 0:  # Throughput
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{actual:.1f}', ha='center', va='bottom')
            elif i == 1:  # Detection rate
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{actual:.3f}', ha='center', va='bottom')
            else:  # Latency
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{actual:.1f}ms', ha='center', va='bottom')
        
        plt.tight_layout()
        output_path = output_dir / "performance_metrics.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance plots saved to {output_path}")
    
    def save_metrics_report(self, metrics: EvaluationMetrics, output_dir: Path) -> None:
        """Save comprehensive metrics report."""
        report = {
            "summary": {
                "Character Error Rate (CER)": f"{metrics.cer:.4f}",
                "Plate-level Accuracy": f"{metrics.plate_accuracy:.4f}",
                "Plate-level Precision": f"{metrics.plate_precision:.4f}",
                "Plate-level Recall": f"{metrics.plate_recall:.4f}",
                "Plate-level F1 Score": f"{metrics.plate_f1:.4f}",
                "Detection Rate": f"{metrics.detection_rate:.4f}",
                "False Positive Rate": f"{metrics.false_positive_rate:.4f}",
                "False Negative Rate": f"{metrics.false_negative_rate:.4f}"
            },
            "performance": {
                "Average Latency (ms)": f"{metrics.avg_latency * 1000:.2f}",
                "Median Latency (ms)": f"{metrics.median_latency * 1000:.2f}",
                "Min Latency (ms)": f"{metrics.min_latency * 1000:.2f}",
                "Max Latency (ms)": f"{metrics.max_latency * 1000:.2f}",
                "Throughput (plates/sec)": f"{metrics.throughput:.2f}"
            },
            "detailed_metrics": asdict(metrics)
        }
        
        # Save JSON report
        json_path = output_dir / "evaluation_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Save human-readable report
        txt_path = output_dir / "evaluation_report.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("OCR EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("SUMMARY METRICS\n")
            f.write("-" * 20 + "\n")
            for key, value in report["summary"].items():
                f.write(f"{key:.<30} {value:>10}\n")
            
            f.write(f"\nPERFORMANCE METRICS\n")
            f.write("-" * 20 + "\n")
            for key, value in report["performance"].items():
                f.write(f"{key:.<30} {value:>10}\n")
            
            f.write(f"\nDATASET STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"{'Total Samples':.<30} {metrics.total_samples:>10}\n")
            f.write(f"{'Detected Samples':.<30} {metrics.detected_samples:>10}\n")
            f.write(f"{'Correct Detections':.<30} {metrics.correct_detections:>10}\n")
            f.write(f"{'False Positives':.<30} {metrics.false_positives:>10}\n")
            f.write(f"{'False Negatives':.<30} {metrics.false_negatives:>10}\n")
            
            f.write(f"\nCHARACTER-LEVEL ANALYSIS\n")
            f.write("-" * 20 + "\n")
            f.write(f"{'Total Characters':.<30} {metrics.total_characters:>10}\n")
            f.write(f"{'Correct Characters':.<30} {metrics.correct_characters:>10}\n")
            f.write(f"{'Insertion Errors':.<30} {metrics.insertion_errors:>10}\n")
            f.write(f"{'Deletion Errors':.<30} {metrics.deletion_errors:>10}\n")
            f.write(f"{'Substitution Errors':.<30} {metrics.substitution_errors:>10}\n")
        
        logger.info(f"Evaluation reports saved to {json_path} and {txt_path}")


def main():
    """Main function to handle command line arguments and execute evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate OCR performance with comprehensive metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python scripts/ocr_evaluation.py --predictions results.csv --ground_truth annotations.csv --output_dir eval_results
  
  # Case-sensitive evaluation with custom normalization
  python scripts/ocr_evaluation.py --predictions results.csv --ground_truth gt.csv --output_dir eval --case_sensitive --no_normalize
  
  # Evaluate specific dataset
  python scripts/ocr_evaluation.py --predictions output/cartgmt_results.csv --ground_truth lp_data/preprocessed_dataset/valid_anotaciones.csv --output_dir evaluation
        """
    )
    
    parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        help='CSV file with OCR predictions (from ocr_dataset_processor.py)'
    )
    
    parser.add_argument(
        '--ground_truth',
        type=str,
        required=True,
        help='CSV file with ground truth annotations'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluation_results',
        help='Output directory for evaluation results (default: evaluation_results)'
    )
    
    parser.add_argument(
        '--case_sensitive',
        action='store_true',
        help='Perform case-sensitive evaluation'
    )
    
    parser.add_argument(
        '--no_normalize',
        action='store_true',
        help='Disable text normalization (O->0, I->1, etc.)'
    )
    
    parser.add_argument(
        '--ignore_chars',
        nargs='*',
        default=['-', '.', '_', ' '],
        help='Characters to ignore during comparison (default: - . _ space)'
    )
    
    parser.add_argument(
        '--max_confusion_classes',
        type=int,
        default=50,
        help='Maximum number of classes for confusion matrix (default: 50)'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    predictions_path = Path(args.predictions)
    ground_truth_path = Path(args.ground_truth)
    output_dir = Path(args.output_dir)
    
    if not predictions_path.exists():
        print(f"Error: Predictions file does not exist: {predictions_path}")
        return 1
    
    if not ground_truth_path.exists():
        print(f"Error: Ground truth file does not exist: {ground_truth_path}")
        return 1
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure evaluator
    config = {
        "case_sensitive": args.case_sensitive,
        "normalize_text": not args.no_normalize,
        "ignore_chars": args.ignore_chars
    }
    
    evaluator = OCREvaluator(config)
    
    try:
        logger.info("Starting OCR evaluation...")
        logger.info(f"Predictions: {predictions_path}")
        logger.info(f"Ground truth: {ground_truth_path}")
        logger.info(f"Output directory: {output_dir}")
        
        # Load data
        predictions = evaluator.load_csv_data(predictions_path)
        ground_truth = evaluator.load_csv_data(ground_truth_path)
        
        # Evaluate predictions
        metrics = evaluator.evaluate_predictions(predictions, ground_truth)
        
        # Extract latencies for visualization
        latencies = []
        for pred in predictions:
            if "processing_time" in pred:
                try:
                    latencies.append(float(pred["processing_time"]))
                except (ValueError, TypeError):
                    pass
        
        # Generate reports and visualizations
        evaluator.save_metrics_report(metrics, output_dir)
        evaluator.generate_performance_plots(metrics, latencies, output_dir)
        evaluator.generate_confusion_matrix(predictions, ground_truth, output_dir, args.max_confusion_classes)
        evaluator.generate_error_analysis(predictions, ground_truth, output_dir)
        
        # Print summary
        print(f"\n>>> OCR EVALUATION RESULTS")
        print("=" * 50)
        print(f"Character Error Rate (CER): {metrics.cer:.4f}")
        print(f"Plate-level Accuracy: {metrics.plate_accuracy:.4f}")
        print(f"Plate-level F1 Score: {metrics.plate_f1:.4f}")
        print(f"Detection Rate: {metrics.detection_rate:.4f}")
        print(f"Average Latency: {metrics.avg_latency*1000:.2f}ms")
        print(f"Throughput: {metrics.throughput:.2f} plates/sec")
        print(f"\n>>> Detailed results saved to: {output_dir}")
        
        logger.info("Evaluation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 