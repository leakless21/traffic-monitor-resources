#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import sys
sys.path.append('.')
from ocr_evaluation import OCREvaluator
from collections import Counter
from sklearn.metrics import confusion_matrix

# Create evaluator
evaluator = OCREvaluator()

# Load some sample data to test
predictions = evaluator.load_csv_data(Path('eval_all_ocr/fpo/results.csv'))[:50]
ground_truth = evaluator.load_csv_data(Path('lp_all_dataset/all_anotaciones.csv'))[:50]

# Debug the character collection process
aligned_pairs = evaluator.align_datasets(predictions, ground_truth)
pred_chars = []
gt_chars = []

for pred_item, gt_item in aligned_pairs:
    pred_text = evaluator.normalize_plate_text(pred_item.get('plate_text', ''))
    gt_text = evaluator.normalize_plate_text(gt_item.get('plate_text', ''))
    
    max_len = max(len(pred_text), len(gt_text))
    pred_padded = pred_text.ljust(max_len, ' ')
    gt_padded = gt_text.ljust(max_len, ' ')
    
    pred_chars.extend(list(pred_padded))
    gt_chars.extend(list(gt_padded))

print(f'Total characters collected: {len(gt_chars)}')
print(f'Unique characters in ground truth: {sorted(set(gt_chars))}')
print(f'Unique characters in predictions: {sorted(set(pred_chars))}')

# Simulate the filtering process
all_chars = set(pred_chars + gt_chars)
char_counts = Counter(gt_chars)
max_classes = 50
most_common_chars = [char for char, _ in char_counts.most_common(max_classes)]

print(f'Most common characters (max_classes={max_classes}): {most_common_chars}')

# Filter to most common characters
filtered_pred = []
filtered_gt = []
for p, g in zip(pred_chars, gt_chars):
    if g in most_common_chars:
        if p in most_common_chars:
            filtered_pred.append(p)
        elif p.strip() and p not in [' ', '']:
            filtered_pred.append('OTHER')
        else:
            filtered_pred.append(p)
        filtered_gt.append(g)

print(f'Filtered ground truth characters: {len(filtered_gt)}')
print(f'Filtered prediction characters: {len(filtered_pred)}')

# Create confusion matrix
unique_chars = sorted(set(filtered_gt + filtered_pred))
print(f'Unique characters for confusion matrix: {unique_chars}')
print(f'Number of unique characters: {len(unique_chars)}')

cm = confusion_matrix(filtered_gt, filtered_pred, labels=unique_chars)
print(f'Confusion matrix shape: {cm.shape}')
print(f'Expected shape based on unique_chars: ({len(unique_chars)}, {len(unique_chars)})')

# Check if downsampling would occur
nc = len(unique_chars)
if nc >= 100:
    k = max(2, nc // 60)
    unique_chars_downsampled = [unique_chars[i] for i in range(0, nc, k)]
    print(f'Downsampling would occur: k={k}, downsampled chars: {unique_chars_downsampled}')
else:
    unique_chars_downsampled = unique_chars
    print(f'No downsampling needed. unique_chars_downsampled = unique_chars')

print(f'Final labels for plot: {unique_chars_downsampled}')
print(f'Number of labels: {len(unique_chars_downsampled)}') 