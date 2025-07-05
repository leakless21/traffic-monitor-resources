#!/usr/bin/env python3
"""
Dataset Summary Generator

This script analyzes the license plate dataset and generates a comprehensive summary
including character frequency, plate length distribution, and other statistics.

Usage:
    python generate_dataset_summary.py --ground_truth lp_all_dataset/all_anotaciones.csv --output_dir dataset_summary
"""

import argparse
import csv
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

class DatasetAnalyzer:
    """Analyze license plate dataset and generate comprehensive statistics."""
    
    def __init__(self, normalize_text: bool = False):
        self.normalize_text = normalize_text
        self.ignore_chars = set(["-", ".", "_", " "])
        
    def normalize_plate_text(self, text: str) -> str:
        """Normalize plate text for analysis."""
        if not text:
            return ""
        
        # Remove ignored characters
        for char in self.ignore_chars:
            text = text.replace(char, "")
        
        # Case normalization
        text = text.upper()
        
        # Additional normalization if enabled
        if self.normalize_text:
            text = text.replace("O", "0")  # Letter O to digit 0
            text = text.replace("I", "1")  # Letter I to digit 1
            text = text.replace("S", "5")  # Sometimes S looks like 5
            
        return text.strip()
    
    def load_ground_truth(self, csv_path: Path) -> List[Dict]:
        """Load ground truth data from CSV file."""
        data = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row)
            print(f"Loaded {len(data)} records from {csv_path}")
        except Exception as e:
            print(f"Error loading CSV {csv_path}: {e}")
            raise
        return data
    
    def analyze_dataset(self, ground_truth: List[Dict]) -> Dict:
        """Analyze the dataset and return comprehensive statistics."""
        
        # Extract plate texts
        plate_texts = []
        for item in ground_truth:
            plate_text = item.get("plate_text", "").strip()
            if plate_text:
                normalized_text = self.normalize_plate_text(plate_text)
                if normalized_text:
                    plate_texts.append(normalized_text)
        
        # Basic statistics
        total_plates = len(plate_texts)
        unique_plates = len(set(plate_texts))
        
        # Character frequency analysis
        all_characters = []
        for plate in plate_texts:
            all_characters.extend(list(plate))
        
        char_frequency = Counter(all_characters)
        total_characters = len(all_characters)
        
        # Plate length distribution
        length_distribution = Counter(len(plate) for plate in plate_texts)
        
        # Position-based character analysis
        position_chars = defaultdict(Counter)
        max_length = max(len(plate) for plate in plate_texts) if plate_texts else 0
        
        for plate in plate_texts:
            for pos, char in enumerate(plate):
                position_chars[pos][char] += 1
        
        # Character type analysis
        digits = set('0123456789')
        letters = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        digit_count = sum(count for char, count in char_frequency.items() if char in digits)
        letter_count = sum(count for char, count in char_frequency.items() if char in letters)
        other_count = total_characters - digit_count - letter_count
        
        # Most common plates
        plate_frequency = Counter(plate_texts)
        most_common_plates = plate_frequency.most_common(20)
        
        # Prefix/suffix analysis
        prefix_2char = Counter(plate[:2] for plate in plate_texts if len(plate) >= 2)
        suffix_2char = Counter(plate[-2:] for plate in plate_texts if len(plate) >= 2)
        
        return {
            "basic_stats": {
                "total_plates": total_plates,
                "unique_plates": unique_plates,
                "duplicate_plates": total_plates - unique_plates,
                "total_characters": total_characters,
                "unique_characters": len(char_frequency),
                "average_plate_length": np.mean([len(plate) for plate in plate_texts]),
                "median_plate_length": np.median([len(plate) for plate in plate_texts]),
                "min_plate_length": min(len(plate) for plate in plate_texts) if plate_texts else 0,
                "max_plate_length": max_length
            },
            "character_frequency": dict(char_frequency.most_common()),
            "character_percentages": {
                char: (count / total_characters) * 100 
                for char, count in char_frequency.items()
            },
            "length_distribution": dict(length_distribution),
            "position_analysis": {
                pos: dict(chars.most_common()) 
                for pos, chars in position_chars.items()
            },
            "character_types": {
                "digits": digit_count,
                "letters": letter_count,
                "others": other_count,
                "digit_percentage": (digit_count / total_characters) * 100,
                "letter_percentage": (letter_count / total_characters) * 100,
                "other_percentage": (other_count / total_characters) * 100
            },
            "most_common_plates": most_common_plates,
            "prefix_analysis": dict(prefix_2char.most_common(20)),
            "suffix_analysis": dict(suffix_2char.most_common(20))
        }
    
    def generate_visualizations(self, stats: Dict, output_dir: Path):
        """Generate separate visualization plots for the dataset analysis."""
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. ALL Character Frequency Chart (separate image)
        chars = list(stats["character_frequency"].keys())
        frequencies = list(stats["character_frequency"].values())
        
        # Calculate appropriate figure size based on number of characters
        fig_width = max(12, len(chars) * 0.4)
        fig_height = max(8, len(chars) * 0.2)
        
        plt.figure(figsize=(fig_width, fig_height))
        bars = plt.bar(chars, frequencies, color='steelblue', alpha=0.8)
        plt.title('Complete Character Frequency Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Characters', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add frequency labels on bars (only for bars with significant height)
        max_freq = max(frequencies)
        for i, (char, freq) in enumerate(zip(chars, frequencies)):
            if freq > max_freq * 0.02:  # Only label bars with >2% of max frequency
                plt.text(i, freq + max_freq * 0.01, str(freq), 
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "character_frequency_all.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Plate Length Distribution (separate image)
        lengths = list(stats["length_distribution"].keys())
        length_counts = list(stats["length_distribution"].values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(lengths, length_counts, color='skyblue', alpha=0.8, edgecolor='navy')
        plt.title('Plate Length Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Plate Length (characters)', fontsize=12)
        plt.ylabel('Number of Plates', fontsize=12)
        
        # Add count and percentage labels on bars
        total_plates = sum(length_counts)
        for length, count in zip(lengths, length_counts):
            percentage = (count / total_plates) * 100
            plt.text(length, count + max(length_counts) * 0.01, 
                    f'{count}\n({percentage:.1f}%)', 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "plate_length_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Character Type Distribution (separate image)
        type_labels = ['Digits', 'Letters', 'Others']
        type_counts = [
            stats["character_types"]["digits"],
            stats["character_types"]["letters"],
            stats["character_types"]["others"]
        ]
        
        # Remove zero counts for cleaner pie chart
        non_zero_labels = []
        non_zero_counts = []
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        non_zero_colors = []
        
        for i, (label, count) in enumerate(zip(type_labels, type_counts)):
            if count > 0:
                non_zero_labels.append(label)
                non_zero_counts.append(count)
                non_zero_colors.append(colors[i])
        
        plt.figure(figsize=(10, 8))
        wedges, texts, autotexts = plt.pie(non_zero_counts, labels=non_zero_labels, 
                                          autopct='%1.1f%%', startangle=90,
                                          colors=non_zero_colors, explode=[0.05] * len(non_zero_counts))
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
        
        for text in texts:
            text.set_fontsize(14)
            text.set_fontweight('bold')
        
        plt.title('Character Type Distribution', fontsize=16, fontweight='bold', pad=20)
        
        # Add count information
        total_chars = sum(non_zero_counts)
        legend_labels = [f'{label}: {count:,} chars' for label, count in zip(non_zero_labels, non_zero_counts)]
        plt.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        plt.savefig(output_dir / "character_type_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Most Common Plates (separate image)
        if stats["most_common_plates"]:
            top_plates = stats["most_common_plates"][:15]  # Show top 15 instead of 10
            plate_names = [plate for plate, _ in top_plates]
            plate_counts = [count for _, count in top_plates]
            
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(plate_names)), plate_counts, color='lightcoral', alpha=0.8)
            plt.yticks(range(len(plate_names)), plate_names)
            plt.title('Most Common License Plates', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Frequency', fontsize=12)
            plt.ylabel('License Plate', fontsize=12)
            
            # Add count and percentage labels
            total_plates = stats['basic_stats']['total_plates']
            for i, count in enumerate(plate_counts):
                percentage = (count / total_plates) * 100
                plt.text(count + max(plate_counts) * 0.01, i, 
                        f'{count} ({percentage:.2f}%)', 
                        ha='left', va='center', fontsize=10, fontweight='bold')
            
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "most_common_plates.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Position-based Character Analysis Heatmap (separate image)
        if stats["position_analysis"]:
            max_pos = max(stats["position_analysis"].keys())
            all_chars = set()
            for pos_data in stats["position_analysis"].values():
                all_chars.update(pos_data.keys())
            
            all_chars = sorted(list(all_chars))
            
            # Create matrix for heatmap
            matrix = np.zeros((len(all_chars), max_pos + 1))
            
            for pos, char_counts in stats["position_analysis"].items():
                for char, count in char_counts.items():
                    char_idx = all_chars.index(char)
                    matrix[char_idx, pos] = count
            
            # Create heatmap
            plt.figure(figsize=(max(14, max_pos + 2), max(10, len(all_chars) * 0.5)))
            sns.heatmap(matrix, 
                       xticklabels=[f"Position {i}" for i in range(max_pos + 1)],
                       yticklabels=all_chars,
                       annot=True, 
                       fmt='g',
                       cmap='YlOrRd',
                       cbar_kws={'label': 'Frequency'},
                       linewidths=0.5)
            
            plt.title('Character Distribution by Position in License Plate', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Position in Plate', fontsize=12)
            plt.ylabel('Characters', fontsize=12)
            plt.tight_layout()
            plt.savefig(output_dir / "position_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 6. Character Frequency Comparison: Digits vs Letters (separate image)
        digits = {char: freq for char, freq in stats["character_frequency"].items() if char.isdigit()}
        letters = {char: freq for char, freq in stats["character_frequency"].items() if char.isalpha()}
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Digits subplot
        if digits:
            digit_chars = list(digits.keys())
            digit_freqs = list(digits.values())
            ax1.bar(digit_chars, digit_freqs, color='lightblue', alpha=0.8)
            ax1.set_title('Digit Frequency Distribution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Digits')
            ax1.set_ylabel('Frequency')
            
            # Add labels
            for char, freq in zip(digit_chars, digit_freqs):
                ax1.text(char, freq + max(digit_freqs) * 0.01, str(freq), 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Letters subplot
        if letters:
            letter_chars = list(letters.keys())
            letter_freqs = list(letters.values())
            ax2.bar(letter_chars, letter_freqs, color='lightgreen', alpha=0.8)
            ax2.set_title('Letter Frequency Distribution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Letters')
            ax2.set_ylabel('Frequency')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add labels for significant frequencies
            max_letter_freq = max(letter_freqs) if letter_freqs else 0
            for char, freq in zip(letter_chars, letter_freqs):
                if freq > max_letter_freq * 0.05:  # Only label significant frequencies
                    ax2.text(char, freq + max_letter_freq * 0.01, str(freq), 
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / "digits_vs_letters_frequency.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 7. Character Frequency Pie Chart (separate image)
        # Show top 15 characters in pie chart, group the rest as "Others"
        top_chars = list(stats["character_frequency"].items())[:15]
        other_chars_count = sum(freq for char, freq in list(stats["character_frequency"].items())[15:])
        
        # Prepare data for pie chart
        pie_labels = [char for char, freq in top_chars]
        pie_values = [freq for char, freq in top_chars]
        
        if other_chars_count > 0:
            pie_labels.append('Others')
            pie_values.append(other_chars_count)
        
        # Create pie chart
        plt.figure(figsize=(12, 10))
        
        # Create a color palette
        colors = plt.cm.Set3(np.linspace(0, 1, len(pie_labels)))
        
        wedges, texts, autotexts = plt.pie(pie_values, labels=pie_labels, 
                                          autopct='%1.1f%%', startangle=90,
                                          colors=colors, explode=[0.02] * len(pie_labels))
        
        # Enhance text formatting
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        
        plt.title('Character Frequency Distribution (Top 15 + Others)', fontsize=16, fontweight='bold', pad=20)
        
        # Add legend with counts
        total_chars = sum(pie_values)
        legend_labels = []
        for label, value in zip(pie_labels, pie_values):
            percentage = (value / total_chars) * 100
            legend_labels.append(f'{label}: {value:,} ({percentage:.1f}%)')
        
        plt.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / "character_frequency_pie.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_summary_report(self, stats: Dict, output_dir: Path):
        """Save detailed summary report to text and JSON files."""
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Text report
        report_lines = [
            "=" * 80,
            "LICENSE PLATE DATASET SUMMARY REPORT",
            "=" * 80,
            f"Generated: {timestamp}",
            f"Normalization: {'Enabled' if self.normalize_text else 'Disabled'}",
            "",
            "BASIC STATISTICS",
            "-" * 40,
            f"Total plates: {stats['basic_stats']['total_plates']:,}",
            f"Unique plates: {stats['basic_stats']['unique_plates']:,}",
            f"Duplicate plates: {stats['basic_stats']['duplicate_plates']:,}",
            f"Total characters: {stats['basic_stats']['total_characters']:,}",
            f"Unique characters: {stats['basic_stats']['unique_characters']}",
            f"Average plate length: {stats['basic_stats']['average_plate_length']:.2f}",
            f"Median plate length: {stats['basic_stats']['median_plate_length']:.1f}",
            f"Min plate length: {stats['basic_stats']['min_plate_length']}",
            f"Max plate length: {stats['basic_stats']['max_plate_length']}",
            "",
            "CHARACTER FREQUENCY (Top 20)",
            "-" * 40,
        ]
        
        # Add character frequency
        for i, (char, freq) in enumerate(list(stats["character_frequency"].items())[:20]):
            percentage = stats["character_percentages"][char]
            report_lines.append(f"{char:>3}: {freq:>6,} ({percentage:>5.2f}%)")
        
        # Add character types
        report_lines.extend([
            "",
            "CHARACTER TYPES",
            "-" * 40,
            f"Digits: {stats['character_types']['digits']:,} ({stats['character_types']['digit_percentage']:.1f}%)",
            f"Letters: {stats['character_types']['letters']:,} ({stats['character_types']['letter_percentage']:.1f}%)",
            f"Others: {stats['character_types']['others']:,} ({stats['character_types']['other_percentage']:.1f}%)",
            "",
            "PLATE LENGTH DISTRIBUTION",
            "-" * 40,
        ])
        
        for length, count in sorted(stats["length_distribution"].items()):
            percentage = (count / stats['basic_stats']['total_plates']) * 100
            report_lines.append(f"{length} chars: {count:>6,} plates ({percentage:>5.2f}%)")
        
        # Add most common plates
        if stats["most_common_plates"]:
            report_lines.extend([
                "",
                "MOST COMMON PLATES (Top 10)",
                "-" * 40,
            ])
            
            for plate, count in stats["most_common_plates"][:10]:
                percentage = (count / stats['basic_stats']['total_plates']) * 100
                report_lines.append(f"{plate:>10}: {count:>4} times ({percentage:>5.2f}%)")
        
        # Save text report
        with open(output_dir / "dataset_summary.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # Save JSON report
        json_stats = {
            "metadata": {
                "generated_at": timestamp,
                "normalization_enabled": self.normalize_text
            },
            "statistics": stats
        }
        
        with open(output_dir / "dataset_summary.json", 'w', encoding='utf-8') as f:
            json.dump(json_stats, f, indent=2, ensure_ascii=False)
        
        print(f"Summary reports saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive dataset summary with character frequency analysis")
    
    parser.add_argument(
        '--ground_truth',
        type=str,
        required=True,
        help='CSV file with ground truth annotations'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='dataset_summary',
        help='Output directory for summary files (default: dataset_summary)'
    )
    
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Apply text normalization (O->0, I->1, S->5)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    ground_truth_path = Path(args.ground_truth)
    if not ground_truth_path.exists():
        print(f"Error: Ground truth file does not exist: {ground_truth_path}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting dataset analysis...")
    print(f"Ground truth file: {ground_truth_path}")
    print(f"Output directory: {output_dir}")
    print(f"Text normalization: {'Enabled' if args.normalize else 'Disabled'}")
    
    # Initialize analyzer
    analyzer = DatasetAnalyzer(normalize_text=args.normalize)
    
    # Load and analyze data
    ground_truth = analyzer.load_ground_truth(ground_truth_path)
    stats = analyzer.analyze_dataset(ground_truth)
    
    # Generate outputs
    analyzer.generate_visualizations(stats, output_dir)
    analyzer.save_summary_report(stats, output_dir)
    
    print("\n" + "="*50)
    print("DATASET ANALYSIS COMPLETE")
    print("="*50)
    print(f"Total plates analyzed: {stats['basic_stats']['total_plates']:,}")
    print(f"Unique characters found: {stats['basic_stats']['unique_characters']}")
    print(f"Most common character: {list(stats['character_frequency'].keys())[0]} ({list(stats['character_frequency'].values())[0]:,} times)")
    print(f"Files generated in: {output_dir}")
    print("  - dataset_summary.txt (detailed text report)")
    print("  - dataset_summary.json (machine-readable data)")
    print("  - character_frequency_all.png (complete character frequency chart)")
    print("  - character_frequency_pie.png (character frequency pie chart)")
    print("  - plate_length_distribution.png (plate length analysis)")
    print("  - character_type_distribution.png (digits vs letters pie chart)")
    print("  - most_common_plates.png (top 15 most frequent plates)")
    print("  - position_analysis.png (character position heatmap)")
    print("  - digits_vs_letters_frequency.png (separate digit/letter frequency)")

if __name__ == "__main__":
    main() 