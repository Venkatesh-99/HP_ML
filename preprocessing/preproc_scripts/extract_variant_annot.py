"""Extracts variant annotation statistics from VCF files in a directory and summarizes the results."""

import os
import pandas as pd
from fuc import pyvcf, pysnpeff
import argparse

def extract_variant_annotation_stats(vcf_dir, output_file):
    # Step 1: Find all unique variant types across all VCF files
    all_variant_types = set()

    for filename in os.listdir(vcf_dir):
        if filename.endswith(".vcf") or filename.endswith(".vcf.gz"):
            vcf_path = os.path.join(vcf_dir, filename)
            vcf = pyvcf.VcfFrame.from_file(vcf_path)

            # Filter out synonymous variants
            filt_vcf = pysnpeff.filter_ann(vcf, targets=["synonymous_variant"], include=False)

            # Parse annotations
            parse_filt_vcf = pysnpeff.parseann(filt_vcf, idx=[1])
            
            # Get unique variant types in this file
            all_variant_types.update(parse_filt_vcf.unique())

    # Convert set to a sorted list for consistency
    variant_types = sorted(all_variant_types)

    # Step 2: Process all VCF files using the found variant types
    all_data = []

    for filename in os.listdir(vcf_dir):
        if filename.endswith(".vcf") or filename.endswith(".vcf.gz"):
            vcf_path = os.path.join(vcf_dir, filename)
            vcf = pyvcf.VcfFrame.from_file(vcf_path)

            # Filter out synonymous variants
            filt_vcf = pysnpeff.filter_ann(vcf, targets=["synonymous_variant"], include=False)

            # Parse annotations
            parse_filt_vcf = pysnpeff.parseann(filt_vcf, idx=[1])

            # Count occurrences of each variant type
            variant_counts = parse_filt_vcf.value_counts().to_dict()

            # Create a dictionary with filename and variant counts
            stats_data = {"Filename": filename}
            for vt in variant_types:
                stats_data[vt] = variant_counts.get(vt, 0)  # Default to 0 if missing
            
            all_data.append(stats_data)

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)
    print(f"Saved extracted variant annotation stats to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Extracts variant annotation statistics from VCF files in a directory and summarizes the results."
    )
    parser.add_argument(
        "--vcf_dir",
        help="Directory containing VCF files"
    )
    parser.add_argument(
        "--output_file",
        help="Path to output CSV file"
    )
    args = parser.parse_args()
    extract_variant_annotation_stats(args.vcf_dir, args.output_file)

if __name__ == "__main__":
    main()