#!/usr/bin/env python3

import os
import sys
import iFeatureOmegaCLI as ifoc

import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract features using iFeatureOmega for a genome file.")
    parser.add_argument("-i", "--input", required=True, help="Path to genome FASTA file")
    parser.add_argument("-f", "--features", required=True, help="Path to file containing list of feature types (one per line)")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to store output feature CSVs")
    
    args = parser.parse_args()

    genome_file = args.input
    feature_file_path = args.features
    output_dir = args.output_dir

    genome_name = os.path.splitext(os.path.basename(genome_file))[0]
    genome_output_dir = os.path.join(output_dir, genome_name)
    os.makedirs(genome_output_dir, exist_ok=True)

    # Read feature types
    with open(feature_file_path, "r") as f:
        feature_types = [line.strip() for line in f if line.strip()]

    print(f"\nProcessing genome: {genome_name}")
    
    # Initialize DNA sequence
    try:
        dna = ifoc.iDNA(genome_file)
    except Exception as e:
        print(f"Failed to load genome: {e}")
        sys.exit(1)

    # Loop through each feature type
    for feature in feature_types:
        output_file = os.path.join(genome_output_dir, f"{genome_name}_{feature}.csv")
        
        if os.path.exists(output_file):
            print(f"Skipping {feature}: already exists.")
            continue

        try:
            print(f"Extracting: {feature}")
            dna.get_descriptor(feature)
            dna.to_csv(output_file, header=True, index=True)
            print(f"Saved: {output_file}")
        except Exception as e:
            print(f"Error with {feature}: {e}")

if __name__ == "__main__":
    main()
