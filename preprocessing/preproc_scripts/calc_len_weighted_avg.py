import os
import pandas as pd
from Bio import SeqIO
import argparse

def get_contig_lengths(fasta_path):
    """
    Parses a FASTA/FNA file and returns a dict of contig_id -> length.
    """
    contig_lengths = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        contig_id = record.id
        contig_lengths[contig_id] = len(record.seq)
    return contig_lengths

def compute_length_weighted_averages(df):
    """
    Computes the length-weighted average for each numeric column in the dataframe.
    """
    if 'length' not in df.columns or 'contig_ID' not in df.columns:
        raise ValueError("The dataframe must contain 'length' and 'contig_ID' columns.")

    # Get numeric columns (excluding contig_ID and length)
    feature_cols = df.select_dtypes(include=['number']).columns.difference(['length'])

    # Compute weighted averages
    weighted_averages = {}
    total_length = df['length'].sum()

    for col in feature_cols:
        weighted_averages[col] = (df[col] * df['length']).sum() / total_length

    return weighted_averages

def add_weighted_averages_to_features(feature_file, contig_lengths, output_path=None, contig_length_output_path=None):
    """
    Adds length-weighted averages to the output CSV, saves it, and writes contig lengths to a separate file.
    """
    df = pd.read_csv(feature_file)

    # Identify and rename the first column if it has no header
    first_col = df.columns[0]
    if first_col.startswith("nameseq"):
        print(f"No column name for contig IDs in {feature_file}. Assigning default name 'contig_ID'.")
        df.rename(columns={first_col: "contig_ID"}, inplace=True)
    else:
        df.rename(columns={first_col: "contig_ID"}, inplace=True)

    # Map contig lengths
    df['length'] = df['contig_ID'].map(contig_lengths)

    if df['length'].isnull().any():
        missing = df[df['length'].isnull()]['contig_ID'].tolist()
        raise ValueError(f"Missing contig lengths for: {missing}")

    # Compute length-weighted averages
    weighted_avg = compute_length_weighted_averages(df)

    # Create a new DataFrame to save the weighted averages as a single row
    weighted_avg_df = pd.DataFrame(weighted_avg, index=[0])

    # Save weighted averages
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        weighted_avg_df.to_csv(output_path, index=False)
    else:
        weighted_avg_df.to_csv(feature_file, index=False)

    # Save contig lengths to a separate file
    if contig_length_output_path:
        contig_lengths_df = pd.DataFrame(list(contig_lengths.items()), columns=["contig_ID", "length"])
        os.makedirs(os.path.dirname(contig_length_output_path), exist_ok=True)
        contig_lengths_df.to_csv(contig_length_output_path, index=False)

    print(f"Length-weighted averages for {feature_file}:")
    for col, avg in weighted_avg.items():
        print(f"   {col}: {avg:.4f}")

    print(f"Contig lengths saved to: {contig_length_output_path}")

def main(genome_dir, feature_dir, output_dir=None, contig_length_dir=None):
    """
    Processes all feature CSVs in feature_dir, mapping contig lengths from genome_dir,
    computing length-weighted averages, and saving contig lengths to a separate file.
    """
    for feature_file in os.listdir(feature_dir):
        if feature_file.endswith(".csv"):
            genome_id = os.path.splitext(feature_file)[0].replace("_Shannon", "")

            # Try to find the corresponding genome file in FASTA/FNA format
            possible_exts = [".fna", ".fasta", ".fa"]
            fasta_path = None
            for ext in possible_exts:
                candidate = os.path.join(genome_dir, genome_id + ext)
                if os.path.exists(candidate):
                    fasta_path = candidate
                    break

            if not fasta_path:
                print(f"Skipping {feature_file}: No matching genome FASTA found.")
                continue

            print(f"Updating: {feature_file} with lengths from {os.path.basename(fasta_path)}")

            contig_lengths = get_contig_lengths(fasta_path)

            # Define output file paths
            output_path = os.path.join(output_dir, feature_file) if output_dir else None
            contig_length_output_path = os.path.join(contig_length_dir, genome_id + "_contig_lengths.csv") if contig_length_dir else None

            add_weighted_averages_to_features(os.path.join(feature_dir, feature_file), contig_lengths, output_path, contig_length_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add length-weighted averages to feature CSVs and save contig lengths to a separate file.")
    parser.add_argument("genome_dir", help="Directory with genome FASTA/FNA files")
    parser.add_argument("feature_dir", help="Directory with per-contig feature CSV files")
    parser.add_argument("--output_dir", help="Directory to save updated CSV files (optional)", default=None)
    parser.add_argument("--contig_length_dir", help="Directory to save contig lengths CSV files (optional)", default=None)
    args = parser.parse_args()

    main(args.genome_dir, args.feature_dir, args.output_dir, args.contig_length_dir)
