#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarizes variant statistics from VCF files using bcftools stats.

For each .vcf or .vcf.gz file in the input directory, it runs `bcftools stats`,
extracts SNPs, MNPs, indels, transitions, transversions, and base substitutions,
and writes a summary CSV.
"""

import os
import re
import pandas as pd
import subprocess
import argparse

def extract_stats(vcf_dir, stats_dir, output_file):
    os.makedirs(stats_dir, exist_ok=True)
    all_data = []

    for filename in os.listdir(vcf_dir):
        if filename.endswith(".vcf") or filename.endswith(".vcf.gz"):
            vcf_path = os.path.join(vcf_dir, filename)
            stats_filename = filename + ".stats"
            stats_path = os.path.join(stats_dir, stats_filename)

            # Run bcftools stats
            command = f"bcftools stats {vcf_path} > {stats_path}"
            subprocess.run(command, shell=True, check=True)

            stats_data = {
                "Filename": filename,
                "SNPs": 0,
                "MNPs": 0,
                "Indels": 0,
                "Transitions": 0,
                "Transversions": 0
            }
            substitutions = {}

            with open(stats_path, "r") as file:
                for line in file:
                    parts = line.strip().split("\t")

                    if line.startswith("SN"):
                        if "number of SNPs" in line:
                            stats_data["SNPs"] = int(re.findall(r'\d+', line)[-1])
                        elif "number of MNPs" in line:
                            stats_data["MNPs"] = int(re.findall(r'\d+', line)[-1])
                        elif "number of indels" in line:
                            stats_data["Indels"] = int(re.findall(r'\d+', line)[-1])

                    elif line.startswith("TSTV"):
                        numbers = re.findall(r'\d+', line)
                        if len(numbers) >= 3:
                            stats_data["Transitions"] = int(numbers[1])
                            stats_data["Transversions"] = int(numbers[2])

                    elif line.startswith("ST") and len(parts) == 4:
                        ref_alt = parts[2]
                        count = int(parts[3])
                        substitutions[ref_alt] = count

            stats_data.update(substitutions)
            all_data.append(stats_data)

    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)
    print(f"Saved extracted variant stats to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize variant statistics from VCF files using bcftools stats"
    )
    parser.add_argument(
        "-i", "--vcf_dir", required=True, help="Directory containing VCF or VCF.GZ files"
    )
    parser.add_argument(
        "-s", "--stats_dir", required=True, help="Directory to save intermediate bcftools stats output"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to output CSV file for summarized stats"
    )

    args = parser.parse_args()

    extract_stats(
        vcf_dir=args.vcf_dir,
        stats_dir=args.stats_dir,
        output_file=args.output
    )
