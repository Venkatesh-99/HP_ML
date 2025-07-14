#!/usr/bin/env bash

# Usage function
usage() {
    echo "Usage: $0 -i INPUT_DIR -r REFERENCE_GENOME -o OUTPUT_DIR"
    echo ""
    echo "Arguments:"
    echo "  -i INPUT_DIR        Directory with genome files (.fasta/.fna)"
    echo "  -r REFERENCE_GENOME Reference genome in GenBank format (.gbk)"
    echo "  -o OUTPUT_DIR       Directory to save Snippy outputs"
    exit 1
}

# Parse arguments
while getopts ":i:r:o:" opt; do
    case $opt in
        i) INPUT_DIR="$OPTARG" ;;
        r) REF_GENOME="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        *) usage ;;
    esac
done

# Validate required inputs
if [[ -z "$INPUT_DIR" || -z "$REF_GENOME" || -z "$OUTPUT_DIR" ]]; then
    usage
fi

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Run Snippy for each genome
shopt -s nullglob
for genome in "$INPUT_DIR"/*.{fasta,fna}; do
    sample=$(basename "$genome" | sed 's/\..*//')
    echo "Running Snippy for: $sample"

    snippy --outdir "$OUTPUT_DIR/$sample" \
           --ref "$REF_GENOME" \
           --ctgs "$genome"
done

echo "Snippy variant calling completed!"
