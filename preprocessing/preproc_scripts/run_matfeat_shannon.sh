#!/bin/bash

# Usage/help function
usage() {
    echo "Usage: $0 -i INPUT_DIR -o OUTPUT_DIR -k KMER_SIZE -e ENTROPY_TYPE [-p PYTHON_EXEC] [-m SCRIPT_PATH]"
    echo ""
    echo "Arguments:"
    echo "  -i INPUT_DIR       Directory containing FASTA files"
    echo "  -o OUTPUT_DIR      Directory to store output CSVs"
    echo "  -k KMER_SIZE       Size of k-mers (e.g., 3)"
    echo "  -e ENTROPY_TYPE    Entropy type (e.g., Shannon, Tsallis, Renyi)"
    echo "  -p PYTHON_EXEC     Python executable to use (default: python3)"
    echo "  -m SCRIPT_PATH     Full path to EntropyClass.py"
    exit 1
}

# Defaults
PYTHON_EXEC="python3"
# SCRIPT_PATH="./EntropyClass.py"

# Parse arguments
while getopts ":i:o:k:e:p:m:" opt; do
  case $opt in
    i) INPUT_DIR="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    k) KMER_SIZE="$OPTARG" ;;
    e) ENTROPY_TYPE="$OPTARG" ;;
    p) PYTHON_EXEC="$OPTARG" ;;
    m) SCRIPT_PATH="$OPTARG" ;;
    *) usage ;;
  esac
done

# Validate required arguments
if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" || -z "$KMER_SIZE" || -z "$ENTROPY_TYPE" ]]; then
    usage
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Process each .fasta file
for file in "$INPUT_DIR"/*.fasta; do
    if [[ -f "$file" ]]; then
        filename=$(basename -- "$file")
        output_file="$OUTPUT_DIR/${filename%.fasta}.csv"
        echo "Processing $filename..."

        $PYTHON_EXEC "$SCRIPT_PATH" \
            -i "$file" \
            -o "$output_file" \
            -l "$filename" \
            -k "$KMER_SIZE" \
            -e "$ENTROPY_TYPE"
    fi
done

echo "Entropy-based feature extraction completed!"
