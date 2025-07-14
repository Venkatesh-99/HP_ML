#!/bin/bash

# Usage/help function
usage() {
    echo "Usage: $0 -i INPUT_DIR -o OUTPUT_DIR [-e EXTENSIONS] [-m SCRIPT_PATH] [-p PYTHON_EXEC]"
    echo ""
    echo "Arguments:"
    echo "  -i INPUT_DIR      Directory containing input FASTA/FNA files"
    echo "  -o OUTPUT_DIR     Directory to save preprocessed output files"
    echo "  -e EXTENSIONS     File extensions to process (default: fna,fasta)"
    echo "  -m SCRIPT_PATH    Full path to preprocessing script"
    echo "  -p PYTHON_EXEC    Python executable to use (default: python3)"
    exit 1
}

# Defaults
PYTHON_EXEC="python3"
EXTENSIONS="fna,fasta"

# Parse arguments
while getopts ":i:o:e:p:" opt; do
  case $opt in
    i) IN_DIR="$OPTARG" ;;
    o) OUT_DIR="$OPTARG" ;;
    e) EXTENSIONS="$OPTARG" ;;
    m) SCRIPT_PATH="$OPTARG" ;;
    p) PYTHON_EXEC="$OPTARG" ;;
    *) usage ;;
  esac
done

# Check required arguments
if [[ -z "$IN_DIR" || -z "$OUT_DIR" ]]; then
    usage
fi

# Ensure output directory exists
mkdir -p "$OUT_DIR"

# Convert comma-separated extensions into array
IFS=',' read -ra EXT_ARR <<< "$EXTENSIONS"

# Loop through all specified extensions
for ext in "${EXT_ARR[@]}"; do
    for file in "$IN_DIR"/*."$ext"; do
        if [ -f "$file" ]; then
            filename=$(basename -- "$file")
            name="${filename%.*}"
            out_file="$OUT_DIR/${name}_preproc.$ext"
            echo "Processing $filename -> $out_file"
            $PYTHON_EXEC "$SCRIPT_PATH" -i "$file" -o "$out_file"
        fi
    done
done

echo "Preprocessing completed! Outputs saved in $OUT_DIR"
