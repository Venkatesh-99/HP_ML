#!/bin/bash

# Usage function
usage() {
    echo "Usage: $0 -i INPUT_DIR -o OUTPUT_DIR -l LABEL [-m SCRIPT_PATH] [-p PYTHON_EXEC]"
    echo "  -i INPUT_DIR     Directory containing input files (e.g., FASTA files)"
    echo "  -o OUTPUT_DIR    Directory to save output results"
    echo "  -l LABEL         Label to pass to the Python script"
    echo "  -m SCRIPT_PATH   Full path to the Python script"
    echo "  -p PYTHON_EXEC   Python executable (default: python3)"
    exit 1
}

# Default python executable
PYTHON_EXEC="python3"

# Parse arguments
while getopts ":i:o:l:p:" opt; do
  case $opt in
    i) INPUT_DIR="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    l) LABEL="$OPTARG" ;;
    m) SCRIPT_PATH="$OPTARG" ;;
    p) PYTHON_EXEC="$OPTARG" ;;
    *) usage ;;
  esac
done

# Validate required arguments
if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" || -z "$LABEL" ]]; then
    usage
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through all files in the input directory
for file in "$INPUT_DIR"/*; do
    if [[ -f "$file" ]]; then
        echo "Processing $file..."
        $PYTHON_EXEC "$SCRIPT_PATH" -i "$file" -o "$OUTPUT_DIR" -l "$LABEL"
    fi
done

echo "Processing completed! Results saved in $OUTPUT_DIR"
