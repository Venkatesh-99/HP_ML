## Preprocessing Scripts

This folder contains scripts for preprocessing genome and variant data for downstream machine learning and analysis. Below is a summary of each script:

- **calc_len_weighted_avg.py**  
    Calculates length-weighted averages for per-contig feature CSVs and saves contig lengths.

- **extract_variant_annot.py**  
    Extracts and summarizes variant annotation statistics from VCF files, excluding synonymous variants.

- **ifeatomega.py**  
    Runs iFeatureOmegaCLI to extract sequence-based features from genome FASTA files.
    - **ifeatomega_features.txt**:  
        A plain text file listing the feature types to be extracted by `ifeatomega.py`. Each line should specify a feature type supported by iFeatureOmegaCLI (e.g., `Kmer type 1`, `NAC`, `Z_curve_144bit`). This file is provided as an argument to the script and determines which sequence-based features are generated for each genome.

- **run_bcftools_stats.py**  
    Summarizes variant statistics (SNPs, indels, transitions, etc.) from VCF files using `bcftools stats`.

- **run_matfeat_orf.sh**  
    Bash script to extract ORF-based features from input files using a Python script.

- **run_matfeat_preproc.sh**  
    Bash script to preprocess FASTA/FNA files using a specified Python script.

- **run_matfeat_shannon.sh**  
    Bash script to extract entropy-based features (e.g., Shannon entropy) from FASTA files.

- **run_snippy.sh**  
    Bash script to run Snippy for variant calling on multiple genomes against a reference.

- **vir_gene_matrix.py**  
    Detects presence/absence of specific genes in genome files using BLASTn and outputs a matrix.

    The data required for `vir_gene_matrix.py` should be placed in the `prepoc_data` folder. This includes:

    - **vir_genes.fasta**: A FASTA file containing the nucleotide sequences of the target genes to be detected. Each sequence header should correspond to a gene name.

    The `vir_genes.fasta` file is used as the query for BLASTn searches to determine the presence or absence of specific genes in genome FASTA files. Ensure that gene names provided to the script match the headers in this FASTA file.

Refer to each script's usage/help section for details on arguments and workflow.