import os
import subprocess
import pandas as pd
import argparse

def detect_gene_presence(
    genome_dir: str,
    gene_fasta: str,
    output_tsv: str,
    gene_names: list,
    blast_db_dir: str,
    blast_results_dir: str,
    identity_threshold: float = 90.0,
    coverage_threshold: float = 80.0,
    evalue_threshold: float = 1e-5
):
    os.makedirs(blast_db_dir, exist_ok=True)
    os.makedirs(blast_results_dir, exist_ok=True)

    # Map FASTA headers to gene names
    gene_id_map = {}
    with open(gene_fasta, "r") as f:
        for line in f:
            if line.startswith(">"):
                gene_name = line.strip().split()[0][1:]
                gene_id_map[gene_name] = gene_name

    results = {}

    for genome_file in os.listdir(genome_dir):
        if genome_file.endswith(".fasta") or genome_file.endswith(".fna"):
            genome_path = os.path.join(genome_dir, genome_file)
            genome_id = os.path.splitext(genome_file)[0]

            db_prefix = os.path.join(blast_db_dir, genome_id)

            if not os.path.exists(db_prefix + ".nhr"):
                subprocess.run(["makeblastdb", "-in", genome_path, "-dbtype", "nucl", "-out", db_prefix],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            blast_output = os.path.join(blast_results_dir, f"{genome_id}_blast_results.txt")
            subprocess.run(["blastn", "-query", gene_fasta, "-db", db_prefix,
                            "-task", "blastn",
                            "-outfmt", "6 qseqid pident qcovs evalue",
                            "-out", blast_output],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            gene_presence = {gene: 0 for gene in gene_names}

            with open(blast_output, "r") as blast_results:
                for line in blast_results:
                    query, pident, qcovs, evalue = line.strip().split("\t")
                    pident = float(pident)
                    qcovs = float(qcovs)
                    evalue = float(evalue)

                    if pident >= identity_threshold and qcovs >= coverage_threshold and evalue <= evalue_threshold:
                        if query in gene_id_map:
                            gene_name = gene_id_map[query]
                            gene_presence[gene_name] = 1

            results[genome_id] = gene_presence

    df = pd.DataFrame.from_dict(results, orient="index", columns=gene_names)
    df.index.name = "Genome_ID"
    df.to_csv(output_tsv, sep="\t")
    print(f"Gene presence/absence matrix saved to: {output_tsv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect presence/absence of specific genes in genome files using BLASTn"
    )

    parser.add_argument(
        "-g", "--genome_dir", required=True, help="Directory containing genome FASTA files (.fasta/.fna)"
    )
    parser.add_argument(
        "-q", "--gene_fasta", required=True, help="FASTA file containing query gene sequences"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output TSV file path for presence/absence matrix"
    )
    parser.add_argument(
        "-n", "--gene_names", required=True, nargs="+",
        help="List of gene names (should match headers in gene FASTA)"
    )
    parser.add_argument(
        "--blast_db_dir", required=True, help="Directory to store intermediate BLAST databases"
    )
    parser.add_argument(
        "--blast_results_dir", required=True, help="Directory to store intermediate BLAST result files"
    )
    parser.add_argument(
        "--identity", type=float, default=90.0, help="Minimum percent identity to count gene as present (default: 90)"
    )
    parser.add_argument(
        "--coverage", type=float, default=80.0, help="Minimum query coverage percent (default: 80)"
    )
    parser.add_argument(
        "--evalue", type=float, default=1e-5, help="Maximum acceptable E-value (default: 1e-5)"
    )

    args = parser.parse_args()

    detect_gene_presence(
        genome_dir=args.genome_dir,
        gene_fasta=args.gene_fasta,
        output_tsv=args.output,
        gene_names=args.gene_names,
        blast_db_dir=args.blast_db_dir,
        blast_results_dir=args.blast_results_dir,
        identity_threshold=args.identity,
        coverage_threshold=args.coverage,
        evalue_threshold=args.evalue
    )
